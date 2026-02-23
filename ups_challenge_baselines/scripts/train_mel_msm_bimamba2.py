"""
Train masked spectrogram modeling (MSM) on precomputed mel shards using Bi-Mamba2.

Example:
  python scripts/train_mel_msm_bimamba2.py \
    --precompute_dir /content/drive/MyDrive/ups_precompute/cached_now_full_mel_20260202_220041 \
    --save_dir ./artifacts/msm_bimamba2 \
    --num_steps 200 \
    --batch_size 8
"""

import os

# Safety: disable torch compile/dynamo before importing torch.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import glob
import json
import math
import random
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ssd.modules import BiMamba2
except Exception as exc:
    raise ImportError(
        "Failed to import BiMamba2 from ssd.modules. "
        "Ensure the ssd package is installed and available on PYTHONPATH."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Masked spectrogram modeling with Bi-Mamba2 (from shards)")
    parser.add_argument("--precompute_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--lr_decay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--val_shards", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_batches", type=int, default=5)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--sampling_mode", type=str, default="natural", choices=["natural", "balanced", "hybrid"])
    parser.add_argument("--hybrid_balanced_frac", type=float, default=0.30)
    parser.add_argument("--lid_cache_path", type=str, default="")
    parser.add_argument("--lid_loss_weight", type=float, default=0.1)
    parser.add_argument("--vicreg_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_lang_samples", type=int, default=20)
    parser.add_argument("--max_en_frac", type=float, default=0.1)
    parser.add_argument("--shuffle_shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle_within_shard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kmeans_centroids_path", type=str, default=None)
    parser.add_argument("--num_clusters", type=int, default=256)
    parser.add_argument("--vicreg_var_weight", type=float, default=25.0)
    parser.add_argument("--vicreg_cov_weight", type=float, default=1.0)
    return parser.parse_args()


class BiMambaMSM(torch.nn.Module):
    def __init__(self, d_model: int, num_layers: int, discrete_mode: bool, num_clusters: int):
        super().__init__()
        self.discrete_mode = discrete_mode
        self.proj_in = torch.nn.Linear(80, d_model)
        self.backbone = nn.ModuleList([
            BiMamba2(d_model=d_model, d_state=16, d_conv=7, expand=2, use_mem_eff_path=False)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.input_norm = torch.nn.LayerNorm(d_model)
        self.lid_head = None
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        if self.discrete_mode:
            self.projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.classifier = nn.Linear(d_model, num_clusters)
            self.proj_out = None
        else:
            self.proj_out = torch.nn.Linear(d_model, 80)
            self.projector = None
            self.classifier = None

    def forward(self, x, masked_positions: Optional[torch.Tensor] = None):
        x = self.proj_in(x)
        if masked_positions is not None:
            n_masked = int(masked_positions.sum().item())
            if n_masked > 0:
                mask_token = self.mask_token.to(dtype=x.dtype).unsqueeze(0).expand(n_masked, -1)
                x = x.clone()
                x[masked_positions] = mask_token
        for layer, norm in zip(self.backbone, self.layer_norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = x + residual.to(x.dtype)
        hidden = self.final_norm(x)
        if self.discrete_mode:
            return None, hidden
        pred = self.proj_out(hidden)
        return pred, hidden


def sample_stream(shard_files: list[str], shuffle: bool, shuffle_within_shard: bool, seed: int):
    rng = random.Random(seed)
    shard_files = list(shard_files)
    while True:
        shard_list = list(shard_files)
        if shuffle:
            rng.shuffle(shard_list)
        for shard_path in shard_list:
            try:
                samples = torch.load(shard_path, map_location="cpu")
            except Exception:
                continue
            if not isinstance(samples, list):
                continue
            if shuffle_within_shard:
                rng.shuffle(samples)
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                yield sample


class ShardLRUCache:
    def __init__(self, max_items: int = 3):
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, shard_path: str):
        if shard_path in self.cache:
            self.cache.move_to_end(shard_path)
            return self.cache[shard_path]
        samples = torch.load(shard_path, map_location="cpu")
        self.cache[shard_path] = samples
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return samples


def load_lid_index(lid_cache_path: str, shard_files: list[str]):
    if not lid_cache_path or not os.path.exists(lid_cache_path):
        return None
    with open(lid_cache_path, "r") as f:
        data = json.load(f)
    shard_set = set(shard_files)
    index = {}
    for lid, refs in data.items():
        filtered = [ref for ref in refs if ref[0] in shard_set]
        if filtered:
            index[lid] = filtered
    return index if index else None


def build_lid_index(shard_files: list[str]):
    index = {}
    for shard_path in shard_files:
        try:
            samples = torch.load(shard_path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(samples, list):
            continue
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            lid = sample.get("lid")
            if lid is None:
                continue
            lid_key = str(lid)
            index.setdefault(lid_key, []).append([shard_path, i])
    return index


def make_batch(samples: list[dict]):
    mels = []
    kept_samples = []
    for sample in samples:
        mel = sample.get("mel") if isinstance(sample, dict) else None
        if mel is None or not isinstance(mel, torch.Tensor):
            continue
        if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < 1:
            continue
        mels.append(mel.float().cpu())
        kept_samples.append(sample)
    if not mels:
        return None
    seqs = [s.t().contiguous() for s in mels]  # [T, 80]
    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)

    batch_size = len(seqs)
    x = torch.zeros(batch_size, max_len, 80, dtype=torch.float32)
    pad_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        cur_len = seq.shape[0]
        x[i, :cur_len] = seq
        pad_mask[i, :cur_len] = True
    return x, pad_mask, torch.tensor(lengths, dtype=torch.long), kept_samples


def apply_time_mask(x: torch.Tensor, pad_mask: torch.Tensor, mask_ratio: float, rng: torch.Generator):
    batch_size, max_len, _ = x.shape
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=x.device)
    for i in range(batch_size):
        li = int(pad_mask[i].sum().item())
        if li < 1:
            continue
        num_to_mask = max(1, int(mask_ratio * li))
        # Block masking: 3-5 contiguous spans
        num_spans = random.randint(3, 5)
        span_len = max(1, num_to_mask // num_spans)
        masked = 0
        attempts = 0
        while masked < num_to_mask and attempts < 20:
            attempts += 1
            start = int(torch.randint(0, max(1, li - span_len), (1,), generator=rng, device=x.device).item())
            end = min(li, start + span_len)
            mask[i, start:end] = True
            masked = int(mask[i].sum().item())
    valid_mask = mask & pad_mask
    x_masked = x.clone()
    x_masked[valid_mask] = 0.0
    return x_masked, valid_mask


def vicreg_loss(z: torch.Tensor, gamma: float = 1.0, epsilon: float = 1e-4):
    """Variance + Covariance regularization on batch embeddings."""
    B, D = z.shape
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + epsilon)
    var_loss = torch.mean(F.relu(gamma - std))
    cov = (z.T @ z) / (B - 1)
    off_diag = cov * (1 - torch.eye(D, device=z.device))
    cov_loss = off_diag.pow(2).sum() / D
    return var_loss, cov_loss


def load_kmeans_centroids(
    path: str, device: torch.device
) -> dict[str, Optional[torch.Tensor]]:
    scaler_mean: Optional[torch.Tensor] = None
    scaler_scale: Optional[torch.Tensor] = None

    if path.endswith(".npz"):
        loaded = np.load(path)
        centroids_np = loaded["centroids"]
        if "scaler_mean" in loaded and "scaler_scale" in loaded:
            scaler_mean = torch.from_numpy(loaded["scaler_mean"].astype(np.float32, copy=False)).to(device)
            scaler_scale = torch.from_numpy(loaded["scaler_scale"].astype(np.float32, copy=False)).to(device)
    else:
        centroids_np = np.load(path)

    if centroids_np.ndim != 2 or centroids_np.shape[1] != 80:
        raise ValueError(f"Expected centroids shape [K, 80], got {centroids_np.shape}")
    centroids = torch.from_numpy(centroids_np.astype(np.float32, copy=False)).to(device)
    return {
        "centroids": centroids,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }


def get_kmeans_targets(
    mel_frames: torch.Tensor,
    centroids: torch.Tensor,
    scaler_mean: Optional[torch.Tensor] = None,
    scaler_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mel_frames.ndim != 2 or mel_frames.shape[1] != 80:
        raise ValueError(f"Expected mel_frames [N,80], got {tuple(mel_frames.shape)}")
    frames = mel_frames.float()
    if scaler_mean is not None:
        if scaler_scale is None:
            raise ValueError("scaler_scale must be provided when scaler_mean is provided")
        frames = (frames - scaler_mean.float()) / scaler_scale.float()
    dists = torch.cdist(frames, centroids.float())
    return dists.argmin(dim=-1).long()


@dataclass
class TrainLog:
    step: int
    loss: float
    lid_loss: float
    lr: float
    val_loss: Optional[float]
    elapsed_sec: float
    batch_lid_counts: dict
    recon_loss: float = 0.0
    vicreg_loss: float = 0.0
    discrete_loss: float = 0.0


def compute_lr(step: int, num_steps: int, base_lr: float, warmup_steps: int, lr_decay: bool) -> float:
    warmup_steps = max(1, warmup_steps)
    if step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    if not lr_decay:
        return base_lr

    min_lr = base_lr * 0.1
    decay_steps = max(1, num_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def save_checkpoint(
    save_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: dict,
):
    ckpt_path = os.path.join(save_dir, f"ckpt_step_{step}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "step": step,
            "cfg": cfg,
        },
        ckpt_path,
    )
    return ckpt_path


@torch.no_grad()
def run_val(
    model: torch.nn.Module,
    val_shard_files: list[str],
    batch_size: int,
    mask_ratio: float,
    device: torch.device,
    seed: int,
    val_batches: int,
    discrete_mode: bool,
    centroids: Optional[torch.Tensor],
    scaler_mean: Optional[torch.Tensor],
    scaler_scale: Optional[torch.Tensor],
    freq: int = 80,
):
    if not val_shard_files:
        return None
    if freq != 80:
        raise ValueError(f"Expected freq=80, got {freq}")

    model.eval()

    stream = sample_stream(
        shard_files=val_shard_files,
        shuffle=False,
        shuffle_within_shard=False,
        seed=seed,
    )

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    losses = []
    attempts = 0
    while len(losses) < val_batches and attempts < val_batches * 3:
        attempts += 1
        examples = [next(stream) for _ in range(batch_size)]
        batch = make_batch(examples)
        if batch is None:
            continue
        x, pad_mask, _lengths, _kept_samples = batch
        x = x.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        x_masked, valid_mask = apply_time_mask(x, pad_mask, mask_ratio, rng)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, hidden = model(x_masked, masked_positions=valid_mask)
            if discrete_mode:
                masked_hidden = hidden[valid_mask]
                if masked_hidden.numel() == 0:
                    continue
                logits = model.classifier(model.projector(masked_hidden))
                targets = get_kmeans_targets(x[valid_mask], centroids, scaler_mean, scaler_scale)
                loss = F.cross_entropy(logits, targets)
            else:
                loss = F.mse_loss(pred[valid_mask], x[valid_mask])

        losses.append(float(loss.detach().cpu().item()))

    model.train()

    if not losses:
        return None
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    discrete_mode = bool(args.kmeans_centroids_path)

    shard_dir = os.path.join(args.precompute_dir, "shards")
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
    if not shard_files:
        raise ValueError(f"No shard-*.pt files found in {shard_dir}")

    if args.val_shards <= 0:
        val_shard_files = []
        train_shard_files = shard_files
    else:
        rng_val = random.Random(args.seed)
        en_shards = [s for s in shard_files if os.path.basename(s).startswith("en_")]
        nonen_shards = [s for s in shard_files if not os.path.basename(s).startswith("en_")]

        target_total = min(args.val_shards, len(shard_files))
        target_en = min(2, target_total)
        target_nonen = min(8, max(0, target_total - target_en))

        selected_en = rng_val.sample(en_shards, k=min(target_en, len(en_shards)))
        selected_nonen = rng_val.sample(nonen_shards, k=min(target_nonen, len(nonen_shards)))

        selected_en_set = set(selected_en)
        selected_nonen_set = set(selected_nonen)
        remaining_en = [s for s in en_shards if s not in selected_en_set]
        remaining_nonen = [s for s in nonen_shards if s not in selected_nonen_set]

        # If either group is short, fill from the other group first.
        short_en = target_en - len(selected_en)
        if short_en > 0 and remaining_nonen:
            take = min(short_en, len(remaining_nonen))
            fill = rng_val.sample(remaining_nonen, k=take)
            selected_nonen.extend(fill)
            fill_set = set(fill)
            remaining_nonen = [s for s in remaining_nonen if s not in fill_set]

        short_nonen = target_nonen - len(selected_nonen)
        if short_nonen > 0 and remaining_en:
            take = min(short_nonen, len(remaining_en))
            fill = rng_val.sample(remaining_en, k=take)
            selected_en.extend(fill)
            fill_set = set(fill)
            remaining_en = [s for s in remaining_en if s not in fill_set]

        val_shard_files = selected_en + selected_nonen

        # Final top-up only if total val size is still below requested.
        remaining_needed = target_total - len(val_shard_files)
        if remaining_needed > 0:
            leftovers = [s for s in shard_files if s not in set(val_shard_files)]
            if leftovers:
                val_shard_files.extend(rng_val.sample(leftovers, k=min(remaining_needed, len(leftovers))))

        val_set = set(val_shard_files)
        train_shard_files = [s for s in shard_files if s not in val_set]
    print(f"train shards: {len(train_shard_files)} | val shards: {len(val_shard_files)}", flush=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training on CPU will be slow.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cfg = {
        **vars(args),
        "shard_dir": shard_dir,
        "device": str(device),
        "torch_version": torch.__version__,
        "discrete_mode": discrete_mode,
    }
    cfg_path = os.path.join(args.save_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    centroids = None
    scaler_mean = None
    scaler_scale = None
    if discrete_mode:
        kmeans_artifacts = load_kmeans_centroids(args.kmeans_centroids_path, device=device)
        centroids = kmeans_artifacts["centroids"]
        scaler_mean = kmeans_artifacts["scaler_mean"]
        scaler_scale = kmeans_artifacts["scaler_scale"]
        if args.num_clusters != int(centroids.shape[0]):
            raise ValueError(
                f"num_clusters ({args.num_clusters}) does not match centroid count ({centroids.shape[0]})"
            )

    model = BiMambaMSM(
        d_model=args.d_model,
        num_layers=args.num_layers,
        discrete_mode=discrete_mode,
        num_clusters=args.num_clusters,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    lid_index = load_lid_index(args.lid_cache_path, train_shard_files)
    if lid_index is None:
        lid_index = build_lid_index(train_shard_files)
        if args.lid_cache_path:
            os.makedirs(os.path.dirname(args.lid_cache_path) or ".", exist_ok=True)
            with open(args.lid_cache_path, "w") as f:
                json.dump(lid_index, f)

    lid2idx = {}
    if lid_index:
        lid_ids = sorted(set(lid_index.keys()))
        lid2idx = {lid: idx for idx, lid in enumerate(lid_ids)}
        if lid2idx:
            model.lid_head = torch.nn.Linear(args.d_model, len(lid2idx)).to(device)
            optimizer.add_param_group({"params": model.lid_head.parameters(), "lr": args.lr * 20})

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", {}), strict=False)
        try:
            optimizer.load_state_dict(ckpt.get("optimizer_state", {}))
        except (ValueError, KeyError):
            print("Warning: optimizer state not loaded (parameter group mismatch), starting optimizer fresh.")
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except (ValueError, KeyError):
            pass
        start_step = int(ckpt.get("step", 0))
        last_good_ckpt_path = args.resume_from
        print(f"Resumed from {args.resume_from} at step {start_step}", flush=True)
    else:
        start_step = 0
        last_good_ckpt_path = ""

    train_stream = sample_stream(
        shard_files=train_shard_files,
        shuffle=args.shuffle_shards,
        shuffle_within_shard=args.shuffle_within_shard,
        seed=args.seed + start_step,
    )
    sampling_rng = random.Random(args.seed + 7)
    lid_cache = ShardLRUCache(max_items=500)

    if lid_index and args.min_lang_samples > 0:
        before = len(lid_index)
        lid_index = {k: v for k, v in lid_index.items() if len(v) >= args.min_lang_samples}
        dropped = before - len(lid_index)
        if dropped:
            print(
                f"Filtered {dropped} languages with <{args.min_lang_samples} samples. Keeping {len(lid_index)} languages.",
                flush=True,
            )
    lid_keys = list(lid_index.keys()) if lid_index else []

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    start_time = time.time()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)
    effective_lid_weight = args.lid_loss_weight

    for step in range(start_step + 1, args.num_steps + 1):
        examples = []
        for _ in range(args.batch_size):
            use_balanced = args.sampling_mode == "balanced"
            if args.sampling_mode == "hybrid":
                use_balanced = sampling_rng.random() < args.hybrid_balanced_frac

            if use_balanced and lid_index and lid_keys:
                lid = sampling_rng.choice(lid_keys)
                refs = lid_index.get(lid, [])
                if refs:
                    shard_fp, idx = sampling_rng.choice(refs)
                    try:
                        shard_samples = lid_cache.get(shard_fp)
                        if isinstance(shard_samples, list) and 0 <= idx < len(shard_samples):
                            examples.append(shard_samples[idx])
                            continue
                    except Exception:
                        pass
            examples.append(next(train_stream))

        if args.max_en_frac < 1.0:
            en_samples = [s for s in examples if isinstance(s, dict) and str(s.get("lid", "")) == "en"]
            non_en = [s for s in examples if isinstance(s, dict) and str(s.get("lid", "")) != "en"]
            max_en = max(1, int(len(examples) * args.max_en_frac))
            examples = non_en + en_samples[:max_en]

        batch = make_batch(examples)
        if batch is None:
            if step % args.log_every == 0:
                print(f"Step {step}: skipped (empty batch)", flush=True)
            if step % args.ckpt_every == 0:
                last_good_ckpt_path = save_checkpoint(args.save_dir, step, model, optimizer, scaler, cfg)
            continue

        x, pad_mask, _lengths, kept_samples = batch
        x = x.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        x_masked, valid_mask = apply_time_mask(x, pad_mask, args.mask_ratio, rng)
        x_masked = torch.clamp(x_masked, min=-20.0, max=20.0)
        if torch.isnan(x_masked).any():
            print(f"WARNING: NaN detected in x_masked at step {step}; skipping batch.", flush=True)
            continue

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, hidden = model(x_masked, masked_positions=valid_mask)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)

            if discrete_mode:
                masked_hidden = hidden[valid_mask]
                if masked_hidden.numel() == 0:
                    continue
                logits = model.classifier(model.projector(masked_hidden))
                targets = get_kmeans_targets(x[valid_mask], centroids, scaler_mean, scaler_scale)
                task_loss = F.cross_entropy(logits, targets)
                recon_loss = torch.tensor(0.0, device=device)
                vicreg = torch.tensor(0.0, device=device)
                var_loss = torch.tensor(0.0, device=device)
                cov_loss = torch.tensor(0.0, device=device)
            else:
                if pred is None or not torch.isfinite(pred).all() or not torch.isfinite(hidden).all():
                    print(f"WARNING: NaN/Inf in model output at step {step}; restoring checkpoint.", flush=True)
                    if last_good_ckpt_path:
                        restore_ckpt = torch.load(last_good_ckpt_path, map_location=device)
                        model.load_state_dict(restore_ckpt.get("model_state", {}), strict=False)
                        optimizer.load_state_dict(restore_ckpt.get("optimizer_state", {}))
                        try:
                            scaler.load_state_dict(restore_ckpt["scaler_state"])
                        except (ValueError, KeyError):
                            pass
                        print("Restored from last good checkpoint", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                denom_v = pad_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
                z_pooled = (hidden * pad_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom_v
                z_pooled = z_pooled.clamp(-100.0, 100.0)
                var_loss, cov_loss = vicreg_loss(z_pooled)
                vicreg = args.vicreg_var_weight * var_loss + args.vicreg_cov_weight * cov_loss

                if torch.isnan(hidden).any():
                    kept_lids = [
                        str(sample.get("lid"))
                        for sample in kept_samples
                        if isinstance(sample, dict) and sample.get("lid") is not None
                    ]
                    print(
                        f"WARNING: NaN detected at step {step}; skipping batch. batch_lids={kept_lids}",
                        flush=True,
                    )
                    if last_good_ckpt_path:
                        restore_ckpt = torch.load(last_good_ckpt_path, map_location=device)
                        model.load_state_dict(restore_ckpt.get("model_state", {}), strict=False)
                        optimizer.load_state_dict(restore_ckpt.get("optimizer_state", {}))
                        try:
                            scaler.load_state_dict(restore_ckpt["scaler_state"])
                        except (ValueError, KeyError):
                            pass
                        print("Restored from last good checkpoint", flush=True)
                    continue

                pred = pred.clamp(-50.0, 50.0)
                recon_loss = F.mse_loss(pred[valid_mask], x[valid_mask])
                task_loss = recon_loss

            lid_ce_loss = torch.tensor(0.0, device=device)
            if model.lid_head is not None and kept_samples:
                denom = pad_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
                pooled = (hidden * pad_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
                lid_logits = model.lid_head(pooled)
                valid_rows = []
                lid_targets = []
                for i, sample in enumerate(kept_samples):
                    if not isinstance(sample, dict):
                        continue
                    lid = sample.get("lid")
                    if lid is None:
                        continue
                    lid_idx = lid2idx.get(str(lid))
                    if lid_idx is None:
                        continue
                    valid_rows.append(i)
                    lid_targets.append(lid_idx)
                if valid_rows:
                    row_idx = torch.tensor(valid_rows, device=device, dtype=torch.long)
                    target = torch.tensor(lid_targets, device=device, dtype=torch.long)
                    lid_ce_loss = F.cross_entropy(lid_logits.index_select(0, row_idx), target)

            if not torch.isfinite(task_loss) or task_loss.item() > 200:
                print(f"WARNING: loss spike ({task_loss.item():.1f}) at step {step}; skipping.", flush=True)
                if last_good_ckpt_path:
                    restore_ckpt = torch.load(last_good_ckpt_path, map_location=device)
                    model.load_state_dict(restore_ckpt.get("model_state", {}), strict=False)
                    optimizer.load_state_dict(restore_ckpt.get("optimizer_state", {}))
                    try:
                        scaler.load_state_dict(restore_ckpt["scaler_state"])
                    except (ValueError, KeyError):
                        pass
                    print("Restored from last good checkpoint", flush=True)
                continue

            if discrete_mode:
                total_loss = task_loss + (effective_lid_weight * lid_ce_loss)
            else:
                total_loss = task_loss + (effective_lid_weight * lid_ce_loss) + (args.vicreg_weight * vicreg)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "A_log"):
                    module.A_log.clamp_(-10.0, 2.0)
                if hasattr(module, "dt_bias"):
                    module.dt_bias.clamp_(-5.0, 5.0)

        lid_decay_start = start_step
        lid_decay_steps = 50000
        progress = max(0.0, min(1.0, (step - lid_decay_start) / lid_decay_steps))
        effective_lid_weight = args.lid_loss_weight * (1.0 - 0.9 * progress)

        current_lr = compute_lr(
            step=step,
            num_steps=args.num_steps,
            base_lr=args.lr,
            warmup_steps=args.warmup_steps,
            lr_decay=args.lr_decay,
        )
        optimizer.param_groups[0]["lr"] = current_lr
        for group in optimizer.param_groups[1:]:
            group["lr"] = current_lr * 20

        loss_val = float(total_loss.detach().cpu().item())
        lid_loss_val = float(lid_ce_loss.detach().cpu().item())
        discrete_loss_val = float(task_loss.detach().cpu().item()) if discrete_mode else 0.0
        recon_loss_val = float(task_loss.detach().cpu().item()) if not discrete_mode else 0.0
        vicreg_loss_val = float(vicreg.detach().cpu().item()) if not discrete_mode else 0.0
        elapsed = time.time() - start_time

        batch_lid_counts = Counter()
        for sample in examples:
            if isinstance(sample, dict) and "lid" in sample:
                batch_lid_counts[str(sample["lid"])] += 1

        if step % args.log_every == 0:
            if val_shard_files and step % args.val_every == 0:
                val_loss = run_val(
                    model,
                    val_shard_files=val_shard_files,
                    batch_size=args.batch_size,
                    mask_ratio=args.mask_ratio,
                    device=device,
                    seed=args.seed,
                    val_batches=args.val_batches,
                    discrete_mode=discrete_mode,
                    centroids=centroids,
                    scaler_mean=scaler_mean,
                    scaler_scale=scaler_scale,
                )
            else:
                val_loss = None

            log_payload = {
                "step": step,
                "loss": loss_val,
                "lid_loss": lid_loss_val,
                "lr": current_lr,
                "val_loss": val_loss,
                "elapsed_sec": elapsed,
                "batch_lid_counts": dict(batch_lid_counts),
            }
            if discrete_mode:
                log_payload["discrete_loss"] = discrete_loss_val
            else:
                log_payload["recon_loss"] = recon_loss_val
                log_payload["vicreg_loss"] = vicreg_loss_val
                log_payload["var_loss"] = float(var_loss.detach().cpu().item())
                log_payload["cov_loss"] = float(cov_loss.detach().cpu().item())

            print(log_payload, flush=True)
        else:
            val_loss = None

        log_entry = TrainLog(
            step=step,
            loss=loss_val,
            lid_loss=lid_loss_val,
            lr=current_lr,
            val_loss=val_loss,
            elapsed_sec=elapsed,
            batch_lid_counts=dict(batch_lid_counts),
            recon_loss=recon_loss_val,
            vicreg_loss=vicreg_loss_val,
            discrete_loss=discrete_loss_val,
        )
        with open(log_path, "a") as f:
            f.write(json.dumps(asdict(log_entry)) + "\n")

        if step % args.ckpt_every == 0:
            last_good_ckpt_path = save_checkpoint(args.save_dir, step, model, optimizer, scaler, cfg)

    save_checkpoint(args.save_dir, args.num_steps, model, optimizer, scaler, cfg)


if __name__ == "__main__":
    main()
