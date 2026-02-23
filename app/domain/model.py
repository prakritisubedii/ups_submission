import base64, io, sys, tempfile, wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# Works both in a real .py file (Dynabench) and in Colab notebooks
ZIP_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ZIP_ROOT / "ups_challenge_baselines"))

BiMambaMSM = None
CUDA_AVAILABLE = torch.cuda.is_available()
BIMAMBA_IMPORT_ERROR = None
try:
    from scripts.train_mel_msm_bimamba2 import BiMambaMSM  # noqa: E402
except Exception as e:
    BiMambaMSM = None
    BIMAMBA_IMPORT_ERROR = e

# ── Architecture constants matching our v27 training run ──────────────────────
# d_model=512, num_layers=8, discrete_mode=True, num_clusters=256
# These are used as fallback defaults if cfg is missing from the checkpoint.
_DEFAULT_D_MODEL = 512
_DEFAULT_NUM_LAYERS = 8
_DEFAULT_NUM_CLUSTERS = 256


class ModelController:
    SR = 16000
    N_FFT = 400
    HOP = 160
    N_MELS = 80
    CHUNK_SEC = 10.0
    EPS = 1e-6
    VALID_FRAME_THRESHOLD = -19.5
    # OUTPUT_DIM matches d_model=512. Do NOT pad zeros — they hurt cosine similarity.
    OUTPUT_DIM = 512

    @staticmethod
    def _decode_wav_bytes_stdlib(wav_bytes: bytes) -> (torch.Tensor, int):
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            raw = wf.readframes(num_frames)

        if sample_width == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 3:
            a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (
                a[:, 0].astype(np.int32)
                | (a[:, 1].astype(np.int32) << 8)
                | (a[:, 2].astype(np.int32) << 16)
            )
            sign = signed & 0x800000
            signed = signed - (sign << 1)
            audio = signed.astype(np.float32) / 8388608.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)
        wav = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)  # [1, N]
        return wav, sample_rate

    @staticmethod
    def _load_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
        """Load checkpoint robustly across torch versions (incl. 2.6 weights_only change)."""
        load_kwargs = {"map_location": "cpu"}
        torch_version_cls = None
        try:
            torch_version_cls = torch.torch_version.TorchVersion
        except Exception:
            torch_version_cls = None

        try:
            if torch_version_cls is not None and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([torch_version_cls]):
                    return torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
            return torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
        except Exception:
            try:
                return torch.load(str(ckpt_path), weights_only=False, **load_kwargs)
            except TypeError:
                return torch.load(str(ckpt_path), **load_kwargs)

    def __init__(self, device: str = "cuda" if CUDA_AVAILABLE else "cpu") -> None:
        # ---------------------------------------------------------------------
        # STRICT DEBUG PATCH:
        # Enforce real BiMamba CUDA loading and fail loudly on any fallback path
        # so Dynabench runs cannot silently use a different backend.
        # ---------------------------------------------------------------------
        self.model = None
        self.backend = "bimamba_cuda"
        self.device = torch.device("cuda")

        ckpt_path = ZIP_ROOT / "app" / "resources" / "ckpt_step_20000.pt"
        ckpt_exists = ckpt_path.exists()
        print(f"[UPS_DEBUG] BiMambaMSM import succeeded: {BiMambaMSM is not None}", flush=True)
        print(f"[UPS_DEBUG] Selected device: {self.device}", flush=True)
        print(f"[UPS_DEBUG] Backend: {self.backend}", flush=True)
        print(f"[UPS_DEBUG] Checkpoint path: {ckpt_path}", flush=True)
        print(f"[UPS_DEBUG] Checkpoint exists: {ckpt_exists}", flush=True)

        if BiMambaMSM is None:
            raise RuntimeError(
                f"[UPS_DEBUG] Failed to import BiMambaMSM: {BIMAMBA_IMPORT_ERROR}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("[UPS_DEBUG] CUDA is unavailable; this debug build requires CUDA.")
        if not ckpt_exists:
            raise FileNotFoundError(str(ckpt_path))

        ckpt = self._load_checkpoint(ckpt_path)
        if not isinstance(ckpt, dict):
            raise RuntimeError(
                f"[UPS_DEBUG] Invalid checkpoint format at {ckpt_path}: expected dict, got {type(ckpt)}"
            )

        cfg = ckpt.get("cfg", {})
        # FIX: defaults now match our actual training config, not the old 128/1 values
        d_model = int(cfg.get("d_model", _DEFAULT_D_MODEL))
        num_layers = int(cfg.get("num_layers", _DEFAULT_NUM_LAYERS))
        num_clusters = int(cfg.get("num_clusters", _DEFAULT_NUM_CLUSTERS))
        self.d_model = d_model
        print(
            f"[UPS_DEBUG] Wrapper config: d_model={d_model}, output_dim={self.OUTPUT_DIM}, "
            f"num_layers={num_layers}, num_clusters={num_clusters}",
            flush=True,
        )

        state = ckpt.get("model_state", {})
        if not isinstance(state, dict) or len(state) == 0:
            raise RuntimeError(
                f"[UPS_DEBUG] Checkpoint at {ckpt_path} has empty or invalid model_state."
            )
        has_expected_backbone_key = any(
            isinstance(k, str)
            and (k.startswith("backbone.") or k.startswith("module.backbone."))
            for k in state.keys()
        )
        if not has_expected_backbone_key:
            sample_keys = list(state.keys())[:10]
            raise RuntimeError(
                "[UPS_DEBUG] Checkpoint model_state appears incompatible: "
                f"no expected backbone keys found. Sample keys: {sample_keys}"
            )

        try:
            # pass discrete_mode=True and num_clusters — required by constructor.
            self.model = BiMambaMSM(
                d_model=d_model,
                num_layers=num_layers,
                discrete_mode=True,
                num_clusters=num_clusters,
            ).to(self.device)
            # lid_head may be present in checkpoint even with lid_loss_weight=0.0
            # because the training script builds it whenever lid_index is non-empty.
            if "lid_head.weight" in state:
                n_lids = state["lid_head.weight"].shape[0]
                self.model.lid_head = torch.nn.Linear(d_model, n_lids).to(self.device)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(
                f"[UPS_DEBUG] load_state_dict summary: missing_keys={len(missing)}, "
                f"unexpected_keys={len(unexpected)}",
                flush=True,
            )
            if missing:
                print(
                    f"[UPS_DEBUG] load_state_dict missing keys (first 5): {missing[:5]}",
                    flush=True,
                )
            if unexpected:
                print(
                    f"[UPS_DEBUG] load_state_dict unexpected keys (first 5): {unexpected[:5]}",
                    flush=True,
                )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"[UPS_DEBUG] Failed to construct/load BiMamba CUDA model: {e}") from e

        self.initialized = True

        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            n_mels=self.N_MELS,
            power=2.0,
        )
        self._inference_debug_printed = False

        print(
            f"[UPS_DEBUG] initialized | backend={self.backend} | device={self.device} "
            f"| d_model={d_model} | num_layers={num_layers} | num_clusters={num_clusters}",
            flush=True,
        )

    def single_inference(self, input_data: Any, sample_rate: int = 16000) -> Any:
        """Run inference on a single audio sample.

        Args:
            input_data: 1D torch.Tensor of audio samples [T], or a dict payload,
                        or a base64-encoded WAV string.
            sample_rate: Sample rate of the provided audio.

        Returns:
            torch.Tensor: L2-normalised embedding of shape [OUTPUT_DIM] on CPU.
        """
        # Backward-compatible dict payload path
        if isinstance(input_data, dict):
            return self.single_evaluation(input_data)

        if isinstance(input_data, str):
            wav = self._decode_wav_b64(input_data)
            with torch.no_grad():
                x = self._wav_to_logmel_bt80(wav)      # [1, T, 80]
                reps = self._extract_backbone_reps(x)  # [1, T, d_model]
                reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
                emb, valid_frames, total_frames = self._masked_mean_pool(reps, x)
                emb = self._project_to_output_dim(emb)
                emb = F.normalize(emb, dim=-1)
                self._maybe_print_inference_debug(
                    wav_len_samples=wav.size(-1),
                    mel_shape=tuple(x.shape),
                    hidden_shape=tuple(reps.shape),
                    valid_frames=valid_frames,
                    total_frames=total_frames,
                    emb_norm=float(emb.norm(p=2).item()),
                )
                return emb.detach().cpu().float()

        if not torch.is_tensor(input_data):
            raise TypeError("input_data must be a torch.Tensor (audio), a dict, or a base64 str")

        if input_data.dim() == 2 and input_data.size(0) == 1:
            input_data = input_data.squeeze(0)
        if input_data.dim() != 1:
            raise ValueError(
                f"input_data must be 1D (shape [T]); got shape={tuple(input_data.shape)}"
            )

        with torch.no_grad():
            wav = input_data.detach().to("cpu", dtype=torch.float32).unsqueeze(0)  # [1, T]
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
            if sample_rate != self.SR:
                wav = torchaudio.functional.resample(wav, sample_rate, self.SR)

            x = self._wav_to_logmel_bt80(wav)          # [1, T, 80]
            reps = self._extract_backbone_reps(x)      # [1, T, d_model]
            reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
            emb, valid_frames, total_frames = self._masked_mean_pool(reps, x)
            emb = self._project_to_output_dim(emb)
            emb = F.normalize(emb, dim=-1)
            self._maybe_print_inference_debug(
                wav_len_samples=wav.size(-1),
                mel_shape=tuple(x.shape),
                hidden_shape=tuple(reps.shape),
                valid_frames=valid_frames,
                total_frames=total_frames,
                emb_norm=float(emb.norm(p=2).item()),
            )
            return emb.detach().cpu().float()

    def _decode_wav_b64(self, wav_b64: str) -> torch.Tensor:
        wav_bytes = base64.b64decode(wav_b64)
        try:
            wav, in_sr = torchaudio.load(io.BytesIO(wav_bytes))  # [C, N]
        except Exception:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    f.write(wav_bytes)
                    f.flush()
                    wav, in_sr = torchaudio.load(f.name)
            except Exception:
                wav, in_sr = self._decode_wav_bytes_stdlib(wav_bytes)
        wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if in_sr != self.SR:
            wav = torchaudio.functional.resample(wav, in_sr, self.SR)
        return wav  # [1, N]

    def _pad_or_crop_10s(self, wav: torch.Tensor) -> torch.Tensor:
        target_len = int(self.SR * self.CHUNK_SEC)
        n = wav.size(-1)
        if n < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - n))
        elif n > target_len:
            wav = wav[..., :target_len]
        return wav

    def _wav_to_logmel_bt80(self, wav: torch.Tensor) -> torch.Tensor:
        # Keep variable-length audio; only pad minimally for stable STFT on tiny clips.
        if wav.size(-1) < self.N_FFT:
            wav = torch.nn.functional.pad(wav, (0, self.N_FFT - wav.size(-1)))
        wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        wav = torch.clamp(wav, min=-1.0, max=1.0)
        mel = self.mel_fn(wav)  # [1, 80, T]
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = torch.log(torch.clamp(mel, min=self.EPS))
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = torch.clamp(mel, min=-20.0, max=20.0)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0).contiguous()  # [1, T, 80]
        return mel.to(self.device, dtype=torch.float32)

    def _extract_backbone_reps(self, x_bt80: torch.Tensor) -> torch.Tensor:
        """Forward through backbone; returns final_norm hidden states [1, T, d_model].

        In discrete_mode the model forward returns (None, hidden).
        We use hidden (final_norm output), NOT the classifier/projector outputs —
        those are training-time targets only.
        """
        with torch.no_grad():
            if self.backend != "bimamba_cuda" or self.model is None:
                raise RuntimeError(
                    "[UPS_DEBUG] Invalid backend state during inference; expected bimamba_cuda."
                )
            x_bt80 = x_bt80.to(
                device=next(self.model.parameters()).device,
                dtype=torch.float32,
            ).contiguous()
            _, hidden = self.model(x_bt80)  # discrete_mode: (None, hidden)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
            return hidden  # [1, T, d_model]

    def _project_to_output_dim(self, emb: torch.Tensor) -> torch.Tensor:
        """Truncate or pad embedding to OUTPUT_DIM.

        With d_model=512 and OUTPUT_DIM=512 this is a no-op.
        Only pads/truncates if checkpoint uses a different d_model than OUTPUT_DIM.
        """
        if emb.numel() == self.OUTPUT_DIM:
            return emb.contiguous()
        if emb.numel() > self.OUTPUT_DIM:
            return emb[: self.OUTPUT_DIM].contiguous()
        return F.pad(emb, (0, self.OUTPUT_DIM - emb.numel()))

    def _masked_mean_pool(self, reps_btD: torch.Tensor, x_bt80: torch.Tensor) -> (torch.Tensor, int, int):
        frame_energy = x_bt80.squeeze(0).mean(dim=-1)  # [T]
        valid_mask = frame_energy > self.VALID_FRAME_THRESHOLD
        if not bool(valid_mask.any()):
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        reps_tD = reps_btD.squeeze(0)
        pooled = reps_tD[valid_mask].mean(dim=0)
        return pooled, int(valid_mask.sum().item()), int(valid_mask.numel())

    def _maybe_print_inference_debug(
        self,
        wav_len_samples: int,
        mel_shape: tuple,
        hidden_shape: tuple,
        valid_frames: int,
        total_frames: int,
        emb_norm: float,
    ) -> None:
        if self._inference_debug_printed:
            return
        print(f"[UPS_DEBUG] inference wav_len_samples={wav_len_samples}", flush=True)
        print(f"[UPS_DEBUG] inference mel_shape={mel_shape}", flush=True)
        print(f"[UPS_DEBUG] inference hidden_shape={hidden_shape}", flush=True)
        print(f"[UPS_DEBUG] inference valid_frames={valid_frames}/{total_frames}", flush=True)
        print(f"[UPS_DEBUG] inference final_emb_norm={emb_norm:.6f}", flush=True)
        self._inference_debug_printed = True

    def single_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "wav_b64" not in payload:
            raise ValueError("payload must contain 'wav_b64'")
        wav = self._decode_wav_b64(payload["wav_b64"])
        x = self._wav_to_logmel_bt80(wav)
        reps = self._extract_backbone_reps(x)   # [1, T, d_model]
        reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
        emb, _, _ = self._masked_mean_pool(reps, x)

        emb = self._project_to_output_dim(emb)
        emb = emb / (emb.norm(p=2) + 1e-12)
        return {"embedding": emb.detach().cpu().tolist()}

    def batch_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = payload.get("items", [])
        return {"results": [self.single_evaluation(it) for it in items]}

    # Alternate API names used by some Dynabench runners
    def batch_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_evaluation(payload)

    def single_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.single_evaluation(payload)

    def batch_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_evaluation(payload)


Model = ModelController
