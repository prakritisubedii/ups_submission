
import base64, io, sys, tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio

# Works both in a real .py file (Dynabench) and in Colab notebooks
ZIP_ROOT = Path(__file__).resolve().parents[2]
_DYNA_LOG = {"n": 0, "max": 25}

sys.path.insert(0, str(ZIP_ROOT / "ups_challenge_baselines"))

from scripts.train_mel_msm_bimamba2 import BiMambaMSM  # noqa: E402


class ModelController:
    # Dynalab-style wrapper. Returns backbone reps [T, 128]. CUDA required.
    SR = 16000
    N_FFT = 400
    HOP = 160
    N_MELS = 80
    CHUNK_SEC = 10.0
    EPS = 1e-6

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.initialized = True
        self.device = torch.device(device)

        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required (Bi-Mamba2/Triton kernels).")

        ckpt_path = ZIP_ROOT / "submit_app" / "resources" / "ckpt_step_11000_infer.pt"
        if not ckpt_path.exists():
            ckpt_path = ZIP_ROOT / "app" / "resources" / "ckpt_step_11000_infer.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        # PyTorch 2.6 defaults torch.load(..., weights_only=True), which can
        # reject metadata objects in older checkpoints (e.g., TorchVersion).
        try:
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception:
            # Fallback for older torch versions and environments without
            # safe_globals/weights_only support.
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
        d_model = ckpt["cfg"]["d_model"]

        self.model = BiMambaMSM(d_model=d_model).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state"], strict=True)

        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SR,
            n_fft=self.N_FFT,
            hop_length=self.HOP,
            n_mels=self.N_MELS,
            power=2.0,
        )

    def single_inference(self, input_data: Any, sample_rate: int = 16000) -> Any:
        """Run inference on a single audio sample.

        Args:
            input_data (torch.Tensor): A 1D tensor containing the audio samples (shape [T])
            sample_rate (int): Sample rate of the provided audio.

        Returns:
            torch.Tensor: Embeddings (shape [T, D]). Returned on the model device.
        """
        # Legacy Dynabench payload path
        if isinstance(input_data, dict):
            return self.single_evaluation(input_data)

        if not torch.is_tensor(input_data):
            raise TypeError("input_data must be a torch.Tensor (audio) or a dict payload")

        if input_data.dim() == 2 and input_data.size(0) == 1:
            input_data = input_data.squeeze(0)
        if input_data.dim() != 1:
            raise ValueError(f"input_data must be 1D (shape [T]); got shape={tuple(input_data.shape)}")

        with torch.no_grad():
            # torchaudio preprocessing is most reliable on CPU.
            wav = input_data.detach().to("cpu", dtype=torch.float32).unsqueeze(0)  # [1, T]
            wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
            if sample_rate != self.SR:
                wav = torchaudio.functional.resample(wav, sample_rate, self.SR)

            x = self._wav_to_logmel_bt80(wav)  # [1, T, 80]
            reps = self._extract_backbone_reps(x).squeeze(0).contiguous()  # [T, 128]
            reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
            return reps

    def _decode_wav_b64(self, wav_b64: str) -> torch.Tensor:
        wav_bytes = base64.b64decode(wav_b64)
        try:
            wav, in_sr = torchaudio.load(io.BytesIO(wav_bytes))  # [C, N]
        except RuntimeError:
            # Some torchaudio backends cannot decode file-like objects.
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                f.write(wav_bytes)
                f.flush()
                wav, in_sr = torchaudio.load(f.name)
        orig_sr = int(in_sr)
        orig_channels = int(wav.shape[0])
        wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if in_sr != self.SR:
            wav = torchaudio.functional.resample(wav, in_sr, self.SR)
        final_sr = int(self.SR)
        num_samples = int(wav.shape[-1])
        dur_sec = float(num_samples / final_sr) if final_sr > 0 else 0.0
        self._last_audio_meta = {
            "orig_sr": orig_sr,
            "orig_channels": orig_channels,
            "final_sr": final_sr,
            "num_samples": num_samples,
            "dur_sec": dur_sec,
            "min": float(wav.min().item()),
            "max": float(wav.max().item()),
            "mean": float(wav.mean().item()),
            "std": float(wav.std(unbiased=False).item()),
        }
        return wav  # [1, N]

    def _pad_or_crop_10s(self, wav: torch.Tensor) -> torch.Tensor:
        target_len = int(self.SR * self.CHUNK_SEC)
        n = wav.size(-1)
        self._last_padcrop = {"mode": "none", "pad_sec": 0.0, "crop_sec": 0.0}
        if n < target_len:
            pad_samples = int(target_len - n)
            self._last_padcrop = {
                "mode": "pad",
                "pad_sec": float(pad_samples / self.SR),
                "crop_sec": 0.0,
            }
            wav = torch.nn.functional.pad(wav, (0, pad_samples))
        elif n > target_len:
            cropped_samples = int(n - target_len)
            self._last_padcrop = {
                "mode": "crop",
                "pad_sec": 0.0,
                "crop_sec": float(cropped_samples / self.SR),
            }
            wav = wav[..., :target_len]
        return wav

    def _wav_to_logmel_bt80(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self._pad_or_crop_10s(wav)
        wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
        wav = torch.clamp(wav, min=-1.0, max=1.0)
        mel = self.mel_fn(wav)  # [1, 80, T]
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = torch.log(torch.clamp(mel, min=self.EPS))
        mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0).contiguous()  # [1, T, 80]
        return mel.to(self.device, dtype=torch.float32)

    def _extract_backbone_reps(self, x_bt80: torch.Tensor) -> torch.Tensor:
        """Project log-mel features and return backbone representations [1, T, 128]."""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            x_bt80 = x_bt80.to(device=device, dtype=torch.float32).contiguous()
            h = self.model.proj_in(x_bt80)  # [1, T, D]
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            mu = h.mean(dim=1, keepdim=True)
            sigma = h.std(dim=1, keepdim=True)
            h = ((h - mu) / (sigma + 1e-5)).contiguous()
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

            b = self.model.backbone(h)
            if isinstance(b, (tuple, list)):
                b = b[0]
            b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
            b_mu = b.mean(dim=1, keepdim=True)
            b_sigma = b.std(dim=1, keepdim=True)
            b = (b - b_mu) / (b_sigma + 1e-5)
            b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
            return b

    def single_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "wav_b64" not in payload:
            raise ValueError("payload must contain 'wav_b64'")
        wav = self._decode_wav_b64(payload["wav_b64"])
        # --- DYNA LOG (print ASAP after decode) ---
        if _DYNA_LOG["n"] < _DYNA_LOG["max"]:
            i = _DYNA_LOG["n"]
            _DYNA_LOG["n"] += 1

            meta = getattr(self, "_last_audio_meta", None) or {}
            pc = getattr(self, "_last_padcrop", None) or {"mode": "none", "pad_sec": 0.0, "crop_sec": 0.0}

            msg = (
                f"[DYNA_LOG {i}] "
                f"orig_sr={meta.get('orig_sr')} ch={meta.get('orig_channels')} "
                f"final_sr={meta.get('final_sr')} dur={meta.get('dur_sec', float('nan')):.3f}s "
                f"min={meta.get('min', float('nan')):.4f} max={meta.get('max', float('nan')):.4f} "
                f"mean={meta.get('mean', float('nan')):.4f} std={meta.get('std', float('nan')):.4f}"
            )
            print(msg, file=sys.stderr, flush=True)
        # --- end DYNA LOG ---
        x = self._wav_to_logmel_bt80(wav)
        if _DYNA_LOG["n"] < _DYNA_LOG["max"]:
            i = _DYNA_LOG["n"]
            meta = getattr(self, "_last_audio_meta", {}) or {}
            padcrop = getattr(self, "_last_padcrop", {}) or {}
            mode = padcrop.get("mode", "none")
            pad_sec = float(padcrop.get("pad_sec", 0.0) or 0.0)
            crop_sec = float(padcrop.get("crop_sec", 0.0) or 0.0)
            print(
                (
                    f"[DYNA_LOG {i}] "
                    f"orig_sr={meta.get('orig_sr', 'na')} "
                    f"ch={meta.get('orig_channels', 'na')} "
                    f"final_sr={meta.get('final_sr', 'na')} "
                    f"dur={float(meta.get('dur_sec', 0.0) or 0.0):.3f}s "
                    f"min={float(meta.get('min', 0.0) or 0.0):.6f} "
                    f"max={float(meta.get('max', 0.0) or 0.0):.6f} "
                    f"mean={float(meta.get('mean', 0.0) or 0.0):.6f} "
                    f"std={float(meta.get('std', 0.0) or 0.0):.6f} "
                    f"padcrop={mode}:{pad_sec:.3f}s/{crop_sec:.3f}s"
                ),
                file=sys.stderr,
                flush=True,
            )
            _DYNA_LOG["n"] += 1
        reps = self._extract_backbone_reps(x)
        return {"embedding": reps.squeeze(0).detach().cpu().tolist()}

    def batch_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = payload.get("items", [])
        return {"results": [self.single_evaluation(it) for it in items]}

    # Backwards/alternate API name used by some Dynabench runners
    def batch_inference(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_evaluation(payload)

    def single_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.single_evaluation(payload)

    def batch_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.batch_evaluation(payload)
