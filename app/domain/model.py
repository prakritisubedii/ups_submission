
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
try:
    from scripts.train_mel_msm_bimamba2 import BiMambaMSM  # noqa: E402
except Exception:
    BiMambaMSM = None


class ModelController:
    # Dynalab-style wrapper. Returns reps [T, 128].
    SR = 16000
    N_FFT = 400
    HOP = 160
    N_MELS = 80
    CHUNK_SEC = 10.0
    EPS = 1e-6
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

        # Preferred path on newer torch: keep weights_only=True and allowlist TorchVersion metadata.
        try:
            if torch_version_cls is not None and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([torch_version_cls]):
                    return torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
            return torch.load(str(ckpt_path), weights_only=True, **load_kwargs)
        except Exception:
            # Trusted local checkpoint fallback for environments where weights_only path fails.
            try:
                return torch.load(str(ckpt_path), weights_only=False, **load_kwargs)
            except TypeError:
                # Older torch versions without weights_only argument.
                return torch.load(str(ckpt_path), **load_kwargs)

    def __init__(self, device: str = "cuda" if CUDA_AVAILABLE else "cpu") -> None:
        self.initialized = True
        requested_device = torch.device(device)
        self.model = None
        self.backend = "cpu_fallback"
        self.d_model = 128

        ckpt_path = ZIP_ROOT / "app" / "resources" / "ckpt_step_11000_infer.pt"
        ckpt = {}
        if ckpt_path.exists():
            ckpt = self._load_checkpoint(ckpt_path)

        cfg = ckpt.get("cfg", {})
        d_model = cfg.get("d_model", 128)
        num_layers = cfg.get("num_layers", 1)
        self.d_model = d_model

        state = ckpt.get("model_state", {})

        # CPU-safe fallback path (used when CUDA/Triton is not available).
        self.fallback_proj = torch.nn.Linear(80, d_model)
        self.fallback_in_norm = torch.nn.LayerNorm(d_model)
        self.fallback_out_norm = torch.nn.LayerNorm(d_model)
        if "proj_in.weight" in state and "proj_in.bias" in state:
            with torch.no_grad():
                self.fallback_proj.weight.copy_(state["proj_in.weight"])
                self.fallback_proj.bias.copy_(state["proj_in.bias"])
        if "input_norm.weight" in state and "input_norm.bias" in state:
            with torch.no_grad():
                self.fallback_in_norm.weight.copy_(state["input_norm.weight"])
                self.fallback_in_norm.bias.copy_(state["input_norm.bias"])
        if "final_norm.weight" in state and "final_norm.bias" in state:
            with torch.no_grad():
                self.fallback_out_norm.weight.copy_(state["final_norm.weight"])
                self.fallback_out_norm.bias.copy_(state["final_norm.bias"])

        use_cuda_backend = (
            requested_device.type == "cuda" and torch.cuda.is_available() and BiMambaMSM is not None
        )
        if use_cuda_backend:
            self.device = requested_device
            try:
                self.model = BiMambaMSM(d_model=d_model, num_layers=num_layers).to(self.device)
                if "lid_head.weight" in state:
                    n_lids = state["lid_head.weight"].shape[0]
                    self.model.lid_head = torch.nn.Linear(d_model, n_lids).to(self.device)
                if state:
                    self.model.load_state_dict(state, strict=True)
                self.model.eval()
                self.backend = "bimamba_cuda"
            except Exception:
                # If CUDA kernels fail to initialize on runner, fall back to CPU-safe path.
                self.model = None
                self.backend = "cpu_fallback"
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if BiMambaMSM is not None and self.model is None:
            try:
                self.device = torch.device("cpu")
                with torch.no_grad():
                    self.model = BiMambaMSM(d_model=d_model, num_layers=num_layers).to("cpu")
                self.model = self.model.float()
                if "lid_head.weight" in state:
                    n_lids = state["lid_head.weight"].shape[0]
                    self.model.lid_head = torch.nn.Linear(d_model, n_lids)
                if state:
                    self.model.load_state_dict(state, strict=True)
                self.model.eval()
                self.backend = "bimamba_cpu"
            except Exception as e:
                self.model = None
                self.backend = "cpu_fallback"
                self.device = torch.device("cpu")

        self.fallback_proj = self.fallback_proj.to(self.device).eval()
        self.fallback_in_norm = self.fallback_in_norm.to(self.device).eval()
        self.fallback_out_norm = self.fallback_out_norm.to(self.device).eval()

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
        # Backward-compatible payload path
        if isinstance(input_data, dict):
            return self.single_evaluation(input_data)
        if isinstance(input_data, str):
            wav = self._decode_wav_b64(input_data)
            with torch.no_grad():
                x = self._wav_to_logmel_bt80(wav)  # [1, T, 80]
                reps = self._extract_backbone_reps(x)  # [1, T, D]
                reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
                emb = reps.squeeze(0).mean(dim=0)  # [D]
                emb = F.normalize(emb, dim=-1)
                return emb.detach().cpu().float()

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
            reps = self._extract_backbone_reps(x)  # [1, T, D]
            reps = torch.nan_to_num(reps, nan=0.0, posinf=0.0, neginf=0.0)
            emb = reps.squeeze(0).mean(dim=0)  # [D]
            emb = F.normalize(emb, dim=-1)
            return emb.detach().cpu().float()

    def _decode_wav_b64(self, wav_b64: str) -> torch.Tensor:
        wav_bytes = base64.b64decode(wav_b64)
        try:
            wav, in_sr = torchaudio.load(io.BytesIO(wav_bytes))  # [C, N]
        except Exception:
            # Some torchaudio backends cannot decode file-like objects.
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
        with torch.no_grad():
            if self.backend in ("bimamba_cuda", "bimamba_cpu") and self.model is not None:
                x_bt80 = x_bt80.to(
                    device=next(self.model.parameters()).device,
                    dtype=torch.float32,
                ).contiguous()
                _, hidden = self.model(x_bt80)  # forward returns (pred, hidden); use hidden
                hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
                return hidden

            # CPU-safe fallback: projection + normalization + light temporal smoothing.
            x_bt80 = x_bt80.to(self.device, dtype=torch.float32).contiguous()
            hidden = self.fallback_proj(x_bt80)
            hidden = self.fallback_in_norm(hidden)
            hidden = F.gelu(hidden)
            # Depthwise temporal smoothing to provide local context without CUDA kernels.
            h = hidden.transpose(1, 2).contiguous()
            h = F.pad(h, (1, 1), mode="replicate")
            kernel = torch.tensor([0.25, 0.5, 0.25], device=h.device, dtype=h.dtype).view(1, 1, 3)
            kernel = kernel.repeat(h.shape[1], 1, 1)
            h = F.conv1d(h, kernel, groups=h.shape[1])
            hidden = h.transpose(1, 2).contiguous()
            hidden = self.fallback_out_norm(hidden)
            hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
            return hidden

    def _project_to_output_dim(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.numel() >= self.OUTPUT_DIM:
            return emb[: self.OUTPUT_DIM].contiguous()
        return F.pad(emb, (0, self.OUTPUT_DIM - emb.numel()))

    def single_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "wav_b64" not in payload:
            raise ValueError("payload must contain 'wav_b64'")
        wav = self._decode_wav_b64(payload["wav_b64"])
        x = self._wav_to_logmel_bt80(wav)
        reps = self._extract_backbone_reps(x)   # [1, T, D]
        reps = reps.squeeze(0)  # [T, D]
        T = reps.shape[0]
        seg = max(1, T // 5)
        # take 3 non-overlapping segments: start, middle, end
        s1 = reps[:seg].mean(dim=0)
        s2 = reps[T//2 - seg//2 : T//2 + seg//2].mean(dim=0)
        s3 = reps[T-seg:].mean(dim=0)
        emb = (s1 + s2 + s3) / 3.0
        emb = self._project_to_output_dim(emb)
        emb = emb / (emb.norm(p=2) + 1e-12)     # L2 normalize
        return {"embedding": emb.detach().cpu().tolist()}

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


Model = ModelController
