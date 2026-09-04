from __future__ import annotations

from contextlib import nullcontext
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RuntimeProfile:
    device: torch.device
    deterministic: bool
    amp_mode: str
    amp_enabled: bool
    amp_dtype: torch.dtype
    pin_memory: bool
    non_blocking: bool
    device_name: str
    compute_capability: tuple[int, int] | None
    total_memory_gb: float | None

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["device"] = str(self.device)
        data["amp_dtype"] = str(self.amp_dtype).replace("torch.", "")
        return data


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = str(device).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return resolved


def configure_runtime(
    device: str | torch.device,
    deterministic: bool = False,
    amp_mode: str = "auto",
) -> RuntimeProfile:
    resolved = resolve_device(device)
    amp_mode = str(amp_mode).lower()
    if amp_mode not in {"auto", "on", "off"}:
        raise ValueError("amp_mode must be one of: auto, on, off")
    if amp_mode == "on" and resolved.type != "cuda":
        raise RuntimeError("AMP mode 'on' requires a CUDA device")

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=not deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    amp_enabled = resolved.type == "cuda" and amp_mode != "off"
    device_name = "CPU"
    capability = None
    total_memory_gb = None
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        device_name = props.name
        capability = (int(props.major), int(props.minor))
        total_memory_gb = float(props.total_memory) / (1024**3)

    return RuntimeProfile(
        device=resolved,
        deterministic=bool(deterministic),
        amp_mode=amp_mode,
        amp_enabled=amp_enabled,
        amp_dtype=torch.float16,
        pin_memory=resolved.type == "cuda",
        non_blocking=resolved.type == "cuda",
        device_name=device_name,
        compute_capability=capability,
        total_memory_gb=total_memory_gb,
    )



def system_report(device: str | torch.device = "auto") -> dict[str, object]:
    profile = configure_runtime(device, deterministic=False, amp_mode="auto")
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(profile.device),
        "device_name": profile.device_name,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "amp_enabled": bool(profile.amp_enabled),
        "compute_capability": profile.compute_capability,
        "total_memory_gb": profile.total_memory_gb,
        "deterministic_default": False,
    }
    return report

def autocast_context(profile: RuntimeProfile):
    if not profile.amp_enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=profile.amp_dtype, enabled=True)


def make_grad_scaler(profile: RuntimeProfile):
    return torch.amp.GradScaler("cuda", enabled=profile.amp_enabled)


def make_adam(params: Iterable[torch.nn.Parameter], lr: float, device: torch.device) -> torch.optim.Adam:
    params = list(params)
    if device.type == "cuda":
        try:
            return torch.optim.Adam(params, lr=lr, fused=True)
        except (RuntimeError, TypeError, ValueError):
            try:
                return torch.optim.Adam(params, lr=lr, foreach=True)
            except (RuntimeError, TypeError, ValueError):
                pass
    return torch.optim.Adam(params, lr=lr)
