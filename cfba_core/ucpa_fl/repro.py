from __future__ import annotations
import os, random, platform, subprocess, sys
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str = 'auto') -> torch.device:
    """Use CUDA when the caller asks for it and this PyTorch build can see a GPU.

    ``auto`` and ``gpu`` select CUDA when ``torch.cuda.is_available()`` is true
    and otherwise stay on CPU. An explicit ``cuda`` request fails if no CUDA
    device is visible, instead of failing later inside a kernel launch.
    """
    name = (requested or 'auto').strip().lower()
    if name in ('auto', 'gpu'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'Requested device {requested!r}, but torch.cuda.is_available() is false. '
            'Install a CUDA build of PyTorch on a machine with an NVIDIA GPU, or use auto '
            'to run on CPU when no CUDA device is visible.'
        )
    return torch.device(name)


def move_state_dict(state: dict, device) -> dict:
    """Copy a state dict onto ``device`` so load_state_dict does not leave a module on CPU."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}


def environment_snapshot() -> dict:
    try:
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True, timeout=30).splitlines()
    except Exception:
        freeze = []
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'numpy': np.__version__,
        'pip_freeze': freeze,
    }
