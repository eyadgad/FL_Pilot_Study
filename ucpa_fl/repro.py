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
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(requested)


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
