import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/experiment_config.yaml") -> dict[str, Any]:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def project_root() -> Path:
    """Return absolute path to project root."""
    return Path(__file__).resolve().parent.parent.parent
