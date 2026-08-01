"""Shared utilities for the fraud detection calibration project."""

import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_class_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight for imbalanced binary classification."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    if n_pos == 0:
        raise ValueError("No positive samples found")
    return n_neg / n_pos
