"""
Config loading and merging utilities.
Supports YAML base config + CLI overrides via OmegaConf dot notation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load YAML config and optionally merge CLI overrides.

    Args:
        config_path: Path to base YAML config file.
        overrides:   List of dot-notation overrides, e.g. ["training.lr=1e-3"].

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(str(config_path))

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    OmegaConf.set_readonly(cfg, False)
    return cfg


def cfg_to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_config(cfg: DictConfig) -> None:
    """Pretty-print the resolved config."""
    from rich import print as rprint
    from rich.panel import Panel
    rprint(Panel(OmegaConf.to_yaml(cfg, resolve=True), title="[bold green]Config", expand=False))
