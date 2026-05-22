"""
src/utils/config.py
YAML config loader with base-config merging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (override wins on conflicts)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load YAML config. If it contains a 'defaults' key referencing a base
    config, the base is loaded first and the current config is merged on top.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "defaults" in cfg:
        base_refs = cfg.pop("defaults")
        for ref in base_refs:
            base_path = path.parent / f"{ref}.yaml"
            with open(base_path) as f:
                base = yaml.safe_load(f)
            cfg = _deep_merge(base, cfg)

    return cfg


def flatten_config(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested config dict for WandB / logging."""
    flat: dict[str, Any] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, key))
        else:
            flat[key] = v
    return flat
