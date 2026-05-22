"""src/utils/logging.py — Logger + optional WandB wrapper."""
from __future__ import annotations
import logging, sys
from typing import Any, Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger


class WandBLogger:
    """Gracefully degrades if wandb not installed."""

    def __init__(self, cfg: dict, enabled: bool = False) -> None:
        self.enabled = enabled
        if enabled:
            try:
                import wandb as _wandb
                _wandb.init(
                    project=cfg.get("project", {}).get("name", "neuromm2026"),
                    config=cfg,
                    entity=cfg.get("logging", {}).get("wandb_entity"),
                )
                self._w = _wandb
            except ImportError:
                self.enabled = False

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            self._w.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._w.finish()
