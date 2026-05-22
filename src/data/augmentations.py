"""
src/data/augmentations.py

EEG data augmentation strategies for training.
All augmentations operate on numpy arrays (C, T) and are applied
AFTER RobustScaler normalization.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class SpecAugment:
    """
    SpecAugment-style time masking for EEG signals.
    Randomly zeros out contiguous time windows to force the model
    to be robust to missing segments (simulates electrode dropout).
    """

    def __init__(
        self,
        max_time_mask: int = 200,
        num_time_masks: int = 2,
    ) -> None:
        self.max_time_mask = max_time_mask
        self.num_time_masks = num_time_masks

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        T = eeg.shape[1]
        eeg = eeg.copy()
        for _ in range(self.num_time_masks):
            if T <= self.max_time_mask:
                continue
            t0 = np.random.randint(0, T - self.max_time_mask)
            mask_len = np.random.randint(1, self.max_time_mask + 1)
            eeg[:, t0: t0 + mask_len] = 0.0
        return eeg


class GaussianNoise:
    """Add zero-mean Gaussian noise to simulate sensor noise."""

    def __init__(self, sigma: float = 0.05) -> None:
        self.sigma = sigma

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*eeg.shape).astype(np.float32) * self.sigma
        return eeg + noise


class TemporalMixup:
    """
    Mixup in the temporal domain.
    Linearly interpolates two EEG samples with a Beta(alpha, alpha) weight.
    Applied at the dataset level (requires two samples).
    """

    def __init__(self, alpha: float = 0.20) -> None:
        self.alpha = alpha

    def mix(
        self,
        eeg1: np.ndarray,
        eeg2: np.ndarray,
        label1: float,
        label2: float,
    ) -> tuple[np.ndarray, float]:
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_eeg = lam * eeg1 + (1 - lam) * eeg2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_eeg.astype(np.float32), float(mixed_label)


class AugmentationPipeline:
    """Compose multiple augmentations, each applied with its own probability."""

    def __init__(self, cfg: dict) -> None:
        self.transforms: list = []

        if cfg.get("spec_augment", {}).get("enabled", True):
            sa_cfg = cfg["spec_augment"]
            self.transforms.append(
                SpecAugment(
                    max_time_mask=sa_cfg.get("max_time_mask", 200),
                    num_time_masks=sa_cfg.get("num_time_masks", 2),
                )
            )

        if cfg.get("gaussian_noise", {}).get("enabled", True):
            self.transforms.append(
                GaussianNoise(sigma=cfg["gaussian_noise"].get("sigma", 0.05))
            )

    def __call__(self, eeg: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            eeg = t(eeg)
        return eeg
