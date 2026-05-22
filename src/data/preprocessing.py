"""
src/data/preprocessing.py

Per-channel RobustScaler fitting and application.

CRITICAL: scalers must be fitted ONLY on the fold-train split.
          Fitting on val/test data constitutes label leakage.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def fit_per_channel_scalers(
    df_train: pd.DataFrame,
    eeg_dir: Path,
    n_channels: int = 29,
) -> list[RobustScaler]:
    """
    Fit one RobustScaler per EEG channel on the training fold.

    RobustScaler uses median and IQR → robust to high-amplitude artefacts
    that would skew StandardScaler's mean/std.

    Args:
        df_train:   DataFrame of training samples (fold-train only).
        eeg_dir:    Directory with {sample_id}.npy files.
        n_channels: Number of EEG channels (29).

    Returns:
        List of 29 fitted RobustScaler instances.
    """
    channel_data: list[list[float]] = [[] for _ in range(n_channels)]

    for _, row in df_train.iterrows():
        path = eeg_dir / f"{row['sample_id']}.npy"
        if not path.exists():
            continue
        eeg = np.load(path).astype(np.float32)   # (C, T)
        for c in range(min(n_channels, eeg.shape[0])):
            channel_data[c].extend(eeg[c].tolist())

    scalers: list[RobustScaler] = []
    for c in range(n_channels):
        sc = RobustScaler()
        data = np.array(channel_data[c], dtype=np.float32).reshape(-1, 1)
        sc.fit(data)
        scalers.append(sc)

    return scalers


def apply_scalers(eeg: np.ndarray, scalers: list[RobustScaler]) -> np.ndarray:
    """Apply per-channel scalers to a single EEG sample."""
    out = np.empty_like(eeg)
    for c, sc in enumerate(scalers):
        out[c] = sc.transform(eeg[c].reshape(-1, 1)).ravel()
    return out


def save_scalers(scalers: list[RobustScaler], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)


def load_scalers(path: Path) -> list[RobustScaler]:
    with open(path, "rb") as f:
        return pickle.load(f)
