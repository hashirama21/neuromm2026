"""
src/data/dataset.py

NeuroMM-2026 Dataset + patient-disjoint CV factory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .augmentations import AugmentationPipeline
from .preprocessing import apply_scalers, fit_per_channel_scalers
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeuroMMDataset(Dataset):
    """
    PyTorch Dataset for NeuroMM-2026 (train / val splits).

    Args:
        df:               Pandas DataFrame with columns:
                          sample_id, split, label, label_type, subject_id
        eeg_dir:          Directory with {sample_id}.npy EEG files
        video_dir:        Base dir for video feature files
        video_backbones:  List of backbone names to load
        scalers:          Per-channel RobustScalers (fitted on fold-train)
        track:            1 | 2 | 3
        augment:          Apply augmentations (training only)
        augment_cfg:      Augmentation config dict
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eeg_dir: Path,
        video_dir: Optional[Path] = None,
        video_backbones: Optional[list[str]] = None,
        scalers: Optional[list[RobustScaler]] = None,
        track: int = 1,
        augment: bool = False,
        augment_cfg: Optional[dict] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.eeg_dir = Path(eeg_dir)
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_backbones = video_backbones or []
        self.scalers = scalers
        self.track = track
        self.augment_fn = AugmentationPipeline(augment_cfg or {}) if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sid = row["sample_id"]

        # ── EEG ──────────
        eeg = np.load(self.eeg_dir / f"{sid}.npy").astype(np.float32)   # (C, T)
        if eeg.shape[1] < 2000:
            # Truncated boundary window — zero-pad to 2000 on the right
            eeg = np.pad(eeg, ((0, 0), (0, 2000 - eeg.shape[1])))
        if self.scalers:
            eeg = apply_scalers(eeg, self.scalers)
        if self.augment_fn:
            eeg = self.augment_fn(eeg)

        item: dict = {
            "eeg":        torch.from_numpy(eeg),
            "label":      torch.tensor(float(row["label"]),      dtype=torch.float32),
            "label_type": torch.tensor(int(row["label_type"]),   dtype=torch.long),
            "sample_id":  sid,
        }

        # ── Video features ──
        if self.track in (2, 3) and self.video_dir:
            vf: dict[str, torch.Tensor] = {}
            for bb in self.video_backbones:
                p = self.video_dir / bb / f"{sid}.npy"
                if p.exists():
                    arr = np.load(p).astype(np.float32)
                    vf[bb] = torch.from_numpy(arr)
            item["video_features"] = vf

        return item


# ---------------------------------------------------------------------------
# Candidate Dataset (no labels, per-sample inference)
# ---------------------------------------------------------------------------

class CandidateDataset(Dataset):
    """
    Test-phase candidate set.

    CRITICAL:
    - No labels.
    - Scalers fitted on FULL training set (not fold-specific).
    - No global stats — each sample is independent.
    """

    def __init__(
        self,
        candidate_ids: list[str],
        eeg_dir: Path,
        scalers: list[RobustScaler],
        track: int = 1,
        video_dir: Optional[Path] = None,
        video_backbones: Optional[list[str]] = None,
    ) -> None:
        self.ids = candidate_ids
        self.eeg_dir = Path(eeg_dir)
        self.scalers = scalers
        self.track = track
        self.video_dir = Path(video_dir) if video_dir else None
        self.video_backbones = video_backbones or []

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.ids[idx]
        eeg = np.load(self.eeg_dir / f"{sid}.npy").astype(np.float32)
        if eeg.shape[1] < 2000:
            eeg = np.pad(eeg, ((0, 0), (0, 2000 - eeg.shape[1])))
        eeg = apply_scalers(eeg, self.scalers)

        item: dict = {"eeg": torch.from_numpy(eeg), "sample_id": sid}

        if self.track in (2, 3) and self.video_dir:
            vf: dict[str, torch.Tensor] = {}
            for bb in self.video_backbones:
                p = self.video_dir / bb / f"{sid}.npy"
                if p.exists():
                    vf[bb] = torch.from_numpy(np.load(p).astype(np.float32))
            item["video_features"] = vf

        return item


# ---------------------------------------------------------------------------
# CV splitting
# ---------------------------------------------------------------------------

def make_patient_disjoint_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "subject_id",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Patient-disjoint GroupKFold splits.
    Raises AssertionError if any patient leaks between train and val.
    """
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(df, groups=df[group_col].values))

    for i, (tr, va) in enumerate(folds):
        leak = set(df.iloc[tr][group_col]) & set(df.iloc[va][group_col])
        assert not leak, f"Fold {i}: patient leakage detected: {leak}"

    return folds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_weighted_sampler(
    df: pd.DataFrame,
    label_col: str = "label",
    max_ratio: float = 4.0,
) -> WeightedRandomSampler:
    """Oversample positives, cap neg:pos ratio at max_ratio."""
    labels = df[label_col].values.astype(float)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    neg_w = min(n_pos / (n_neg + 1e-8), 1.0 / max_ratio)
    weights = np.where(labels == 1, 1.0, neg_w)
    weights /= weights.sum() / len(weights)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights.astype(np.float32)),
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloader(
    df: pd.DataFrame,
    eeg_dir: Path,
    scalers: list[RobustScaler],
    track: int = 1,
    video_dir: Optional[Path] = None,
    video_backbones: Optional[list[str]] = None,
    batch_size: int = 64,
    augment: bool = False,
    augment_cfg: Optional[dict] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    use_weighted_sampler: bool = False,
    max_pos_ratio: float = 4.0,
) -> DataLoader:
    # Track 3: positives only
    if track == 3:
        df = df[df["label"] == 1].reset_index(drop=True)

    ds = NeuroMMDataset(
        df=df, eeg_dir=eeg_dir,
        video_dir=video_dir, video_backbones=video_backbones,
        scalers=scalers, track=track,
        augment=augment, augment_cfg=augment_cfg,
    )

    sampler = None
    if use_weighted_sampler and augment and track != 3:
        sampler = make_weighted_sampler(df, max_ratio=max_pos_ratio)
        shuffle = False

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=(augment and track != 3),
    )
