"""
src/evaluation/metrics.py
Metrics, calibration and ensemble utilities.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_auprc(targets, scores) -> float:
    return float(average_precision_score(np.asarray(targets), np.asarray(scores)))


def compute_auroc(targets, scores) -> float:
    try:
        return float(roc_auc_score(np.asarray(targets), np.asarray(scores)))
    except ValueError:
        return float("nan")


def compute_weighted_f1(targets, preds, n_classes: int = 5) -> float:
    return float(f1_score(
        np.asarray(targets), np.asarray(preds),
        labels=list(range(1, n_classes + 1)),
        average="weighted", zero_division=0,
    ))


def find_best_threshold(targets, scores) -> tuple[float, float]:
    """Find F1-optimal threshold on OOF scores."""
    prec, rec, thresh = precision_recall_curve(targets, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1[:-1])
    return float(thresh[idx]), float(f1[idx])


# ---------------------------------------------------------------------------
# Metric accumulator
# ---------------------------------------------------------------------------

class MetricTracker:
    def __init__(self, track: int) -> None:
        self.track = track
        self._t: list = []
        self._s: list = []

    def reset(self) -> None:
        self._t.clear(); self._s.clear()

    def update(self, targets, logits) -> None:
        t = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
        l = logits.detach().cpu().numpy()  if isinstance(logits,  torch.Tensor) else np.asarray(logits)
        self._t.append(t)
        self._s.append(torch.sigmoid(torch.from_numpy(l)).numpy() if self.track in (1, 2) else l)

    def compute(self) -> dict[str, float]:
        t = np.concatenate(self._t)
        s = np.concatenate(self._s)
        if self.track in (1, 2):
            return {"auprc": compute_auprc(t, s), "auroc": compute_auroc(t, s)}
        preds = np.argmax(s, axis=-1) + 1
        return {"weighted_f1": compute_weighted_f1(t, preds)}


# ---------------------------------------------------------------------------
# OOF Calibrator
# ---------------------------------------------------------------------------

class OOFCalibrator:
    """
    Isotonic Regression calibration fitted on Out-of-Fold predictions.
    NEVER fit on the official val set or candidate set.
    """

    def __init__(self) -> None:
        self._oof_t: list[np.ndarray] = []
        self._oof_s: list[np.ndarray] = []
        self.iso: Optional[IsotonicRegression] = None

    def add_fold(self, targets, scores) -> None:
        self._oof_t.append(np.asarray(targets))
        self._oof_s.append(np.asarray(scores))

    def fit(self) -> None:
        t = np.concatenate(self._oof_t)
        s = np.concatenate(self._oof_s)
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(s, t)

    def transform(self, scores) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("Call fit() first")
        return self.iso.transform(np.asarray(scores))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "OOFCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    Logistic regression meta-learner over OOF stacked predictions.
    For binary (T1/T2): fits meta-learner, returns calibrated probability.
    For multiclass (T3): simple averaging of logits, then argmax.
    """

    def __init__(self, task: str = "binary") -> None:
        assert task in ("binary", "multiclass")
        self.task = task
        self._oof: list[np.ndarray] = []
        self._targets: Optional[np.ndarray] = None
        self.meta: Optional[LogisticRegression] = None

    def add_oof(self, scores: np.ndarray) -> None:
        self._oof.append(np.asarray(scores))

    def set_targets(self, t) -> None:
        self._targets = np.asarray(t)

    def fit(self) -> None:
        if self.task == "binary":
            X = np.column_stack(self._oof)
            self.meta = LogisticRegression(C=1.0, max_iter=1000)
            self.meta.fit(X, self._targets)

    def predict(self, test_scores: list[np.ndarray]) -> np.ndarray:
        if self.task == "binary":
            X = np.column_stack(test_scores)
            if self.meta:
                return self.meta.predict_proba(X)[:, 1]
            return np.mean(test_scores, axis=0)
        # Multiclass
        return np.argmax(np.mean(test_scores, axis=0), axis=-1) + 1

    def diversity(self) -> dict[str, float]:
        if len(self._oof) < 2:
            return {}
        from scipy.stats import pearsonr
        preds = np.column_stack(self._oof)
        corrs = [
            pearsonr(preds[:, i], preds[:, j])[0]
            for i in range(preds.shape[1])
            for j in range(i + 1, preds.shape[1])
        ]
        return {"mean_corr": float(np.mean(corrs)), "max_corr": float(np.max(corrs))}
