"""
src/training/losses.py

Loss functions for NeuroMM-2026:
  - FocalLoss         : handles extreme class imbalance
  - PolyLoss          : Taylor-expanded CE, better AUPRC gradient
  - FocalPolyLoss     : combined (primary for T1/T2)
  - APLoss            : Average Precision surrogate (auxiliary)
  - WeightedCELoss    : inverse-frequency weighted CE (T3)
  - MultiTaskLoss     : combines all pretrain objectives
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy negatives, focuses on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class PolyLoss(nn.Module):
    """
    PolyLoss: L_poly = L_CE + epsilon_1 * (1 - p_t)
    Adjusting epsilon_1 directly reshapes AUPRC gradient.
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return bce + self.epsilon * (1 - p_t).mean()


class FocalPolyLoss(nn.Module):
    """Combined Focal + PolyLoss (primary loss for Track 1/2)."""

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        poly_epsilon: float = 1.0,
        focal_weight: float = 1.0,
        poly_weight: float = 0.50,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(focal_gamma, focal_alpha)
        self.poly  = PolyLoss(poly_epsilon)
        self.fw = focal_weight
        self.pw = poly_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.fw * self.focal(logits, targets) + self.pw * self.poly(logits, targets)


class APLoss(nn.Module):
    """
    Differentiable Average Precision surrogate.
    Expensive for large batches — use as auxiliary loss only.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(logits)
        n_pos = targets.sum()
        if n_pos == 0:
            return logits.sum() * 0
        _, idx = torch.sort(scores, descending=True)
        sorted_t = targets[idx]
        cumsum = torch.cumsum(sorted_t, 0)
        rank = torch.arange(1, len(sorted_t) + 1, device=logits.device, dtype=torch.float32)
        ap = (cumsum / rank * sorted_t).sum() / n_pos
        return 1.0 - ap


class WeightedCELoss(nn.Module):
    """
    Cross-entropy with inverse-frequency class weights for Track 3.
    Weights are recomputed dynamically from training fold class counts.
    """

    def __init__(
        self,
        class_counts: list[int],
        label_smoothing: float = 0.10,
    ) -> None:
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        w = 1.0 / (counts + 1e-6)
        w = w / w.sum() * len(counts)
        self.register_buffer("weight", w)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, C), targets: (B,) long, values 0..C-1"""
        return F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )

    @classmethod
    def from_dataframe(
        cls,
        df,
        label_col: str = "label_type",
        n_classes: int = 5,
        label_smoothing: float = 0.10,
    ) -> "WeightedCELoss":
        """Build from a training DataFrame. Classes are 1..n_classes."""
        counts = []
        for c in range(1, n_classes + 1):
            counts.append(int((df[label_col] == c).sum()))
        return cls(counts, label_smoothing)


class MultiTaskLoss(nn.Module):
    """
    Multi-task pretraining loss.

    L = λ_bin * L_focal_poly + λ_recon * L_mse + λ_type * L_ce
    """

    def __init__(
        self,
        lambda_binary: float = 1.0,
        lambda_recon: float = 0.30,
        lambda_labeltype: float = 0.50,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
        poly_epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.lw_bin   = lambda_binary
        self.lw_recon = lambda_recon
        self.lw_type  = lambda_labeltype
        self.bin_loss = FocalPolyLoss(focal_gamma, focal_alpha, poly_epsilon)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        label_types: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_bin = self.bin_loss(outputs["binary_logit"], labels)
        l_type = F.cross_entropy(outputs["labeltype_logit"], label_types)

        l_recon = torch.tensor(0.0, device=labels.device)
        if "recon" in outputs:
            l_recon = F.mse_loss(outputs["recon"], outputs["recon_target"])

        total = self.lw_bin * l_bin + self.lw_recon * l_recon + self.lw_type * l_type

        return {
            "loss":          total,
            "loss_binary":   l_bin,
            "loss_recon":    l_recon,
            "loss_labeltype": l_type,
        }
