"""
src/models/neuromm_model.py

NeuroMM-2026 Y-Architecture — full model.

One shared EEGMamba backbone → three specialized track heads.

Usage:
    from src.models import NeuroMMModel, build_model_from_config

    model = build_model_from_config(cfg)

    # Multi-task pretraining
    out = model(eeg, mode="pretrain", masked_eeg=masked)

    # Track inference
    out = model(eeg, mode="track1")
    out = model(eeg, video_features=vf, mode="track2")
    out = model(eeg, video_features=vf, mode="track3")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EEGBackbone, BackboneConfig
from .heads import Track1Head, Track2Head, Track3Head


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class NeuroMMModel(nn.Module):
    """
    Y-Architecture: shared EEGMamba backbone + three track heads +
    multi-task pretraining heads.

    Design principles:
    - Backbone is shared and pretrained with multi-task objectives.
    - Each track head is fine-tuned independently with its own loss.
    - Track 1 is EEG-only → video heads are only active for T2/T3.
    - Track 3 operates on positives only (enforced at data-loading level).
    - freeze_backbone() / unfreeze_backbone() for staged fine-tuning.
    """

    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        # Track 1
        t1_hidden_dims: Optional[list[int]] = None,
        t1_dropout: float = 0.40,
        # Track 2
        t2_video_configs: Optional[dict] = None,
        t2_target_dim: int = 256,
        t2_n_ca_heads: int = 8,
        t2_n_ca_layers: int = 2,
        t2_mc_passes: int = 5,
        t2_mc_dropout: float = 0.30,
        t2_entropy_threshold: float = 0.50,
        t2_hidden_dims: Optional[list[int]] = None,
        t2_dropout: float = 0.40,
        # Track 3
        t3_video_config: Optional[dict] = None,
        t3_hidden_dim: int = 256,
        t3_n_classes: int = 5,
        t3_n_heads: int = 4,
        t3_n_layers: int = 2,
        t3_dropout: float = 0.30,
        # Pretrain
        pretrain_mask_ratio: float = 0.20,
    ) -> None:
        super().__init__()

        self.backbone_cfg = backbone_cfg
        embed_dim = backbone_cfg.embed_dim
        ch_dim = backbone_cfg.gat_hidden_dim

        # ── Shared Backbone 
        self.backbone = EEGBackbone(backbone_cfg)

        # ── Pretrain Heads (active only during pretraining phase) ────────
        self.pt_binary   = nn.Linear(embed_dim, 1)
        self.pt_labeltype = nn.Linear(embed_dim, 6)       # classes 0..5
        self.pt_recon    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2,
                      backbone_cfg.n_channels * backbone_cfg.n_timesteps),
        )

        # ── Track Heads ────
        self.track1 = Track1Head(embed_dim, t1_hidden_dims or [256, 128], t1_dropout)

        self.track2 = Track2Head(
            embed_dim=embed_dim,
            video_configs=t2_video_configs,
            target_dim=t2_target_dim,
            n_ca_heads=t2_n_ca_heads,
            n_ca_layers=t2_n_ca_layers,
            mc_passes=t2_mc_passes,
            mc_dropout=t2_mc_dropout,
            entropy_threshold=t2_entropy_threshold,
            hidden_dims=t2_hidden_dims or [256, 128],
            dropout=t2_dropout,
        )

        self.track3 = Track3Head(
            embed_dim=embed_dim,
            channel_feat_dim=ch_dim,
            video_config=t3_video_config,
            hidden_dim=t3_hidden_dim,
            n_classes=t3_n_classes,
            n_heads=t3_n_heads,
            n_layers=t3_n_layers,
            dropout=t3_dropout,
        )

    # ── Forward ───────────

    def forward(
        self,
        eeg: torch.Tensor,
        video_features: Optional[dict[str, torch.Tensor]] = None,
        mode: str = "track1",
        masked_eeg: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Unified forward — returns a dict of tensors.

        Modes:
            "pretrain"  — multi-task, uses masked_eeg if provided
            "track1"    — binary logit (no video)
            "track2"    — binary logit + gate (video required)
            "track3"    — 5-class logits + channel_feats (video optional)
        """
        dispatch = {
            "pretrain": self._pretrain,
            "track1":   self._track1,
            "track2":   self._track2,
            "track3":   self._track3,
        }
        fn = dispatch.get(mode)
        if fn is None:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(dispatch)}")
        return fn(eeg, video_features=video_features, masked_eeg=masked_eeg)

    def _pretrain(self, eeg, *, video_features=None, masked_eeg=None):
        src = masked_eeg if masked_eeg is not None else eeg
        embed = self.backbone(src)

        out: dict[str, torch.Tensor] = {
            "binary_logit":    self.pt_binary(embed).squeeze(-1),
            "labeltype_logit": self.pt_labeltype(embed),
        }
        if masked_eeg is not None:
            B, C, T = eeg.shape
            out["recon"]        = self.pt_recon(embed).view(B, C, T)
            out["recon_target"] = eeg
        return out

    def _track1(self, eeg, *, video_features=None, masked_eeg=None):
        embed = self.backbone(eeg)
        return {"logit": self.track1(embed)}

    def _track2(self, eeg, *, video_features=None, masked_eeg=None):
        assert video_features, "video_features required for track2"
        embed = self.backbone(eeg)
        logit, gate = self.track2(embed, video_features)
        return {"logit": logit, "gate": gate}

    def _track3(self, eeg, *, video_features=None, masked_eeg=None):
        embed, ch_feats = self.backbone(eeg, return_channel_feats=True)
        logit = self.track3(embed, ch_feats, video_features)
        return {"logit": logit, "channel_feats": ch_feats}

    # ── Backbone control ──

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    def load_foundation_weights(self, path: str | Path, strict: bool = False) -> None:
        self.backbone.load_foundation_weights(path, strict=strict)

    def apply_lora(self, r: int = 16, alpha: int = 32) -> None:
        self.backbone.apply_lora(r=r, alpha=alpha)

    # ── Parameter groups ──

    def backbone_params(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

    def head_params(self, track: int) -> list[nn.Parameter]:
        heads = {1: self.track1, 2: self.track2, 3: self.track3}
        return list(heads[track].parameters())

    def pretrain_params(self) -> list[nn.Parameter]:
        return (
            list(self.backbone.parameters())
            + list(self.pt_binary.parameters())
            + list(self.pt_labeltype.parameters())
            + list(self.pt_recon.parameters())
        )

    # ── Diagnostics ───────

    def count_parameters(self) -> dict[str, int]:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "backbone": n(self.backbone),
            "track1":   n(self.track1),
            "track2":   n(self.track2),
            "track3":   n(self.track3),
            "pretrain_heads": n(self.pt_binary) + n(self.pt_labeltype) + n(self.pt_recon),
            "total":    n(self),
        }


# ---------------------------------------------------------------------------
# Factory from YAML config
# ---------------------------------------------------------------------------

def build_model_from_config(cfg: dict) -> NeuroMMModel:
    """
    Build the full NeuroMMModel from a loaded YAML config dict.
    Handles both base config and track-specific overrides.
    """
    bb_raw = cfg.get("backbone", {})
    backbone_cfg = BackboneConfig.from_dict(bb_raw)

    # Video configs for T2
    t2_vid = {}
    video_section = cfg.get("video", {})
    active = video_section.get("active_backbones", ["videomae-large", "dinov2-large"])
    all_bbs = video_section.get("all_backbones", {})
    target_dim = video_section.get("target_dim", 256)
    for name in active:
        if name in all_bbs:
            t2_vid[name] = all_bbs[name]

    # Video config for T3 (first active backbone)
    t3_vid = None
    t3_bb_name = cfg.get("video", {}).get("active_backbones", [None])[0]
    if t3_bb_name and t3_bb_name in all_bbs:
        t3_vid = all_bbs[t3_bb_name]

    gating = cfg.get("gating", {})
    ca     = cfg.get("cross_attention", {})
    head   = cfg.get("head", {})
    t3_head = cfg.get("head", {})

    return NeuroMMModel(
        backbone_cfg=backbone_cfg,
        # T1
        t1_hidden_dims=head.get("hidden_dims", [256, 128]),
        t1_dropout=head.get("dropout", 0.40),
        # T2
        t2_video_configs=t2_vid,
        t2_target_dim=target_dim,
        t2_n_ca_heads=ca.get("n_heads", 8),
        t2_n_ca_layers=ca.get("n_layers", 2),
        t2_mc_passes=gating.get("mc_dropout_passes", 5),
        t2_mc_dropout=gating.get("dropout_rate", 0.30),
        t2_entropy_threshold=gating.get("entropy_threshold", 0.50),
        t2_hidden_dims=head.get("hidden_dims", [256, 128]),
        t2_dropout=head.get("dropout", 0.40),
        # T3
        t3_video_config=t3_vid,
        t3_hidden_dim=t3_head.get("hidden_dim", 256),
        t3_n_classes=cfg.get("n_classes", 5),
        t3_n_heads=t3_head.get("n_heads", 4),
        t3_n_layers=t3_head.get("n_layers", 2),
        t3_dropout=t3_head.get("dropout", 0.30),
    )
