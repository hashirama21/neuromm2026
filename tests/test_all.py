"""
tests/test_all.py

Comprehensive unit tests for the full NeuroMM-2026 project.
Run: python -m pytest tests/ -v --tb=short
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.models.dynamic_gat import DynamicGAT, DynamicGATLayer
from src.models.eegmamba_encoder import EEGMambaEncoder, NativeMambaBlock, RMSNorm
from src.models.backbone import EEGBackbone, BackboneConfig
from src.models.heads import Track1Head, Track2Head, Track3Head
from src.models.neuromm_model import NeuroMMModel, build_model_from_config
from src.training.losses import (
    FocalLoss, PolyLoss, FocalPolyLoss, WeightedCELoss, MultiTaskLoss
)
from src.evaluation.metrics import (
    compute_auprc, compute_weighted_f1, OOFCalibrator, MetricTracker
)
from src.data.augmentations import SpecAugment, GaussianNoise, AugmentationPipeline
from src.data.preprocessing import fit_per_channel_scalers, apply_scalers

# ── Constants ──────────────────
B, C, T = 4, 29, 2000
EMBED = 64   # small for fast tests


def make_backbone_cfg(**kwargs) -> BackboneConfig:
    defaults = dict(
        n_channels=C, n_timesteps=T,
        cnn_out_channels=32, cnn_kernel_size=5, cnn_n_layers=2,
        gat_hidden_dim=EMBED, gat_n_layers=2, gat_n_heads=4,
        mamba_d_model=EMBED, mamba_d_state=8, mamba_d_conv=4,
        mamba_expand=2, mamba_n_layers=2,
        embed_dim=EMBED,
    )
    defaults.update(kwargs)
    return BackboneConfig(**defaults)


def make_model(track: int = 1) -> NeuroMMModel:
    bcfg = make_backbone_cfg()
    return NeuroMMModel(
        backbone_cfg=bcfg,
        t1_hidden_dims=[32], t1_dropout=0.1,
        t2_video_configs={"dinov2-large": {"n_frames": 8, "feat_dim": 1024}},
        t2_target_dim=EMBED, t2_n_ca_heads=4, t2_n_ca_layers=1,
        t2_mc_passes=2, t2_hidden_dims=[32],
        t3_video_config={"n_frames": 8, "feat_dim": 1024},
        t3_hidden_dim=32, t3_n_classes=5, t3_n_heads=4, t3_n_layers=1,
    )


# ── RMSNorm ─

class TestRMSNorm:
    def test_shape(self):
        norm = RMSNorm(EMBED)
        x = torch.randn(B, C, EMBED)
        assert norm(x).shape == x.shape

    def test_no_nan(self):
        norm = RMSNorm(EMBED)
        assert not torch.isnan(norm(torch.randn(B, C, EMBED))).any()


# ── NativeMambaBlock ───────────

class TestNativeMambaBlock:
    def test_shape(self):
        block = NativeMambaBlock(d_model=EMBED, d_state=8, d_conv=4, expand=2)
        x = torch.randn(B, C, EMBED)
        assert block(x).shape == x.shape

    def test_residual(self):
        block = NativeMambaBlock(d_model=EMBED, d_state=8)
        x = torch.randn(B, C, EMBED)
        out = block(x)
        # Output should not be identical to input (transformation happened)
        assert not torch.allclose(out, x)

    def test_no_nan(self):
        block = NativeMambaBlock(d_model=EMBED)
        assert not torch.isnan(block(torch.randn(B, C, EMBED))).any()


# ── EEGMambaEncoder ────────────

class TestEEGMambaEncoder:
    def test_output_shape(self):
        enc = EEGMambaEncoder(d_model=EMBED, n_layers=2, force_native=True)
        x = torch.randn(B, C, EMBED)
        out = enc(x)
        assert out.shape == (B, EMBED)

    def test_no_nan(self):
        enc = EEGMambaEncoder(d_model=EMBED, n_layers=2, force_native=True)
        assert not torch.isnan(enc(torch.randn(B, C, EMBED))).any()

    def test_gradient_flow(self):
        enc = EEGMambaEncoder(d_model=EMBED, n_layers=2, force_native=True)
        x = torch.randn(B, C, EMBED, requires_grad=True)
        loss = enc(x).sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ── DynamicGAT ─────────────────

class TestDynamicGAT:
    def test_shape(self):
        gat = DynamicGAT(in_dim=32, hidden_dim=EMBED, n_layers=2, n_heads=4)
        x = torch.randn(B, C, 32)
        assert gat(x).shape == (B, C, EMBED)

    def test_no_nan(self):
        gat = DynamicGAT(in_dim=32, hidden_dim=EMBED, n_layers=2, n_heads=4)
        assert not torch.isnan(gat(torch.randn(B, C, 32))).any()

    def test_layer_residual(self):
        from src.models.dynamic_gat import DynamicGATLayer
        layer = DynamicGATLayer(in_dim=EMBED, out_dim=EMBED, n_heads=4)
        x = torch.randn(B, C, EMBED)
        out = layer(x)
        assert out.shape == x.shape


# ── Backbone 

class TestBackbone:
    def test_embed_shape(self):
        bb = EEGBackbone(make_backbone_cfg())
        out = bb(torch.randn(B, C, T))
        assert out.shape == (B, EMBED)

    def test_channel_feats(self):
        bb = EEGBackbone(make_backbone_cfg())
        embed, ch = bb(torch.randn(B, C, T), return_channel_feats=True)
        assert embed.shape == (B, EMBED)
        assert ch.shape    == (B, C, EMBED)

    def test_freeze_unfreeze(self):
        bb = EEGBackbone(make_backbone_cfg())
        bb.freeze()
        assert all(not p.requires_grad for p in bb.parameters())
        bb.unfreeze()
        assert all(p.requires_grad for p in bb.parameters())

    def test_no_nan(self):
        bb = EEGBackbone(make_backbone_cfg())
        assert not torch.isnan(bb(torch.randn(B, C, T))).any()

    def test_gradient_flow(self):
        bb = EEGBackbone(make_backbone_cfg())
        x  = torch.randn(B, C, T, requires_grad=True)
        bb(x).sum().backward()
        assert x.grad is not None


# ── Full Model ──────────────────

class TestNeuroMMModel:
    def test_track1(self):
        m = make_model()
        out = m(torch.randn(B, C, T), mode="track1")
        assert out["logit"].shape == (B,)

    def test_track2(self):
        m = make_model()
        vf = {"dinov2-large": torch.randn(B, 8, 1024)}
        out = m(torch.randn(B, C, T), video_features=vf, mode="track2")
        assert out["logit"].shape == (B,)
        assert (out["gate"] >= 0).all() and (out["gate"] <= 1).all()

    def test_track3(self):
        m = make_model()
        vf = {"dinov2-large": torch.randn(B, 8, 1024)}
        out = m(torch.randn(B, C, T), video_features=vf, mode="track3")
        assert out["logit"].shape == (B, 5)

    def test_pretrain(self):
        m = make_model()
        eeg    = torch.randn(B, C, T)
        masked = eeg.clone(); masked[:, :, :400] = 0
        out = m(eeg, mode="pretrain", masked_eeg=masked)
        assert "binary_logit"    in out
        assert "labeltype_logit" in out
        assert "recon"           in out
        assert out["recon"].shape == (B, C, T)

    def test_count_parameters(self):
        m = make_model()
        counts = m.count_parameters()
        assert counts["total"] > 0 and counts["backbone"] > 0

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            make_model()(torch.randn(B, C, T), mode="invalid")

    def test_backward_track1(self):
        m = make_model()
        out = m(torch.randn(B, C, T), mode="track1")
        out["logit"].sum().backward()
        # Check at least one grad is not None
        assert any(p.grad is not None for p in m.parameters())


# ── Losses ───

class TestLosses:
    def _logits_targets(self, n=32):
        return torch.randn(n), torch.randint(0, 2, (n,)).float()

    def test_focal(self):
        fn = FocalLoss()
        l, t = self._logits_targets()
        loss = fn(l, t)
        assert loss.item() > 0 and not torch.isnan(loss)

    def test_poly(self):
        fn = PolyLoss()
        l, t = self._logits_targets()
        assert PolyLoss()(l, t).item() > 0

    def test_focal_poly(self):
        fn = FocalPolyLoss()
        l, t = self._logits_targets()
        assert fn(l, t).item() > 0

    def test_weighted_ce(self):
        fn = WeightedCELoss([100, 80, 60, 40, 20])
        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))
        assert WeightedCELoss([100, 80, 60, 40, 20])(logits, targets).item() > 0

    def test_multitask_loss(self):
        fn  = MultiTaskLoss()
        B2, C2, T2 = 4, 29, 2000
        out = {
            "binary_logit":    torch.randn(B2),
            "labeltype_logit": torch.randn(B2, 6),
            "recon":           torch.randn(B2, C2, T2),
            "recon_target":    torch.randn(B2, C2, T2),
        }
        losses = fn(out, torch.randint(0,2,(B2,)).float(), torch.randint(0,6,(B2,)))
        assert "loss" in losses and losses["loss"].item() > 0

    def test_backward_focal_poly(self):
        fn = FocalPolyLoss()
        logits  = torch.randn(32, requires_grad=True)
        targets = torch.randint(0, 2, (32,)).float()
        fn(logits, targets).backward()
        assert logits.grad is not None


# ── Metrics ──

class TestMetrics:
    def test_auprc_perfect(self):
        t = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auprc(t, s) > 0.95

    def test_weighted_f1(self):
        t = np.array([1, 2, 3, 4, 5])
        p = np.array([1, 2, 3, 4, 5])
        assert compute_weighted_f1(t, p) == pytest.approx(1.0)

    def test_oof_calibrator(self):
        calib = OOFCalibrator()
        for _ in range(3):
            t = np.random.randint(0, 2, 50)
            s = np.random.rand(50)
            calib.add_fold(t, s)
        calib.fit()
        result = calib.transform(np.array([0.1, 0.5, 0.9]))
        assert result.shape == (3,)
        assert all(0 <= v <= 1 for v in result)

    def test_metric_tracker_t1(self):
        tracker = MetricTracker(1)
        for _ in range(5):
            tracker.update(
                torch.randint(0, 2, (16,)).float(),
                torch.randn(16),
            )
        m = tracker.compute()
        assert "auprc" in m and 0 <= m["auprc"] <= 1

    def test_metric_tracker_t3(self):
        tracker = MetricTracker(3)
        for _ in range(5):
            tracker.update(
                torch.randint(1, 6, (8,)).float(),
                torch.randn(8, 5),
            )
        m = tracker.compute()
        assert "weighted_f1" in m


# ── Augmentations ───────────────

class TestAugmentations:
    def _eeg(self):
        return np.random.randn(29, 2000).astype(np.float32)

    def test_spec_augment_shape(self):
        aug = SpecAugment(max_time_mask=100, num_time_masks=2)
        eeg = self._eeg()
        out = aug(eeg)
        assert out.shape == eeg.shape

    def test_spec_augment_has_zeros(self):
        aug = SpecAugment(max_time_mask=500, num_time_masks=1)
        out = aug(self._eeg())
        assert (out == 0).any()

    def test_gaussian_noise(self):
        aug = GaussianNoise(sigma=0.1)
        eeg = self._eeg()
        out = aug(eeg)
        assert not np.allclose(out, eeg)

    def test_pipeline(self):
        aug = AugmentationPipeline({
            "spec_augment":   {"enabled": True, "max_time_mask": 100, "num_time_masks": 1},
            "gaussian_noise": {"enabled": True, "sigma": 0.05},
        })
        assert aug(self._eeg()).shape == (29, 2000)


# ── Config ───

class TestConfig:
    def test_load_base(self):
        from src.utils.config import load_config
        cfg = load_config("configs/base.yaml")
        assert "backbone" in cfg
        assert cfg["backbone"]["arch"] == "eegmamba"

    def test_load_track1(self):
        from src.utils.config import load_config
        cfg = load_config("configs/track1.yaml")
        assert cfg["track"] == 1
        assert "backbone" in cfg   # inherited from base

    def test_build_model_from_config(self):
        from src.utils.config import load_config
        cfg = load_config("configs/track2.yaml")
        model = build_model_from_config(cfg)
        assert isinstance(model, NeuroMMModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
