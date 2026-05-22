"""
scripts/train_track.py — Generic CV trainer for Track 1, 2, or 3.

Usage:
    python scripts/train_track.py --track 1 --config configs/track1.yaml
    python scripts/train_track.py --track 2 --config configs/track2.yaml --backbone_ckpt checkpoints/pretrain/best_backbone.pt
    python scripts/train_track.py --track 3 --config configs/track3.yaml --backbone_ckpt checkpoints/pretrain/best_backbone.pt
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.data.dataset import build_dataloader, make_patient_disjoint_folds
from src.data.preprocessing import fit_per_channel_scalers, save_scalers
from src.evaluation.metrics import MetricTracker, OOFCalibrator, StackingEnsemble, compute_auprc, compute_weighted_f1
from src.models import build_model_from_config
from src.training.losses import FocalPolyLoss, WeightedCELoss
from src.utils.config import load_config
from src.utils.logging import get_logger, WandBLogger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def build_loss(cfg: dict, track: int, df_train=None):
    if track in (1, 2):
        l_cfg = cfg.get("loss", {})
        return FocalPolyLoss(
            focal_gamma=l_cfg.get("focal_gamma", 2.0),
            focal_alpha=l_cfg.get("focal_alpha", 0.75),
            poly_epsilon=l_cfg.get("poly_epsilon", 1.0),
            focal_weight=l_cfg.get("focal_weight", 1.0),
            poly_weight=l_cfg.get("poly_weight", 0.50),
        )
    else:
        n_cls = cfg.get("n_classes", 5)
        ls = cfg.get("loss", {}).get("label_smoothing", 0.10)
        if df_train is not None:
            return WeightedCELoss.from_dataframe(df_train, n_classes=n_cls, label_smoothing=ls)
        counts = [1] * n_cls  # fallback uniform
        return WeightedCELoss(counts, ls)


def build_optimizer(model, cfg: dict, track: int):
    t_cfg = cfg.get("training", {})
    bb_lr = t_cfg.get("backbone_lr", 1e-5)
    h_lr  = t_cfg.get("lr", 1e-4)
    wd    = t_cfg.get("weight_decay", 1e-4)

    param_groups = [
        {"params": model.backbone_params(), "lr": bb_lr, "name": "backbone"},
        {"params": model.head_params(track), "lr": h_lr,  "name": f"track{track}_head"},
    ]
    if track == 2:
        # Also include the gating module
        try:
            gating_params = list(model.track2.gating.parameters())
            param_groups.append({"params": gating_params,
                                  "lr": cfg["training"].get("gating_lr", h_lr),
                                  "name": "gating"})
        except AttributeError:
            pass

    return AdamW(param_groups, weight_decay=wd)


def primary_metric(metrics: dict, track: int) -> float:
    return metrics.get("auprc", metrics.get("weighted_f1", 0.0))


def run_fold(
    model,
    train_loader,
    val_loader,
    cfg: dict,
    track: int,
    device: torch.device,
    fold_idx: int,
    save_path: Path,
    df_train=None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train one CV fold. Returns (best_metric, oof_scores, oof_targets)."""

    t_cfg = cfg.get("training", {})
    n_epochs   = t_cfg.get("epochs", 50)
    patience   = t_cfg.get("early_stopping_patience", 10)
    grad_clip  = t_cfg.get("grad_clip", 1.0)
    warmup     = t_cfg.get("warmup_epochs", 3)
    freeze_ep  = t_cfg.get("freeze_backbone_epochs", 0)  # Track 2 only

    criterion = build_loss(cfg, track, df_train)
    optimizer = build_optimizer(model, cfg, track)

    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup, 1))
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(n_epochs - warmup, 1), eta_min=1e-7)
    scheduler    = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])

    best_m = -1.0
    patience_cnt = 0
    best_oof_scores = np.array([])
    best_oof_targets = np.array([])

    for epoch in range(n_epochs):

        # Staged unfreezing for Track 2
        if epoch == 0 and freeze_ep > 0:
            model.freeze_backbone()
        if epoch == freeze_ep and freeze_ep > 0:
            model.unfreeze_backbone()
            logger.info(f"  Fold {fold_idx+1} | Epoch {epoch+1}: backbone unfrozen")

        # ── Train ────────
        model.train()
        total_loss = 0.0
        n_b = 0
        for batch in tqdm(train_loader, desc=f"F{fold_idx+1}E{epoch+1} train", leave=False):
            eeg    = batch["eeg"].to(device)
            labels = batch["label"].to(device)
            vf     = {k: v.to(device) for k, v in batch.get("video_features", {}).items()}

            optimizer.zero_grad()
            out = model(eeg, video_features=vf or None, mode=f"track{track}")
            logit = out["logit"]

            if track == 3:
                # Map label_type 1..5 → 0..4 for CE
                tgt = batch["label_type"].to(device) - 1
                loss = criterion(logit, tgt)
            else:
                loss = criterion(logit, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = total_loss / max(n_b, 1)

        # ── Eval ─────────
        model.eval()
        tracker = MetricTracker(track)
        all_s, all_t = [], []
        with torch.no_grad():
            for batch in val_loader:
                eeg    = batch["eeg"].to(device)
                labels = batch["label"]
                vf     = {k: v.to(device) for k, v in batch.get("video_features", {}).items()}
                out    = model(eeg, video_features=vf or None, mode=f"track{track}")
                tracker.update(
                    batch["label_type"] if track == 3 else labels,
                    out["logit"],
                )
                if track in (1, 2):
                    s = torch.sigmoid(out["logit"]).cpu().numpy()
                else:
                    s = out["logit"].cpu().numpy()
                all_s.append(s)
                all_t.append((batch["label_type"] if track == 3 else labels).numpy())

        metrics = tracker.compute()
        pm = primary_metric(metrics, track)
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"  F{fold_idx+1} E{epoch+1:3d} | loss={avg_loss:.4f} | {metric_str}")

        if pm > best_m:
            best_m = pm
            patience_cnt = 0
            best_oof_scores  = np.concatenate(all_s)
            best_oof_targets = np.concatenate(all_t)
            torch.save({"model_state": model.state_dict(), "metric": pm}, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"  Fold {fold_idx+1} best: {best_m:.4f}")
    return best_m, best_oof_scores, best_oof_targets


def main(args):
    cfg = load_config(args.config)
    track = args.track
    set_seed(cfg.get("seed", 42))

    dev_str = cfg.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if dev_str == "auto" else torch.device(dev_str)
    logger.info(f"Track {track} | Device: {device}")

    df = pd.read_csv(cfg["data"]["annotations"])
    df_tr_all = df[df["split"] == "train"].reset_index(drop=True)
    eeg_dir   = Path(cfg["data"]["eeg_dir"])

    # Video
    video_dir = Path(cfg["data"]["video_dir"]) if track > 1 else None
    vid_bbs = cfg.get("video", {}).get("active_backbones", []) if track > 1 else []

    ckpt_dir = Path(cfg["logging"]["save_dir"]) / f"track{track}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    folds = make_patient_disjoint_folds(df_tr_all, cfg["cv"]["n_splits"], cfg["cv"]["group_col"])
    calibrator = OOFCalibrator()
    ensemble   = StackingEnsemble("binary" if track in (1,2) else "multiclass")
    ensemble.set_targets(df_tr_all["label"].values if track in (1,2) else df_tr_all["label_type"].values)
    fold_metrics = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        logger.info(f"\n{'='*55}")
        logger.info(f"Fold {fold_idx+1}/{len(folds)}")
        set_seed(cfg["seed"] + fold_idx)

        df_tr  = df_tr_all.iloc[tr_idx]
        df_va  = df_tr_all.iloc[va_idx]

        # Fit scalers on fold-train only
        scalers = fit_per_channel_scalers(df_tr, eeg_dir, cfg["data"]["n_channels"])
        save_scalers(scalers, ckpt_dir / f"scalers_fold{fold_idx}.pkl")

        train_loader = build_dataloader(
            df_tr, eeg_dir, scalers, track=track,
            video_dir=video_dir, video_backbones=vid_bbs,
            batch_size=cfg["training"]["batch_size"],
            augment=True, augment_cfg=cfg.get("augmentation", {}),
            num_workers=cfg.get("num_workers", 4),
            use_weighted_sampler=(track != 3),
            max_pos_ratio=cfg["training"].get("positive_oversample_ratio", 4),
        )
        val_loader = build_dataloader(
            df_va, eeg_dir, scalers, track=track,
            video_dir=video_dir, video_backbones=vid_bbs,
            batch_size=cfg["training"]["batch_size"],
            augment=False, shuffle=False,
            num_workers=cfg.get("num_workers", 4),
        )

        model = build_model_from_config(cfg).to(device)
        if args.backbone_ckpt:
            model.load_foundation_weights(args.backbone_ckpt, strict=False)

        best_m, oof_s, oof_t = run_fold(
            model, train_loader, val_loader, cfg, track, device,
            fold_idx, ckpt_dir / f"fold{fold_idx}.pt", df_tr,
        )
        fold_metrics.append(best_m)

        if track in (1, 2):
            calibrator.add_fold(oof_t, oof_s)
        ensemble.add_oof(oof_s)

    # Calibration
    if track in (1, 2):
        calibrator.fit()
        calibrator.save(ckpt_dir / "calibrator.pkl")
        logger.info(f"Calibrator saved.")

    mean_m = np.mean(fold_metrics)
    std_m  = np.std(fold_metrics)
    metric_name = "AUPRC" if track in (1,2) else "Weighted-F1"
    logger.info(f"\n{'='*55}")
    logger.info(f"Track {track} CV {metric_name}: {mean_m:.4f} ± {std_m:.4f}")
    logger.info(f"Per-fold: {[f'{v:.4f}' for v in fold_metrics]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track",         type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--config",        required=True)
    p.add_argument("--backbone_ckpt", default=None)
    main(p.parse_args())
