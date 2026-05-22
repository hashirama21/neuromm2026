"""
scripts/pretrain.py — Multi-task EEGMamba backbone pretraining.

Usage:
    python scripts/pretrain.py --config configs/base.yaml
    python scripts/pretrain.py --config configs/base.yaml --foundation_ckpt pretrained/eegmamba.pt

GPU run (recommended):
    python scripts/pretrain.py --config configs/base.yaml --batch_size 128
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.data.dataset import build_dataloader
from src.data.preprocessing import fit_per_channel_scalers, save_scalers
from src.models import NeuroMMModel, build_model_from_config
from src.training.losses import MultiTaskLoss
from src.utils.config import load_config
from src.utils.logging import get_logger, WandBLogger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def apply_mask(eeg: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask ratio fraction of timesteps (set to 0)."""
    B, C, T = eeg.shape
    n_mask = int(T * ratio)
    masked = eeg.clone()
    mask = torch.zeros(B, T, dtype=torch.bool, device=eeg.device)
    for b in range(B):
        idx = torch.randperm(T, device=eeg.device)[:n_mask]
        mask[b, idx] = True
        masked[b, :, idx] = 0.0
    return masked, mask


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device_str = cfg.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device_str == "auto" else torch.device(device_str)
    use_amp = device.type == "cuda"
    logger.info(f"Device: {device}  |  AMP: {use_amp}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # ── Data ──────────────────────────────────────────────────────────────
    annotations = args.annotations or cfg["data"]["annotations"]
    df = pd.read_csv(annotations)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    logger.info(f"Annotations: {annotations}")
    logger.info(f"Training samples: {len(df_train)}")

    eeg_dir = Path(cfg["data"]["eeg_dir"])
    logger.info("Fitting per-channel RobustScalers on full training set...")
    scalers = fit_per_channel_scalers(df_train, eeg_dir, cfg["data"]["n_channels"])

    ckpt_dir = Path(cfg["logging"]["save_dir"]) / "pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_scalers(scalers, ckpt_dir / "scalers_full_train.pkl")
    logger.info(f"Scalers saved to {ckpt_dir / 'scalers_full_train.pkl'}")

    pt_cfg = cfg["pretrain"]
    batch_size = args.batch_size or pt_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)

    train_loader = build_dataloader(
        df_train, eeg_dir, scalers,
        track=1,
        batch_size=batch_size,
        augment=True, augment_cfg=cfg.get("augmentation", {}),
        num_workers=num_workers,
        use_weighted_sampler=True,
    )
    logger.info(f"Dataloader: batch_size={batch_size}, num_workers={num_workers}, "
                f"batches/epoch={len(train_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model_from_config(cfg).to(device)

    foundation = args.foundation_ckpt or pt_cfg.get("foundation_ckpt")
    if foundation:
        model.load_foundation_weights(foundation, strict=False)
        if pt_cfg.get("lora_r"):
            model.apply_lora(r=pt_cfg["lora_r"], alpha=pt_cfg["lora_alpha"])

    counts = model.count_parameters()
    logger.info(f"Parameters: total={counts['total']:,}  backbone={counts['backbone']:,}  "
                f"pretrain_heads={counts['pretrain_heads']:,}")

    # ── Loss & Optimiser ──────────────────────────────────────────────────
    criterion = MultiTaskLoss(
        lambda_binary=pt_cfg["lambda_binary"],
        lambda_recon=pt_cfg["lambda_recon"],
        lambda_labeltype=pt_cfg["lambda_labeltype"],
        focal_gamma=pt_cfg.get("focal_gamma", 2.0),
        focal_alpha=pt_cfg.get("focal_alpha", 0.75),
        poly_epsilon=pt_cfg.get("poly_epsilon", 1.0),
    )

    optimizer = AdamW(
        model.pretrain_params(),
        lr=pt_cfg["lr"],
        weight_decay=pt_cfg["weight_decay"],
    )

    n_epochs = args.max_epochs if args.max_epochs else pt_cfg["epochs"]
    warmup_epochs = min(pt_cfg.get("warmup_epochs", 5), n_epochs - 1)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, n_epochs - warmup_epochs), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    # AMP scaler — no-op on CPU
    scaler = GradScaler(device=device.type, enabled=use_amp)

    wandb_log = WandBLogger(cfg, cfg.get("logging", {}).get("use_wandb", False))

    # ── Training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    mask_ratio = pt_cfg.get("mask_ratio", 0.20)

    for epoch in range(n_epochs):
        model.train()
        agg = {"loss": 0., "loss_binary": 0., "loss_recon": 0., "loss_labeltype": 0.}
        n = 0

        for batch in tqdm(train_loader, desc=f"Pretrain E{epoch+1:03d}", leave=False):
            eeg         = batch["eeg"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)
            label_types = batch["label_type"].to(device, non_blocking=True)
            masked_eeg, _ = apply_mask(eeg, mask_ratio)

            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                out = model(eeg, mode="pretrain", masked_eeg=masked_eeg)
                losses = criterion(out, labels, label_types)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), pt_cfg.get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()

            for k in agg:
                agg[k] += losses[k].item()
            n += 1

        scheduler.step()
        avg = {k: v / n for k, v in agg.items()}
        logger.info(
            f"E{epoch+1:03d} | " + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
        )
        wandb_log.log(avg, step=epoch)

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            torch.save({
                "backbone_state": model.backbone.state_dict(),
                "epoch": epoch, "loss": best_loss,
            }, ckpt_dir / "best_backbone.pt")
            logger.info(f"  [SAVED] Backbone (loss={best_loss:.4f})")

    wandb_log.finish()
    logger.info(f"Pretraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",          default="configs/base.yaml")
    p.add_argument("--foundation_ckpt", default=None)
    p.add_argument("--annotations",     default=None,
                   help="Override annotations CSV for smoke tests.")
    p.add_argument("--max_epochs",      type=int,   default=None,
                   help="Cap epochs (overrides config). Use 2-3 for smoke tests.")
    p.add_argument("--batch_size",      type=int,   default=None,
                   help="Override batch size (e.g. 128 for T4, 256 for L4).")
    p.add_argument("--num_workers",     type=int,   default=None,
                   help="Override DataLoader num_workers (e.g. 4 on cloud GPU).")
    main(p.parse_args())
