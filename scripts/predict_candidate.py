"""
scripts/predict_candidate.py — Candidate set inference + submission ZIP.

CRITICAL rules:
  - batch_size=1 (no global statistics)
  - Scalers fitted on full train set
  - Predict ALL candidate_ids.txt (scorer ignores distractors automatically)
  - Isotonic calibration applied if available
  - Output: CSV then zipped per Codabench format

Usage:
    python scripts/predict_candidate.py --track 1 --config configs/track1.yaml
    python scripts/predict_candidate.py --track 2 --config configs/track2.yaml
    python scripts/predict_candidate.py --track 3 --config configs/track3.yaml
"""
from __future__ import annotations
import argparse, pickle, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CandidateDataset
from src.data.preprocessing import load_scalers
from src.models import build_model_from_config
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_model(ckpt_path: Path, cfg: dict, device: torch.device):
    model = build_model_from_config(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def infer(model, dataset: CandidateDataset, track: int, device: torch.device) -> tuple[list, np.ndarray]:
    """
    Infer ONE sample at a time — batch_size=1 enforces independence.
    No BatchNorm statistics computed across the candidate set.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    ids, scores = [], []

    for batch in tqdm(loader, desc=f"Infer T{track}", leave=False):
        sid = batch["sample_id"][0]
        eeg = batch["eeg"].to(device)
        vf  = {k: v.to(device) for k, v in batch.get("video_features", {}).items()} or None

        out  = model(eeg, video_features=vf, mode=f"track{track}")
        logit = out["logit"]

        if track in (1, 2):
            s = torch.sigmoid(logit).item()
        else:
            s = int(torch.argmax(logit, dim=-1).item()) + 1  # classes 1..5

        ids.append(sid)
        scores.append(s)

    return ids, np.array(scores)


def main(args):
    cfg   = load_config(args.config)
    track = args.track
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Track {track} | Device: {device}")

    # Candidate IDs
    cand_ids_path = Path(cfg["data"]["candidate_ids"])
    with open(cand_ids_path) as f:
        candidate_ids = [l.strip() for l in f if l.strip()]
    logger.info(f"Candidate samples: {len(candidate_ids)}")

    # Scalers fitted on full training set
    ckpt_dir = Path(cfg["logging"]["save_dir"]) / f"track{track}"
    scaler_path = Path(cfg["logging"]["save_dir"]) / "pretrain" / "scalers_full_train.pkl"
    if not scaler_path.exists():
        scaler_path = ckpt_dir / "scalers_fold0.pkl"
        logger.warning(f"Full-train scalers not found, using {scaler_path}")
    scalers = load_scalers(scaler_path)

    # Video
    vid_bbs  = cfg.get("video", {}).get("active_backbones", []) if track > 1 else []
    vid_dir  = Path(cfg["data"]["candidate_video_dir"]) if track > 1 else None
    eeg_dir  = Path(cfg["data"]["candidate_eeg_dir"])

    dataset = CandidateDataset(candidate_ids, eeg_dir, scalers, track, vid_dir, vid_bbs)

    # Ensemble over fold checkpoints
    fold_ckpts = sorted(ckpt_dir.glob("fold*.pt"))
    if not fold_ckpts:
        raise FileNotFoundError(f"No fold checkpoints in {ckpt_dir}")
    logger.info(f"Ensembling {len(fold_ckpts)} checkpoints...")

    all_fold_scores = []
    sample_ids = []
    for ckpt_path in fold_ckpts:
        logger.info(f"  {ckpt_path.name}")
        model = load_model(ckpt_path, cfg, device)
        ids, scores = infer(model, dataset, track, device)
        sample_ids = ids   # same order every time
        all_fold_scores.append(scores)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    stack = np.stack(all_fold_scores, axis=0)
    if track in (1, 2):
        ensemble_scores = stack.mean(axis=0)
    else:
        # Majority vote for 5-class
        ensemble_scores = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int), minlength=7).argmax(),
            axis=0, arr=stack.astype(int),
        )

    # Isotonic calibration
    calib_path = ckpt_dir / "calibrator.pkl"
    if calib_path.exists() and track in (1, 2):
        with open(calib_path, "rb") as f:
            calib = pickle.load(f)
        ensemble_scores = calib.transform(ensemble_scores)
        logger.info("Isotonic calibration applied.")

    # Build submission CSV
    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"track{track}_predictions.csv"

    if track in (1, 2):
        df_out = pd.DataFrame({"sample_id": sample_ids, "score": ensemble_scores})
    else:
        df_out = pd.DataFrame({"sample_id": sample_ids, "prediction": ensemble_scores.astype(int)})

    df_out.to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path} ({len(df_out)} samples)")

    # Zip for Codabench
    zip_path = out_dir / f"track{track}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, csv_path.name)
    logger.info(f"Submission ZIP: {zip_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--track",  type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--config", required=True)
    main(p.parse_args())
