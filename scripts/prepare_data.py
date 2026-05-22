"""
scripts/prepare_data.py — Download, extract, validate and index NeuroMM-2026.

Steps
-----
1. Download  : fetch selected archives from HuggingFace (gated, needs token)
2. Extract   : untar archives into the directory tree expected by base.yaml
3. Validate  : verify every sample_id has matching .npy; check shapes/dtypes
4. Index     : write annotations/neuromm2026_index.csv with per-backbone flags
5. Subset    : sample N% of patients → smoke-test CSV for end-to-end checks

Data layout after preparation (relative to project root / data_dir)
---------------------------------------------------------------------
annotations/
  neuromm2026_train_val.csv     ← official manifest (downloaded)
  neuromm2026_index.csv         ← generated: + availability flags
  neuromm2026_smoke_10pct.csv   ← generated: N% patient subset
  validation_report.json        ← generated: shape/missing report
archives/
  eeg.tar                       ← downloaded (kept for reference)
  video_<backbone>.tar
processed/features/
  eeg/<sample_id>.npy           ← (29, 2000)
  video/
    videomae-large/<sample_id>.npy   ← (1, 1024)
    dinov2-large/<sample_id>.npy     ← (8, 1024)
    ...
candidate/
  candidate_ids.txt
  archives/eeg.tar
  processed/features/eeg/<id>.npy
  processed/features/video/<backbone>/<id>.npy

Usage
-----
# First run — full pipeline with two default video backbones
python scripts/prepare_data.py --hf_token hf_XXXX

# Custom backbone selection
python scripts/prepare_data.py --hf_token hf_XXXX \\
    --backbones videomae-large dinov2-large siglip-base

# Download all 7 backbones
python scripts/prepare_data.py --hf_token hf_XXXX --backbones all

# Also grab the test-phase candidate set
python scripts/prepare_data.py --hf_token hf_XXXX --include_candidate

# Skip download/extract if archives are already on disk
python scripts/prepare_data.py --skip_download --skip_extract

# Only re-create the smoke subset at a different percentage
python scripts/prepare_data.py --skip_download --skip_extract --smoke_pct 0.20

# Validate only (no download, no extract, no subset)
python scripts/prepare_data.py --skip_download --skip_extract --no_subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────
# Constants
# ─────────

HF_REPO_ID = "NeuroMM/NeuroMM-2026"
ANNOTATIONS_SUBPATH = "annotations/neuromm2026_train_val.csv"
EEG_SHAPE = (29, 2000)

# All 7 backbones provided by the dataset
BACKBONE_SPECS: dict[str, dict] = {
    "clip-base":        {"archive": "video_clip-base.tar",        "n_frames": 8,  "feat_dim": 512},
    "videomae-base":    {"archive": "video_videomae-base.tar",    "n_frames": 1,  "feat_dim": 768},
    "videomae-large":   {"archive": "video_videomae-large.tar",   "n_frames": 1,  "feat_dim": 1024},
    "dinov2-base":      {"archive": "video_dinov2-base.tar",      "n_frames": 8,  "feat_dim": 768},
    "dinov2-large":     {"archive": "video_dinov2-large.tar",     "n_frames": 8,  "feat_dim": 1024},
    "siglip-base":      {"archive": "video_siglip-base.tar",      "n_frames": 8,  "feat_dim": 768},
    "timesformer-k400": {"archive": "video_timesformer-k400.tar", "n_frames": 1,  "feat_dim": 768},
}

# Default: only the two backbones used by base.yaml (Track 2 + 3)
DEFAULT_BACKBONES = ["videomae-large", "dinov2-large"]


# ─────────
# Logging helper
# ─────────

def _log(msg: str, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{msg}", flush=True)


# ─────────
# Step 1 — Download from HuggingFace
# ─────────

def step_download(
    data_dir: Path,
    backbones: list[str],
    hf_token: str | None,
    include_candidate: bool,
) -> None:
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        raise SystemExit(
            "[Download] huggingface_hub not installed.\n"
            "  pip install huggingface_hub"
        )

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "[Download] No HuggingFace token found.\n"
            "  Pass --hf_token hf_XXXX  or  set env var HF_TOKEN=hf_XXXX\n"
            "  Get a token at: https://huggingface.co/settings/tokens"
        )

    login(token=token, add_to_git_credential=False)

    # Build the list of files to download (avoid pulling 15 GB when not needed)
    patterns: list[str] = [
        "annotations/*",
        "splits/*",
        "README.md",
        "archives/eeg.tar",
    ]
    for bb in backbones:
        patterns.append(f"archives/{BACKBONE_SPECS[bb]['archive']}")

    if include_candidate:
        patterns += ["candidate/candidate_ids.txt", "candidate/README.md",
                     "candidate/archives/eeg.tar"]
        for bb in backbones:
            patterns.append(f"candidate/archives/{BACKBONE_SPECS[bb]['archive']}")

    _log(f"\n[1/5] Downloading from {HF_REPO_ID}")
    _log(f"destination : {data_dir}", 1)
    _log(f"backbones   : {backbones}", 1)
    _log(f"candidate   : {include_candidate}", 1)
    _log(f"patterns    : {patterns}", 1)

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(data_dir),
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
        token=token,
    )
    _log("[1/5] Download complete.\n")


# ─────────
# Step 2 — Extract archives
# ─────────

def _extract_one(archive_path: Path, extract_root: Path) -> int:
    """Extract a single .tar into extract_root. Returns number of files extracted."""
    if not archive_path.exists():
        _log(f"[!] Archive not found, skipping: {archive_path}", 1)
        return 0

    _log(f"  {archive_path.name}", 1)
    t0 = time.time()
    with tarfile.open(archive_path, "r") as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc=f"    {archive_path.stem}", unit="file",
                      ncols=90, leave=False):
            tf.extract(m, path=extract_root)
    elapsed = time.time() - t0
    _log(f"    {len(members):,} files in {elapsed:.1f}s", 1)
    return len(members)


def step_extract(data_dir: Path, backbones: list[str], include_candidate: bool) -> None:
    _log("\n[2/5] Extracting archives")
    archives = data_dir / "archives"

    _extract_one(archives / "eeg.tar", data_dir)
    for bb in backbones:
        _extract_one(archives / BACKBONE_SPECS[bb]["archive"], data_dir)

    if include_candidate:
        cand_archives = data_dir / "candidate" / "archives"
        _extract_one(cand_archives / "eeg.tar", data_dir)
        for bb in backbones:
            _extract_one(cand_archives / BACKBONE_SPECS[bb]["archive"], data_dir)

    _log("[2/5] Extraction complete.\n")


# ─────────
# Step 3 — Validate
# ─────────

def step_validate(
    data_dir: Path,
    backbones: list[str],
    sample_limit: int | None,
) -> dict:
    """
    Check every sample_id in the CSV has matching .npy files with correct shapes.
    Uses mmap_mode='r' so only the array header is read — fast even for 25k files.
    """
    ann_path = data_dir / ANNOTATIONS_SUBPATH
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotations CSV not found: {ann_path}\n"
            "  Run --skip_download=False to download it first."
        )

    df = pd.read_csv(ann_path)
    _log(f"\n[3/5] Validating dataset — {len(df):,} samples total")
    _log(f"Split distribution:", 1)
    for split, cnt in df["split"].value_counts().items():
        _log(f"  {split}: {cnt:,}", 1)
    _log(f"Label distribution (label_type):", 1)
    for lt, cnt in df["label_type"].value_counts().sort_index().items():
        tag = "negative" if lt == 0 else f"subtype-{lt}"
        _log(f"  {lt} ({tag}): {cnt:,}", 1)

    eeg_dir   = data_dir / "processed" / "features" / "eeg"
    video_dir = data_dir / "processed" / "features" / "video"

    ids = df["sample_id"].tolist()
    if sample_limit and sample_limit < len(ids):
        rng = np.random.default_rng(42)
        ids = rng.choice(ids, sample_limit, replace=False).tolist()
        _log(f"Validating {len(ids):,} samples (random sample for speed — "
             f"use --validate_n 0 for all)", 1)
    else:
        _log(f"Validating all {len(ids):,} samples...", 1)

    missing_eeg: list[str] = []
    bad_eeg: list[tuple] = []
    missing_video: dict[str, list] = {bb: [] for bb in backbones}
    bad_video: dict[str, list]    = {bb: [] for bb in backbones}

    for sid in tqdm(ids, desc="  Checking", unit="sample", ncols=90):
        # EEG
        p = eeg_dir / f"{sid}.npy"
        if not p.exists():
            missing_eeg.append(sid)
        else:
            shape = np.load(p, mmap_mode="r").shape
            if shape != EEG_SHAPE:
                bad_eeg.append((sid, shape))

        # Video per backbone
        for bb in backbones:
            vp = video_dir / bb / f"{sid}.npy"
            if not vp.exists():
                missing_video[bb].append(sid)
            else:
                spec = BACKBONE_SPECS[bb]
                expected = (spec["n_frames"], spec["feat_dim"])
                shape = np.load(vp, mmap_mode="r").shape
                if shape != expected:
                    bad_video[bb].append((sid, shape))

    # Build report
    report: dict = {
        "n_samples_in_csv": len(df),
        "n_validated": len(ids),
        "eeg_expected_shape": list(EEG_SHAPE),
        "eeg_missing": len(missing_eeg),
        "eeg_bad_shape": len(bad_eeg),
        "eeg_missing_examples": missing_eeg[:10],
        "eeg_bad_shape_examples": bad_eeg[:5],
        "video": {},
    }
    for bb in backbones:
        spec = BACKBONE_SPECS[bb]
        report["video"][bb] = {
            "expected_shape": [spec["n_frames"], spec["feat_dim"]],
            "missing": len(missing_video[bb]),
            "bad_shape": len(bad_video[bb]),
            "missing_examples": missing_video[bb][:10],
            "bad_shape_examples": bad_video[bb][:5],
        }

    _log("\n  Validation summary:", 1)
    eeg_ok = len(ids) - len(missing_eeg) - len(bad_eeg)
    _log(f"  EEG          : {eeg_ok:,}/{len(ids):,} OK  "
         f"(missing={len(missing_eeg)}, bad_shape={len(bad_eeg)})", 1)
    for bb in backbones:
        n_ok = len(ids) - len(missing_video[bb]) - len(bad_video[bb])
        _log(f"  {bb:20s}: {n_ok:,}/{len(ids):,} OK  "
             f"(missing={len(missing_video[bb])}, bad_shape={len(bad_video[bb])})", 1)

    has_issues = len(missing_eeg) + len(bad_eeg) + sum(
        len(missing_video[bb]) + len(bad_video[bb]) for bb in backbones
    )
    if has_issues:
        _log(f"\n  [!] {has_issues} issues found — see validation_report.json", 1)
    else:
        _log("\n  All files OK.", 1)

    _log("[3/5] Validation complete.\n")
    return report


# ─────────
# Step 4 — Build index
# ─────────

def step_index(data_dir: Path, backbones: list[str]) -> pd.DataFrame:
    """
    Augment the official CSV with per-backbone availability flags.
    Also converts EEG float16 dtype flag for reference.
    Saves: annotations/neuromm2026_index.csv
    """
    _log("[4/5] Building availability index")

    df = pd.read_csv(data_dir / ANNOTATIONS_SUBPATH)
    eeg_dir   = data_dir / "processed" / "features" / "eeg"
    video_dir = data_dir / "processed" / "features" / "video"

    # EEG availability
    df["eeg_ok"] = df["sample_id"].apply(
        lambda sid: (eeg_dir / f"{sid}.npy").exists()
    )

    # Video availability per backbone
    for bb in backbones:
        col = f"video_{bb.replace('-', '_').replace('.', '_')}_ok"
        vdir = video_dir / bb
        df[col] = df["sample_id"].apply(lambda sid: (vdir / f"{sid}.npy").exists())

    # Summary stats
    all_ok_mask = df["eeg_ok"].copy()
    for bb in backbones:
        col = f"video_{bb.replace('-', '_').replace('.', '_')}_ok"
        all_ok_mask = all_ok_mask & df[col]
    df["all_modalities_ok"] = all_ok_mask

    out = data_dir / "annotations" / "neuromm2026_index.csv"
    df.to_csv(out, index=False)

    _log(f"  Columns added: eeg_ok, video_*_ok, all_modalities_ok", 1)
    _log(f"  Rows with all modalities OK: "
         f"{df['all_modalities_ok'].sum():,}/{len(df):,}", 1)
    _log(f"  Saved → {out}", 1)
    _log("[4/5] Index complete.\n")

    return df


# ─────────
# Step 5 — Create smoke-test subset
# ─────────

def step_smoke_subset(
    data_dir: Path,
    df: pd.DataFrame,
    smoke_pct: float,
    seed: int,
) -> pd.DataFrame:
    """
    Sample smoke_pct of unique patients (grouped by subject_id), separately
    per split so the train/val ratio is preserved.

    Guarantees:
    - Patient-disjoint (no subject_id in both train and val subsets)
    - Stratified per split (same patient fraction from train and val)
    - Deterministic with fixed seed

    Saves: annotations/neuromm2026_smoke_{N}pct.csv
    """
    _log(f"[5/5] Creating {smoke_pct*100:.0f}% smoke-test subset (seed={seed})")

    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    for split in ["train", "val"]:
        split_df = df[df["split"] == split].copy()
        subjects = split_df["subject_id"].unique().to_numpy(dtype=str)
        rng.shuffle(subjects)

        n_keep = max(1, int(len(subjects) * smoke_pct))
        chosen = set(subjects[:n_keep])
        sub = split_df[split_df["subject_id"].isin(chosen)].copy()
        parts.append(sub)

        n_pos = int(sub["label"].sum())
        n_neg = len(sub) - n_pos
        _log(f"  {split:5s}: {n_keep}/{len(subjects)} patients → "
             f"{len(sub):,} samples  (pos={n_pos:,}, neg={n_neg:,})", 1)

    smoke = pd.concat(parts).reset_index(drop=True)

    # Strict leakage check
    train_subj = set(smoke[smoke["split"] == "train"]["subject_id"])
    val_subj   = set(smoke[smoke["split"] == "val"]["subject_id"])
    overlap = train_subj & val_subj
    if overlap:
        raise AssertionError(f"Patient leakage detected in smoke subset: {overlap}")

    pct_str = f"{int(smoke_pct * 100)}"
    out = data_dir / "annotations" / f"neuromm2026_smoke_{pct_str}pct.csv"
    smoke.to_csv(out, index=False)

    n_pos_total = int(smoke["label"].sum())
    _log(f"\n  Total: {len(smoke):,} samples  "
         f"(pos={n_pos_total:,}, neg={len(smoke)-n_pos_total:,})", 1)
    _log(f"  No patient overlap between train and val: OK", 1)
    _log(f"  Saved → {out}", 1)
    _log(f"[5/5] Smoke subset complete.\n")

    return smoke


# ─────────
# Main
# ─────────

def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Resolve backbone list
    if args.backbones == ["all"]:
        backbones = list(BACKBONE_SPECS.keys())
    else:
        backbones = args.backbones

    unknown = [b for b in backbones if b not in BACKBONE_SPECS]
    if unknown:
        raise SystemExit(
            f"Unknown backbone(s): {unknown}\n"
            f"Available: {list(BACKBONE_SPECS)}"
        )

    _log("=" * 65)
    _log("NeuroMM-2026 — Data Preparation")
    _log("=" * 65)
    _log(f"data_dir   : {data_dir}")
    _log(f"backbones  : {backbones}")
    _log(f"smoke_pct  : {args.smoke_pct * 100:.0f}%")
    _log(f"candidate  : {args.include_candidate}")
    _log("=" * 65)

    # ── 1. Download ────────────
    if not args.skip_download:
        step_download(
            data_dir=data_dir,
            backbones=backbones,
            hf_token=args.hf_token,
            include_candidate=args.include_candidate,
        )
    else:
        _log("\n[1/5] Download skipped.")

    # ── 2. Extract ─────────────
    if not args.skip_extract:
        step_extract(data_dir, backbones, args.include_candidate)
    else:
        _log("[2/5] Extraction skipped.")

    # ── 3. Validate ────────────
    validate_n = None if args.validate_n == 0 else args.validate_n
    report = step_validate(data_dir, backbones, validate_n)

    report_path = data_dir / "annotations" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    _log(f"  Validation report → {report_path}", 1)

    # ── 4. Index ───────────────
    df = step_index(data_dir, backbones)

    # ── 5. Smoke subset ────────
    if not args.no_subset:
        step_smoke_subset(data_dir, df, args.smoke_pct, args.seed)

    # ── Done 
    _log("=" * 65)
    _log("Data preparation complete.")
    _log("=" * 65)
    pct_str = f"{int(args.smoke_pct * 100)}"
    _log("\nNext steps:")
    _log(f"  # Smoke test — verify pipeline end-to-end on {pct_str}% of data:")
    _log(f"  python scripts/pretrain.py --config configs/base.yaml "
         f"--annotations annotations/neuromm2026_smoke_{pct_str}pct.csv")
    _log(f"\n  # Full training after smoke test passes:")
    _log(f"  python scripts/pretrain.py --config configs/base.yaml")


# ─────────
# CLI
# ─────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download, extract, validate and index NeuroMM-2026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    p.add_argument(
        "--data_dir", default=".",
        help="Project root directory (default: current dir). "
             "Must match path assumptions in configs/base.yaml.",
    )

    # HuggingFace
    p.add_argument(
        "--hf_token", default=None,
        help="HuggingFace access token. Alternatively set env var HF_TOKEN.",
    )
    p.add_argument(
        "--backbones", nargs="+", default=DEFAULT_BACKBONES,
        metavar="BACKBONE",
        help=f"Video backbones to download/validate. "
             f"Options: {list(BACKBONE_SPECS)} | all. "
             f"Default: {DEFAULT_BACKBONES}",
    )
    p.add_argument(
        "--include_candidate", action="store_true",
        help="Also download and extract the test-phase candidate set.",
    )

    # Pipeline control
    p.add_argument(
        "--skip_download", action="store_true",
        help="Skip HuggingFace download (archives already on disk).",
    )
    p.add_argument(
        "--skip_extract", action="store_true",
        help="Skip tar extraction (already extracted).",
    )

    # Validation
    p.add_argument(
        "--validate_n", type=int, default=2000,
        help="Number of samples to validate shapes on (0 = all 25k, slower). "
             "Default: 2000.",
    )

    # Smoke subset
    p.add_argument(
        "--smoke_pct", type=float, default=0.10,
        help="Fraction of patients to keep in smoke-test subset (default: 0.10 = 10%%).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for smoke-test patient sampling (default: 42).",
    )
    p.add_argument(
        "--no_subset", action="store_true",
        help="Skip smoke-test subset creation.",
    )

    main(p.parse_args())
