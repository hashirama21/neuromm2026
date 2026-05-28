# NeuroMM-2026 — EEGMamba Y-Architecture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hashirama21/neuromm2026/blob/main/neuromm2026_pipeline.ipynb)
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hashirama21/neuromm2026/blob/main/neuromm2026_pipeline.ipynb)

Production-grade solution for the [NeuroMM-2026](https://2026.neuromm.org) multimodal seizure detection challenge.

---

## Challenge Overview

NeuroMM-2026 is a multimodal seizure detection benchmark with **25,426 labeled EEG windows** (20,298 train / 5,128 val) and matching pre-extracted visual features from 7 vision backbones.

### Tasks

| Track | Input | Output | Metric |
|---|---|---|---|
| **Track 1** | EEG only | Binary spike/non-spike | AUPRC |
| **Track 2** | EEG + Video | Binary spike/non-spike | AUPRC |
| **Track 3** | EEG + Video (positives only) | 5-class seizure subtype | Weighted-F1 |

### Dataset Layout (HuggingFace)

```
NeuroMM-2026/                               ← HF dataset repo
├── annotations/
│   └── neuromm2026_train_val.csv           ← 25,426 rows
├── splits/
│   └── split.md                            ← patient-level partition doc
└── archives/
    ├── eeg.tar                             ← 25,426 EEG .npy  (29, 2000)
    ├── video_clip-base.tar                 ← CLIP ViT-B/32    (8, 512)
    ├── video_videomae-base.tar             ← VideoMAE-base    (1, 768)
    ├── video_videomae-large.tar            ← VideoMAE-large   (1, 1024)
    ├── video_dinov2-base.tar               ← DINOv2-base      (8, 768)
    ├── video_dinov2-large.tar              ← DINOv2-large     (8, 1024)
    ├── video_siglip-base.tar               ← SigLIP-base      (8, 768)
    └── video_timesformer-k400.tar          ← TimeSformer K400 (1, 768)
```

### Manifest Columns

| Column | Description |
|---|---|
| `sample_id` | Unique window identifier — matches `.npy` filename stem |
| `split` | `train` (20,298) or `val` (5,128); patient-disjoint |
| `label` | Binary: 1 = spike/seizure positive, 0 = negative |
| `label_type` | Multi-class: 0 = negative, 1–5 = seizure subtype |
| `subject_id` | Patient ID — use for patient-disjoint CV splits |

### EEG Format

Each `.npy` is shape `(29, 2000)` (`float16` or `float32`):
- 29 EEG channels at 500 Hz
- 2000 timesteps = 4-second window

```python
import numpy as np, pandas as pd

df = pd.read_csv("annotations/neuromm2026_train_val.csv")
print(df["label_type"].value_counts())           # multi-class distribution

sid = df.iloc[0]["sample_id"]
eeg = np.load(f"processed/features/eeg/{sid}.npy")
print(eeg.shape, eeg.dtype)                      # (29, 2000)

# Video features (DINOv2-large)
vid = np.load(f"processed/features/video/dinov2-large/{sid}.npy")
print(vid.shape, vid.dtype)                      # (8, 1024)
```

### Test-Phase Candidate Set

```
candidate/
├── candidate_ids.txt       ← one opaque id per line (no labels)
└── archives/
    ├── eeg.tar
    └── video_<backbone>.tar
```

Ground-truth labels are held privately by the organizers. Predictions are submitted to the official [Codabench](https://2026.neuromm.org) leaderboard.

---

## Architecture

```
EEG (29 × 2000)
      │
      ▼
┌┐
│  Shared EEGMamba Backbone                                │
│                                                          │
│  LocalCNN  (per-channel, spike morphology 20–70 ms)      │
│     ↓                                                    │
│  Dynamic GAT  (learned inter-channel adjacency)          │
│     ↓                                                    │
│  EEGMamba  (Mamba2 SSM, O(n), d_model=256, 4 layers)    │
│     ↓                                                    │
│  embed: (B, 256)                                         │
└┘
      │
      ├──────────────────┬───────────┐
      ▼                  ▼                              ▼
 Track 1            Track 2                       Track 3
 EEG only           EEG + Video                  EEG + Video
                    (Uncertainty-aware Gating)    (Positives only)
      │                  │                              │
 MLP + FocalPoly    MC-Dropout → Gate →           Channel Attn
                    Cross-Attn (VideoMAE-L,        + Weighted CE
                    DINOv2-L) → Fused MLP
      │                  │                              │
 AUPRC              AUPRC                         Weighted-F1
```

---


---

## Quick Start

### 1. Install

```bash
pip install -e .

# Real Mamba2 CUDA kernels (GPU, CUDA 11.8+) — strongly recommended
pip install mamba-ssm causal-conv1d

# LoRA fine-tuning (only needed when using --foundation_ckpt)
pip install peft
```

### 2. Download and prepare the dataset

The dataset requires gated HuggingFace access — request it at [NeuroMM/NeuroMM-2026](https://huggingface.co/datasets/NeuroMM/NeuroMM-2026), then:

```bash
# Default: EEG + videomae-large + dinov2-large (covers all three tracks)
python scripts/prepare_data.py --hf_token hf_XXXX

# Or pass the token via environment variable
export HF_TOKEN=hf_XXXX
python scripts/prepare_data.py

# Custom backbone selection
python scripts/prepare_data.py --hf_token hf_XXXX \
    --backbones videomae-large dinov2-large siglip-base

# All 7 backbones + test-phase candidate set
python scripts/prepare_data.py --hf_token hf_XXXX --backbones all --include_candidate

# Archives already on disk — re-run validation / index / smoke subset only
python scripts/prepare_data.py --skip_download --skip_extract
```

`prepare_data.py` runs 5 steps automatically:

| Step | What it does | Output |
|---|---|---|
| **Download** | Fetch selected archives from HF (only what you need) | `archives/*.tar` |
| **Extract** | Untar into the tree expected by `base.yaml` | `processed/features/eeg/`, `processed/features/video/<bb>/` |
| **Validate** | Check shapes `(29,2000)` for EEG, `(n_frames, feat_dim)` for each backbone | `annotations/validation_report.json` |
| **Index** | Flag per-sample availability per backbone | `annotations/neuromm2026_index.csv` |
| **Subset** | Sample N% of patients, patient-disjoint, stratified per split | `annotations/neuromm2026_smoke_10pct.csv` |

### 3. Smoke test — verify the full pipeline on 10% of data

**Always run this before a full training run.** It exercises data loading, augmentation, model forward, loss, and checkpointing in a few minutes:

```bash
python scripts/pretrain.py --config configs/base.yaml \
    --annotations annotations/neuromm2026_smoke_10pct.csv
```

To smoke-test at a different fraction:

```bash
python scripts/prepare_data.py --skip_download --skip_extract --smoke_pct 0.20
python scripts/pretrain.py --config configs/base.yaml \
    --annotations annotations/neuromm2026_smoke_20pct.csv
```

### 4. Full pretraining

Once the smoke test passes cleanly:

```bash
python scripts/pretrain.py --config configs/base.yaml
```

Optional — warm-start from an EEGMamba foundation checkpoint (LoRA adapters added automatically):

```bash
python scripts/pretrain.py --config configs/base.yaml \
    --foundation_ckpt pretrained/eegmamba_base.pt
```

### 5. Fine-tune per track

```bash
python scripts/train_track.py --track 1 --config configs/track1.yaml \
    --backbone_ckpt checkpoints/pretrain/best_backbone.pt

python scripts/train_track.py --track 2 --config configs/track2.yaml \
    --backbone_ckpt checkpoints/pretrain/best_backbone.pt

python scripts/train_track.py --track 3 --config configs/track3.yaml \
    --backbone_ckpt checkpoints/pretrain/best_backbone.pt
```

### 6. Generate submissions

```bash
python scripts/predict_candidate.py --track 1 --config configs/track1.yaml
python scripts/predict_candidate.py --track 2 --config configs/track2.yaml
python scripts/predict_candidate.py --track 3 --config configs/track3.yaml
```

### 7. Run tests

```bash
python -m pytest tests/ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **EEGMamba (Mamba2 SSM)** | O(n) vs O(n²) Transformer; 4–8× faster on T=2000 |
| **Dynamic GAT** | Learned adjacency (not fixed 10-20 topology) captures propagation patterns |
| **Multi-task pretrain** | Binary + masked reconstruction + label_type enriches the backbone for Track 3 |
| **Uncertainty-aware Gating** | Video is activated only when EEG is uncertain (MC-Dropout entropy) — mirrors clinical practice |
| **GroupKFold(subject_id)** | Patient-disjoint CV — mandatory to prevent label leakage |
| **RobustScaler per fold** | Fitted on fold-train only; median+IQR robust to artefact spikes |
| **batch_size=1 at inference** | No global statistics on the candidate set; distractors cannot bias predictions |
| **Isotonic Regression (OOF)** | Calibration on out-of-fold scores only — never on val/test |
| **Smoke subset (10%)** | Patient-stratified sample to validate the end-to-end pipeline before committing to full training |

---

## Anti-Leakage Rules (non-negotiable)

1. `GroupKFold(n=5, groups=subject_id)` — a patient is never in both train and val
2. `RobustScaler` fitted per fold on **fold-train only**
3. Candidate set inference: `batch_size=1`, no global normalization
4. Calibration (Isotonic Regression): fit on OOF scores, never on the official val set
5. No threshold optimization using the candidate set distribution

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — academic research and NeuroMM-2026 challenge participation only. No redistribution, no commercial use.

## Citation

```bibtex
@dataset{neuromm2026,
  title  = {NeuroMM-2026: Multimodal Seizure Detection Dataset},
  year   = {2026},
  url    = {https://2026.neuromm.org}
}
```
