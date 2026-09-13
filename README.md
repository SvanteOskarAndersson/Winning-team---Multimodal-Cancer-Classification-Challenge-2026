# Multimodal Oral Cancer Cell Classification

Deep learning pipeline for classifying oral cytology cells as malignant or benign from
paired **brightfield (BF)** and **fluorescence (FL)** microscopy images, developed for the
Kaggle Multimodal Cancer Classification Challenge 2026.

- **Competition:** https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/overview
- **Leaderboard:** https://www.kaggle.com/competitions/multimodal-cancer-classification-challenge-2026/leaderboard

**Authors:** Edvard Schmidt, Leo Lindström Kemetli, Svante Andersson

---

## Overview

Each cell is imaged twice (BF and FL, 128 × 128, same filename). Labels are weak: every
cell inherits its patient's diagnosis, and the labelled set contains 12 patients. The
competition metric is ROC AUC.

The pipeline has two stages:

1. **Cross-modal SimCLR pretraining (optional)** – `mmcc pretrain`. A ConvNeXt V2-Tiny
   encoder is trained without labels on all 114,302 cells (train + test). The positive pair
   for each cell is its augmented BF view and its augmented FL view.
2. **Supervised training** – `mmcc train`. A 6-channel early-fusion ConvNeXt V2-Tiny,
   initialised from ImageNet or from the SimCLR encoder, is trained with patient-grouped
   2-fold CV, retrained on all labelled patients, and used to write `submission.csv`.

## Repository structure

```
.
├── src/mmcc/
│   ├── cli.py          # `mmcc pretrain` / `mmcc train`, all hyperparameters as flags
│   ├── pretrain.py     # SimCLR: cell pairs, augmentation, SimCLRNet, NT-Xent, resumable loop
│   ├── data.py         # data checks, FL channel probe, patient folds, norm stats, dataset
│   ├── models.py       # stem-patched timm backbone, SimCLR init, single/early/late fusion
│   ├── engine.py       # mixup, weighted BCE, epoch loop, cell/patient metrics
│   ├── train.py        # CV training, all-data retrain, threshold tuning
│   ├── inference.py    # TTA prediction, ensembling, submission + sanity check
│   └── utils.py        # device and seeding
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/edvardschmidt33/mmcc multimodal-cancer-classification
cd multimodal-cancer-classification
pip install -e .
```

ImageNet weights are downloaded by `timm` on first use.

## Data

Download the competition data. The expected layout is:

```
<data-root>/
├── train.csv              # Name, Diagnosis
├── sampleSubmission.csv   # Name, Diagnosis
├── BF/{train,test}/*.jpg
└── FL/{train,test}/*.jpg
```

On Kaggle this is `/kaggle/input/multimodal-cancer-classification-challenge-2026` (or
`/kaggle/input/competitions/multimodal-cancer-classification-challenge-2026`).

## Usage

**Supervised model from ImageNet weights:**

```bash
mmcc train --data-root /path/to/data --out-dir outputs/train
```

**Two-stage pipeline with SimCLR initialisation:**

```bash
# Stage 1 – checkpoints every epoch; if interrupted, rerun the same command to resume
mmcc pretrain --data-root /path/to/data --out-dir outputs/pretrain

# Stage 2
mmcc train --data-root /path/to/data --out-dir outputs/train_simclr \
    --simclr-ckpt outputs/pretrain/simclr_convnextv2_encoder.pt
```

Every value from the original configuration is a flag (`mmcc train --help`), e.g.:

```bash
mmcc train --data-root DATA --mode BF                   # single-modality ablation
mmcc train --data-root DATA --fusion L                  # late fusion
mmcc train --data-root DATA --no-retrain-all-data --submission-source cv --no-tta
```

`python -m mmcc ...` is equivalent to `mmcc ...`.

### Outputs

| Command    | File                           | Contents                                             |
|------------|--------------------------------|------------------------------------------------------|
| `pretrain` | `simclr_convnextv2_encoder.pt` | Encoder weights, arch, epoch, SSL norm stats         |
|            | `simclr_resume.pt`             | Full training state for resuming                     |
|            | `simclr_loss_curve.png`        | NT-Xent loss per epoch                               |
| `train`    | `fold{k}.pt`                   | Best checkpoint per CV fold                          |
|            | `alldata.pt`                   | All-data model (with `--retrain-all-data`)           |
|            | `submission.csv`               | Test-set scores in `sampleSubmission.csv` format     |

Per-epoch metrics, the per-fold summary, out-of-fold AUCs and the best F1 threshold are
printed to stdout.

## Method

### Data handling

- **Folds:** `StratifiedGroupKFold` groups by patient (no patient in both train and
  validation) and stratifies by diagnosis so folds have comparable cancer rates.
- **FL channels** are probed from the image files rather than assumed, so a 4-channel
  fluorescence format would not be truncated to RGB.
- **Normalisation** statistics are computed per modality from 2,000 sampled training images.
- **Class weights** are inverse cell-level class frequencies.

### Augmentation

Supervised training applies photometric augmentation per modality, then one geometric
transform to the concatenated BF+FL tensor so both modalities stay aligned.

| Stage              | Brightfield                                                                                  | Fluorescence                                                                  |
|--------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Photometric        | One of posterize / blur / solarize; brightness 0.6, contrast 0.4, saturation 0.4, **hue 0** | Colour jitter 0.8/0.8/0.8, hue 0.1 (RGB FL only); Gaussian blur σ ∈ [0.3, 3.2] |
| Geometric (shared) | H/V flips, rotation ±180°, affine (translate 8 %, scale 0.9–1.1, shear 8°), random erasing (p = 0.25) | same transform                                                  |

SimCLR uses random resized crops (scale 0.4–1.0), flips, rotation, colour jitter
(0.6/0.6/0.6, p = 0.8) and blur (p = 0.5), with **no hue jitter and no random grayscale**.

### Models

- **Backbone:** `convnextv2_tiny.fcmae_ft_in22k_in1k` (timm) at native 128 px.
- **Stem patching:** the stem convolution is widened to 3 + FL channels; RGB filters are
  copied and extra channels are initialised with the mean RGB filter.
- **Early fusion** (default): one backbone on the stacked BF+FL input → dropout → linear.
- **Late fusion** (`--fusion L`): separate BF and FL backbones, concatenated features →
  MLP (256) → linear.
- **SimCLR initialisation:** encoder weights are loaded before the stem is patched.

### Training

| Setting        | Pretraining (SimCLR)               | Supervised                             |
|----------------|------------------------------------|----------------------------------------|
| Loss           | NT-Xent, τ = 0.2, 128-d projection | Class-weighted BCE + mixup (α = 0.4)   |
| Optimiser      | AdamW, lr 5e-4, wd 1e-4            | AdamW, lr 4e-5, wd 0.2, dropout 0.15   |
| Schedule       | 10-epoch linear warmup → cosine    | Cosine to 1e-7                         |
| Epochs / batch | 60 / 128                           | 16 / 128                               |
| Precision      | AMP                                | AMP                                    |

- **Checkpoint selection:** each fold keeps the epoch with the best 3-epoch moving average of
  validation cell AUC. Patient-level AUC (mean cell score per patient) is logged alongside.
- **All-data retrain:** one model on all labelled patients for a fixed number of epochs;
  the final-epoch weights are kept since there is no validation signal.
- **Pretraining time budget:** training stops after 11 h (to fit a 12 h Kaggle session) and
  resumes from `simclr_resume.pt` on the next run.

### Inference

Each checkpoint predicts with flip TTA (identity, horizontal, vertical, both). The submission
uses the fold ensemble (`cv`), the all-data model (`alldata`), or their equal-weight average
(`both`, default), and is checked for NaNs, value range and row count.

## Design notes

Findings from development that shaped the configuration:

- **Multimodal fusion beats either modality alone.** BF-only and FL-only models both scored
  below the fused model, which motivated using BF↔FL pairs as SimCLR positives.
- **Stain colour is signal.** Removing hue jitter from brightfield augmentation raised the
  public leaderboard AUC from 0.819 to 0.845; SimCLR augmentation omits hue jitter and
  grayscale for the same reason.
- **Patient-identity shortcut.** With 12 patients, staining and illumination can identify the
  patient, which shows up as a gap between out-of-fold and leaderboard scores. Strong
  photometric augmentation targets this in supervised training; label-free pretraining has
  no labels through which to learn it.

## Reference

W. Lian, J. Lindblad, C. Runow Stark, J.-M. Hirsch, N. Sladoje. *Let it shine:
Autofluorescence of Papanicolaou-stain improves AI-based cytological oral cancer
detection.* Computers in Biology and Medicine, 2025. [arXiv:2407.01869](https://arxiv.org/abs/2407.01869)
