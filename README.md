<div align="center">

# Dashcam Video Classification Ensemble

*Training reproducible VideoMAE models to classify dashcam video clips*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)
![Transformers](https://img.shields.io/badge/Hugging%20Face-VideoMAE-yellow.svg)

[Project](#1-project-overview) • [Approach](#2-approach) • [Run](#3-setup-and-usage) • [Structure](#4-project-structure-and-outputs)

</div>

---

## 1. Project Overview

This project provides a reproducible training pipeline for binary classification of dashcam video clips. It fine-tunes the pretrained **VideoMAE** video transformer and trains multiple ensemble members using predefined seeds and hyperparameter configurations.

The code is organized into small, focused Python modules rather than a single training script. Data paths are supplied at runtime, so the project can be adapted to a local dataset or a compute cluster without embedding machine-specific paths in the source.

### Project goals

- Fine-tune a pretrained video-classification model on annotated dashcam clips.
- Apply temporal augmentation during training, with optional color and horizontal-flip augmentation.
- Use label smoothing to regularize the classifier.
- Make each ensemble member reproducible through a fixed seed and saved configuration metadata.

> The repository contains the training pipeline, not the dataset or precomputed benchmark results. Training requires access to the video files and clip metadata described below.

---

## 2. Approach

### Data pipeline

The dataset expects a CSV containing an `id` column, MP4 files named after each video ID, and JSON metadata describing three clips per video. Each clip metadata file provides frame indices and a binary `videomae_label`.

For every sample, the loader reads the selected frames, resizes them to **224 × 224**, and applies ImageNet normalization. Training samples receive temporal jitter. The configuration for seed `343` also enables color jitter and random horizontal flips; the other configured seeds use temporal augmentation only.

### Model and training

The pipeline starts from `MCG-NJU/videomae-large-finetuned-kinetics`, replaces its classification head for two classes, and fine-tunes the model with the seed-specific learning rate, dropout, label smoothing, weight decay, warmup ratio, and epoch count.

Three ensemble configurations are currently available:

| Seed | Augmentation | Learning rate | Epochs |
|------|--------------|---------------|--------|
| `42` | Temporal | `9.03e-06` | 3 |
| `343` | Temporal, color, and horizontal flip | `1.86e-05` | 3 |
| `643` | Temporal | `9.03e-06` | 3 |

The metrics module also includes ROC AUC, average precision, expected calibration error, and temperature fitting helpers. The current training command does **not** run evaluation or post-hoc temperature calibration.

---

## 3. Setup and Usage

### Requirements

- Python 3.10 or newer
- PyTorch and a compatible torchvision installation
- A CUDA-capable GPU is recommended for practical training times
- Dataset CSV, MP4 videos, and clip JSON metadata (not included)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the PyTorch/torchvision pair appropriate for your hardware by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

### Train an ensemble member

```bash
python train_ensemble.py 42 \
  --train-csv /path/to/train.csv \
  --video-dir /path/to/train/videos \
  --clips-dir /path/to/train/clip_metadata \
  --output-dir ./runs/seed-42
```

Train the other configured members by using seed `343` or `643`. If `--output-dir` is omitted, the output directory defaults to `./ensemble_temp_seed_all_<seed>`.

The first run downloads the pretrained VideoMAE checkpoint from Hugging Face. Keep the dataset private or appropriately licensed; no dataset files are required to be committed to a public GitHub repository.

---

## 4. Project Structure and Outputs

```text
dashcam-video-ensemble/
├── dashcam_ensemble/
│   ├── __init__.py
│   ├── config.py       # Seed-specific hyperparameters
│   ├── data.py         # Video dataset and augmentations
│   ├── metrics.py      # Evaluation and calibration helpers
│   ├── model.py        # VideoMAE classifier
│   └── training.py     # Single-member training pipeline
├── train_ensemble.py  # Main command-line entry point
├── train_ensemble_temp.py  # Legacy script-name alias
├── train_ensemble_legacy.py  # Alias for the copied legacy script name
├── requirements.txt
├── .gitignore
└── README.md
```

Each run writes the fine-tuned checkpoint to `<output-dir>/final_model/` and a `run_metadata.json` file containing the seed, model index, training-data description, and hyperparameters. Hugging Face Trainer may also create intermediate checkpoints and training logs in the output directory.
