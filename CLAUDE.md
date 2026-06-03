# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RNA motif classification from cryo-EM density maps. Classifies 25 RNA motif types using 3D volumetric data (64×64×64 MRC files) with a 3-level hierarchical label system.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Validate data & models (quick)
python scripts/smoke_test.py

# Run test suite
pytest tests/
pytest tests/test_models.py::TestCNN3DBaseline::test_forward_shape -v  # single test

# Data pipeline (run in order when setting up from scratch)
python scripts/build_data_folder.py   # organize raw .mrc files by class
python scripts/build_manifest.py      # generate data/manifest.csv
python scripts/build_splits.py        # assign train/val/test splits
python scripts/check_malformed.py     # detect corrupt MRC files

# Training (local)
python scripts/run_training.py --model cnn_baseline
python scripts/run_training.py --model resnet3d --epochs 150 --lr 5e-4
python scripts/run_training.py --model swin3d --batch_size 8

# Training (GPU lab servers — trains models in parallel on gpulab06/07/08)
bash scripts/deploy.sh 1   # M1 cnn_baseline on gpulab06
bash scripts/deploy.sh 2   # M2 resnet3d on gpulab07
bash scripts/deploy.sh 3   # M3 swin3d on gpulab08

# Evaluation
python scripts/run_evaluation.py --model swin3d --split test
python scripts/run_evaluation.py --model resnet3d --checkpoint checkpoints/resnet3d/epoch_0050.pt --split val
```

## Label Hierarchy

All three levels are predicted simultaneously. The key constraint: **asymmetric loops have no L3 label** (`l3_idx = -1`), and L3 loss is masked for those samples.

| Level | Classes | Description |
|-------|---------|-------------|
| L1 | 3 | hairpin · internal_loop · bulge |
| L2 | 4 | hairpin · symmetric · asymmetric · bulge |
| L3 | 15 | hairpin3-7 (5) · 1×1–5×5 symmetric (5) · bulge1-5 (5) |

`data/manifest.csv` is the master index — every training run reads from it. Columns: `filepath, class, l1, l2, l3, l1_idx, l2_idx, l3_idx, class_idx, split`. Asymmetric samples have `l3_idx = -1` and `class_idx` 15–24.

## Architecture

### Models (`models/`)

Three competing implementations, all taking `[B, 1, 64, 64, 64]` input:

- **M1 `flat_cnn.py`** — 3D CNN, flat 25-class output `[B, 25]`. Loss: weighted CrossEntropy.
- **M2 `resnet3d.py`** — 3D ResNet-18, outputs `{"l1": [B,3], "l2": [B,4], "l3": [B,15]}` with three independent heads.
- **M3 `swin3d.py`** — 3D Swin Transformer, same output dict. Uses soft-gating conditional heads (L2 head receives L1 logits as input; L3 head receives L2 logits). Gradient checkpointing enabled on stages 3–4.

M2 and M3 use `HierarchicalLoss` = `0.25×L1 + 0.25×L2 + 0.50×L3` (weights configured in `TrainConfig` as `alpha/beta/gamma`).

### Core Pipeline (`src/`)

- **`dataset.py`** — `RNAMotifDataset`: lazy-loads MRC files, 95th-percentile normalization, trilinear resize to 64³, inverse-sqrt class weights.
- **`train.py`** — `TrainConfig` dataclass, training loop, `FlatLoss`/`HierarchicalLoss`, early stopping, checkpoint saving (best val macro-F1 → `best.pt`, periodic every 5 epochs → `epoch_NNNN.pt`).
- **`evaluate.py`** — computes accuracy, macro-F1, sensitivity, specificity, AUC at all three levels; saves confusion matrix PNGs.

### Model-Specific Hyperparameters

| Model | batch_size | lr | patience |
|-------|-----------|-----|---------|
| cnn_baseline | 32 | 1e-3 | 15 |
| resnet3d | 32 | 1e-3 | 15 |
| swin3d | 16 | **1e-4** | 20 |

### Outputs

All outputs land in `checkpoints/<model>/`:
- `best.pt` — best checkpoint by val macro-F1
- `log.csv` — per-epoch metrics (train_loss, val_loss, val_l{1,2,3}_acc, val_l3_f1)
- `train.log` — full stdout/stderr
- `eval_<split>.json` — evaluation metrics dict
- `confusion_l{1,2,3}_<split>.png` — confusion matrices

## Distributed Training

`scripts/deploy.sh` pushes the current branch to GitHub, SSHes into the target gpulab server, clones/updates the repo to `/data/ketsia/rna-motif-classification`, installs deps, rsyncs `data/` (incremental), and launches training inside a named tmux session (`train_m1/m2/m3`). Run all three scripts in separate terminals to train models in parallel.
