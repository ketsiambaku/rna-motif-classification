# Google Colab Training Guide - 6-Class Consolidation

This guide shows how to train the RNA motif classification model on Google Colab with GPU acceleration using **6-class consolidation** for better performance.

## What's New: 6-Class Consolidation

We consolidated 15 fine-grained classes into 6 superclasses for better accuracy:

| Consolidated Class | Original Classes | Samples | Biological Meaning |
|-------------------|------------------|---------|-------------------|
| `small_internal` | 1x1, 2x2 | 7,047 | Small internal loops |
| `large_internal` | 3x3, 4x4, 5x5 | 2,782 | Large internal loops |
| `small_bulge` | bulge1, bulge2 | 9,073 | Small bulge loops |
| `large_bulge` | bulge3-5 | 2,447 | Large bulge loops |
| `small_hairpin` | hairpin3-5 | 4,584 | Small hairpin loops |
| `large_hairpin` | hairpin6-7 | 2,805 | Large hairpin loops |

**Expected Performance:**
- **6 classes**: 50-65% accuracy (achievable!)
- **15 classes**: 25-35% accuracy (for comparison)

---

## Quick Start

### 1. Open Colab & Enable GPU
- Go to [colab.research.google.com](https://colab.research.google.com)
- Upload `colab_training.ipynb` (provided below)
- **Runtime → Change runtime type → GPU (T4)**

### 2. Run All Cells in Notebook
The notebook will automatically:
- Mount Google Drive
- Clone your GitHub repo (dev branch)
- Extract dataset
- Install dependencies
- Train model with 6-class consolidation

### 3. Monitor Training
- Training takes ~15-20 minutes on T4 GPU
- Best model saved automatically
- Results in `experiments/` folder

---

## Manual Setup Steps

### Step 1: Mount Drive & Check GPU
```python
from google.colab import drive
import torch

drive.mount('/content/drive')

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print("⚠ No GPU! Change runtime type to GPU")
```

### Step 2: Clone Repository
```python
!git clone https://github.com/ketsiambaku/rna-motif-classification.git
%cd rna-motif-classification
!git checkout dev
```

### Step 3: Extract Dataset
```python
import os
import tarfile

dataset_path = '/content/drive/MyDrive/dataset2.tar.gz'

if os.path.exists(dataset_path):
    print("Extracting dataset...")
    with tarfile.open(dataset_path, 'r:gz') as tar:
        tar.extractall('.')
    print("✓ Dataset extracted")
else:
    print(f"⚠ Dataset not found at: {dataset_path}")
```

### Step 4: Install Dependencies
```python
!pip install -q mrcfile biopython scikit-learn
```

### Step 5: Train with 6-Class Consolidation
```python
import subprocess
import sys

cmd = [
    sys.executable, 'src/train_hybrid.py',
    '--consolidate',
    '--use-subset', '1.0',
    '--batch-size', '32',
    '--device', 'cuda',
    '--num-workers', '2'
]

print("Starting training with 6-class consolidation...")
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end='')
process.wait()
```

---

## Training Options

### Option 1: 6-Class Consolidation (RECOMMENDED)
```bash
python src/train_hybrid.py --consolidate --use-subset 1.0 --batch-size 32 --device cuda
```
- Expected: 50-65% accuracy
- Time: ~15-20 min on T4

### Option 2: 15-Class Original (for comparison)
```bash
python src/train_hybrid.py --use-subset 1.0 --batch-size 32 --device cuda
```
- Expected: 25-35% accuracy
- Time: ~15-20 min on T4

### Option 3: Quick Test
```bash
python src/train_hybrid.py --consolidate --use-subset 0.1 --batch-size 32
```
- Time: ~2-3 min (10% subset)

---

## Troubleshooting

**No GPU?** Runtime → Change runtime type → GPU (T4)

**Dataset not found?** Upload `dataset2.tar.gz` to Google Drive root

**Out of memory?** Use `--batch-size 16`

**ModuleNotFoundError?** Run: `!pip install -q mrcfile biopython scikit-learn --upgrade`

Good luck with training! 🚀

## Download Results

```python
from google.colab import files

# Download best model
files.download('experiments/hybrid_phase2_1_*/best_model.pth')

# Download training log
files.download('experiments/hybrid_phase2_1_*/training_log.txt')

# Download confusion matrix
files.download('experiments/hybrid_phase2_1_*/confusion_matrix.png')
```

## Monitoring Training

```python
# View training progress
!tail -f experiments/training_output.log

# Check current results
!ls -lh experiments/hybrid_phase2_1_*/

# View final metrics
!cat experiments/hybrid_phase2_1_*/training_log.txt | grep "Best"
```

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
!python train_hybrid.py --batch-size 8

# Use gradient accumulation (effective batch size = 8 * 4 = 32)
# (Need to modify train_hybrid.py to add gradient accumulation)
```

### Slow Data Loading
```python
# Increase num_workers (already set to 2 in train_hybrid.py)
# Colab has limited CPU cores, so 2-4 workers is optimal
```

### Dataset Upload Too Large
```python
# Compress dataset first on your local machine:
# tar -czf dataset2.tar.gz dataset2/

# Then upload and extract in Colab:
!tar -xzf dataset2.tar.gz
```

## Files Required (if uploading manually)

1. **src/data/hybrid_dataset.py** (~450 lines)
2. **src/features/sequence_features.py** (~400 lines)
3. **src/models/hybrid_unet.py** (~350 lines)
4. **train_hybrid.py** (~450 lines)
5. **dataset2/** folder (28,738 .mrc/.pdb pairs, ~2-3 GB compressed)

## Estimated Training Time

| Device | 10% subset | Full dataset |
|--------|------------|--------------|
| M1/M2 Mac (MPS) | 2 hours | 20 hours |
| Colab T4 GPU | 10 mins | 100 mins |
| Colab A100 GPU | 3 mins | 30 mins |

**Recommendation**: Use Colab with full dataset for best results in reasonable time.
