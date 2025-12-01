# Google Colab Training Guide for RNA Motif Classification

## Setup Steps

### 1. Create New Colab Notebook
Go to: https://colab.research.google.com/

### 2. Enable GPU
- Runtime → Change runtime type → Hardware accelerator → **T4 GPU** (or A100 if available)

### 3. Install Dependencies
```python
!pip install -q mrcfile biopython scikit-learn tqdm
```

### 4. Upload Required Files

**Option A: Upload directly to Colab**
```python
from google.colab import files
import os

# Create directory structure
os.makedirs("src/data", exist_ok=True)
os.makedirs("src/features", exist_ok=True)
os.makedirs("src/models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

# Upload files one by one
print("Upload hybrid_dataset.py")
uploaded = files.upload()  # Upload src/data/hybrid_dataset.py
!mv hybrid_dataset.py src/data/

print("Upload sequence_features.py")
uploaded = files.upload()  # Upload src/features/sequence_features.py
!mv sequence_features.py src/features/

print("Upload hybrid_unet.py")
uploaded = files.upload()  # Upload src/models/hybrid_unet.py
!mv hybrid_unet.py src/models/

print("Upload train_hybrid.py")
uploaded = files.upload()  # Upload train_hybrid.py
```

**Option B: Clone from GitHub (recommended)**
```python
!git clone https://github.com/ketsiambaku/rna-motif-classification.git
%cd rna-motif-classification
!git checkout dev
```

### 5. Upload Dataset

**Option A: Upload compressed dataset**
```python
from google.colab import files
print("Upload dataset2.zip")
uploaded = files.upload()
!unzip -q dataset2.zip
```

**Option B: Mount Google Drive (recommended for large datasets)**
```python
from google.colab import drive
drive.mount('/content/drive')

# If dataset is in Google Drive
!ln -s /content/drive/MyDrive/dataset2 dataset2
```

### 6. Start Training
```python
!python train_hybrid.py \
    --dataset-root dataset2 \
    --batch-size 32 \
    --epochs 50 \
    --use-subset 1.0 \
    --device cuda
```

## Expected Performance

- **MPS (M1/M2 Mac)**: ~2.1 it/s, ~2 minutes/epoch, **~20 hours total**
- **Colab T4 GPU**: ~15-20 it/s, ~15-20 seconds/epoch, **~15-20 minutes total**
- **Colab A100 GPU**: ~50+ it/s, ~5 seconds/epoch, **~5 minutes total**

## Training Configuration Options

```bash
# Full dataset, larger batch size (T4 GPU has 16GB)
!python train_hybrid.py --dataset-root dataset2 --batch-size 32 --use-subset 1.0

# Quick test run (10% data)
!python train_hybrid.py --dataset-root dataset2 --batch-size 16 --use-subset 0.1

# With specific learning rate
!python train_hybrid.py --dataset-root dataset2 --batch-size 32 --lr 0.0005
```

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
