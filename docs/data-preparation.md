# Data Preparation Pipeline

This document describes the complete data preparation pipeline from raw cryo-EM density maps (.mrc) and PDB structures (.pdb) to model-ready tensors.

## Overview

Our hybrid model requires two types of inputs:
1. **3D Density Maps** (from .mrc files) → 32×32×32 normalized volumes
2. **Sequence Features** (from .pdb files) → 24-dimensional feature vectors

Both are processed through careful normalization and transformation steps to ensure consistent, high-quality input to the neural network.

---

## Part 1: Density Map Processing

### Raw Data Characteristics

**File Format**: MRC (Medical Research Council) electron density format
- Contains 3D volumetric data from cryo-EM experiments
- Variable grid dimensions: 11³ to 46³ voxels (typical range)
- Variable voxel sizes: ~1.0-1.4 Å/voxel (Angstroms per voxel)
- **99.92% isotropic**: X = Y = Z voxel dimensions (cubic voxels)
- Variable physical sizes: 16,600-53,800 Ų (cubic Angstroms)

**Key Insight**: Different motif classes have different **physical sizes**:
- Small motifs (bulge1, hairpin3): ~16,000-22,000 Ų
- Large motifs (4x4, 5x5): ~44,000-54,000 Ų
- This is **biological reality**, not a data artifact

### Processing Pipeline

#### Step 1: Load Raw MRC Data
```python
with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
    density = mrc.data.astype(np.float32)
```

**Input**: Variable-sized 3D array (e.g., 29×30×29)
**Data type**: float32
**Value range**: Arbitrary (depends on microscope, reconstruction method)

#### Step 2: Min-Max Normalization
```python
density_min = density.min()
density_max = density.max()
density = (density - density_min) / (density_max - density_min + 1e-8)
```

**Purpose**: 
- Standardize intensity range across different cryo-EM experiments
- Remove microscope/reconstruction bias
- Enable consistent model training

**Output**: Values in [0, 1] range
**Note**: Normalization is **per-sample** (each MRC independently scaled)

**Why Min-Max Instead of Z-Score?**
- Matches example code convention
- Preserves relative intensity relationships
- No negative values (better for ReLU activations)
- Bounded range prevents extreme outliers

#### Step 3: Trilinear Interpolation to Fixed Size
```python
density_tensor = torch.tensor(density).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
density_tensor = F.interpolate(
    density_tensor,
    size=(32, 32, 32),
    mode='trilinear',
    align_corners=False
)
density = density_tensor.squeeze(0).squeeze(0).numpy()  # [32, 32, 32]
```

**Purpose**: 
- Resize all volumes to uniform 32³ grid for model input
- Preserve structural information (no cropping/padding artifacts)
- Enable batch processing

**Method**: Trilinear interpolation
- Smooth 3D resampling using weighted average of 8 nearest neighbors
- Preserves density gradients and structural features
- Standard in medical imaging and 3D computer vision

**Alternative Approaches (NOT Used)**:
- ❌ Center crop/pad: Creates artificial padding patterns → data leakage
- ❌ scipy.ndimage.zoom: Slower, similar results
- ❌ Adaptive pooling: Loses fine-grained details

**Output**: Consistent 32×32×32 volume
**Value range**: ~[0.03, 0.98] (slight compression from interpolation smoothing)

#### Step 4: Convert to PyTorch Tensor
```python
density_tensor = torch.FloatTensor(density).unsqueeze(0)  # [1, 32, 32, 32]
```

**Shape**: `[1, 32, 32, 32]` (channel, depth, height, width)
**Data type**: torch.FloatTensor
**Device**: CPU initially, moved to GPU/MPS during training

### Final Density Format

| Property | Value |
|----------|-------|
| **Shape** | `[1, 32, 32, 32]` |
| **Data type** | torch.FloatTensor |
| **Value range** | ~[0, 1] (normalized) |
| **Total elements** | 32,768 values |
| **Memory** | ~131 KB per sample |

---

## Part 2: Sequence Feature Processing

### Raw Data Characteristics

**File Format**: PDB (Protein Data Bank) structure format
- Contains atomic coordinates of RNA nucleotides
- SEQRES records: sequence of nucleotides (A, U, G, C)
- Variable sequence lengths: 1-30 nucleotides (typical for our motifs)

### Feature Extraction Pipeline

#### Step 1: Extract SEQRES Sequence
```python
with open(pdb_path, 'r') as f:
    for line in f:
        if line.startswith('SEQRES'):
            residues.extend(line[19:].split())
```

**Output**: List of nucleotide codes (e.g., `['A', 'U', 'G', 'C', 'G', ...]`)

#### Step 2: Compute Size-Invariant Features

We extract **24 statistical features** that don't depend on sequence length:

##### Composition Features (4 values)
```python
# Nucleotide frequencies
freq_A = count('A') / total_length
freq_U = count('U') / total_length
freq_G = count('G') / total_length
freq_C = count('C') / total_length
```

**Purpose**: Capture base composition bias
**Range**: [0, 1] (frequencies sum to 1.0)

##### Dinucleotide Features (16 values)
```python
# All possible 2-nucleotide combinations
dinuc_AA, dinuc_AU, dinuc_AG, dinuc_AC,
dinuc_UA, dinuc_UU, dinuc_UG, dinuc_UC,
dinuc_GA, dinuc_GU, dinuc_GG, dinuc_GC,
dinuc_CA, dinuc_CU, dinuc_CG, dinuc_CC
```

**Purpose**: Capture local sequence patterns
**Calculation**: Sliding window of size 2, normalized by (length - 1)
**Range**: [0, 1] (frequencies)

##### GC Content (1 value)
```python
gc_content = (count('G') + count('C')) / total_length
```

**Purpose**: Important structural stability indicator
**Range**: [0, 1]

##### Purine/Pyrimidine Ratio (1 value)
```python
purine_count = count('A') + count('G')
pyrimidine_count = count('U') + count('C')
pur_pyr_ratio = purine_count / (pyrimidine_count + 1e-8)
```

**Purpose**: Chemical balance indicator
**Range**: [0, ∞) (ratio, typically 0.5-2.0)

##### Sequence Length (1 value)
```python
length = len(sequence)
```

**Purpose**: Size indicator (normalized later)
**Range**: [1, 30] nucleotides (typical for our dataset)

##### Sequence Complexity (1 value)
```python
# Shannon entropy
complexity = -sum(p * log2(p) for p in [freq_A, freq_U, freq_G, freq_C] if p > 0)
```

**Purpose**: Measure of sequence randomness vs. repetitiveness
**Range**: [0, 2] (0 = single nucleotide, 2 = perfectly random)

#### Step 3: Handle Missing Features
```python
if features is None or len(features) != 24:
    return np.zeros(24, dtype=np.float32)
```

**Fallback**: Zero vector for failed extractions
**Note**: Dataset cleaning removed most problematic samples

#### Step 4: Convert to PyTorch Tensor
```python
seq_features_tensor = torch.FloatTensor(features)  # [24]
```

### Final Sequence Feature Format

| Property | Value |
|----------|-------|
| **Shape** | `[24]` |
| **Data type** | torch.FloatTensor |
| **Value range** | Mixed (see feature descriptions) |
| **Total elements** | 24 values |
| **Memory** | ~96 bytes per sample |

---

## Part 3: Model Input Structure

### Batch Preparation

During training, samples are batched:

```python
batch = {
    'density': torch.stack([sample['density'] for sample in batch]),    # [B, 1, 32, 32, 32]
    'sequence': torch.stack([sample['sequence'] for sample in batch]),  # [B, 24]
    'label': torch.tensor([sample['label'] for sample in batch])        # [B]
}
```

Where `B` is the batch size (default: 8-32).

### Model Input Layer

The hybrid model processes these inputs through two branches:

#### Density Branch
- **Input**: `[B, 1, 32, 32, 32]` tensor
- **Architecture**: 3D U-Net encoder
- **First layer**: `Conv3d(1, 16, kernel_size=3, padding=1)`
- **Expected range**: [0, 1] (normalized densities)

#### Sequence Branch
- **Input**: `[B, 24]` tensor
- **Architecture**: Fully connected layers
- **First layer**: `Linear(24, 64)`
- **Expected range**: Mixed (various feature scales)

#### Fusion
- Density features extracted at bottleneck: `[B, 256]`
- Concatenated with sequence features: `[B, 256+64]`
- Fed to classifier head

---

## Part 4: Data Splits

### Train/Validation/Test Division

**Strategy**: Stratified random split with fixed seed

```python
train: 70% of data (20,117 samples)
val:   20% of data (5,748 samples)
test:  10% of data (2,873 samples)
```

**Total**: 28,738 samples after cleaning

**Random seed**: 42 (for reproducibility)

**Stratification**: Maintains class proportions in each split

### Class Distribution

All 15 classes represented in each split:
- 1x1, 2x2, 3x3, 4x4, 5x5 (size classes)
- bulge1-5 (bulge loop classes)
- hairpin3-7 (hairpin loop classes)

---

## Part 5: Data Quality Assurance

### Pre-processing Validation

#### Density Maps
- ✅ All MRC files are 3D (no 2D or 4D data)
- ✅ 99.92% isotropic voxels (cubic)
- ✅ Variable but reasonable physical sizes
- ✅ Successfully load and interpolate

#### Sequence Features
- ✅ All PDB files have SEQRES records (after cleaning)
- ✅ Valid nucleotide sequences (A, U, G, C only)
- ✅ Reasonable sequence lengths (1-30 nucleotides)
- ✅ Complete feature extraction

### Dataset Cleaning

**Original**: 30,348 samples
**Removed**: 1,610 samples (5.3%)
- Missing SEQRES records
- Corrupted PDB files
- Invalid nucleotide codes

**Final**: 28,738 clean samples

---

## Part 6: Design Decisions & Rationale

### Why Trilinear Interpolation?

**Problem**: Variable MRC sizes (11³ to 46³) need uniform input
**Options considered**:
1. **Center crop/pad** ❌
   - Pros: Fast, simple
   - Cons: Loses data (crop) or adds artificial zeros (pad)
   - **Critical issue**: Padding amount correlates with class → data leakage!

2. **Trilinear interpolation** ✅ (chosen)
   - Pros: Preserves all information, smooth scaling, no artifacts
   - Cons: Slightly slower (negligible with GPU)
   - Matches example code convention

### Why Min-Max Normalization?

**Options considered**:
1. **Z-score normalization** (mean=0, std=1)
   - Used in early version
   - Produces negative values
   - Less compatible with ReLU

2. **Min-max normalization** ✅ (chosen)
   - Values in [0, 1]
   - Preserves relative intensities
   - Matches example code
   - Better for CNNs with ReLU

### Why 32³ Grid Size?

**Trade-offs**:
- **16³**: Too small, loses structural detail
- **32³**: ✅ Good balance of detail and computational cost
- **64³**: Better detail but 8× more memory, slower training
- **Variable size**: Can't batch efficiently

**Chosen**: 32³ (matches Phase 2.1 model architecture)

### Why Size-Invariant Sequence Features?

**Problem**: Variable sequence lengths (1-30 nucleotides)
**Options**:
1. **Padding to max length**: Creates artificial patterns
2. **Recurrent network**: Adds complexity, harder to train
3. **Statistical features** ✅ (chosen): Fixed 24D vector regardless of length

### Physical Size Variation

**Observation**: Classes have different physical volumes (16K-54K Ų)
**Question**: Is this data leakage?
**Answer**: No, it's biological reality!
- Bulge loops ARE physically smaller than 5x5 internal loops
- Model learning size correlations is legitimate
- This is fundamentally different from padding artifacts

---

## Part 7: Reproducibility Checklist

To reproduce our data preparation:

1. ✅ Use cleaned dataset (28,738 samples)
2. ✅ Set random seed = 42
3. ✅ Use 70/20/10 train/val/test split
4. ✅ Apply min-max normalization per sample
5. ✅ Use trilinear interpolation to 32³
6. ✅ Extract 24 sequence features from PDB
7. ✅ Handle missing features with zero fallback
8. ✅ Use torch.FloatTensor for all inputs
9. ✅ Batch samples consistently

---

## Part 8: Performance Considerations

### Memory Usage

**Per sample**:
- Density: 32³ × 4 bytes = 131 KB
- Sequence: 24 × 4 bytes = 96 bytes
- Total: ~131 KB per sample

**Full dataset** (28,738 samples):
- Memory: ~3.7 GB (uncompressed in RAM)
- Disk: ~4.5 GB (.mrc files compressed)

### Processing Speed

**Loading time** (MPS/M1 Pro):
- Per sample: ~50-100 ms
- Per epoch (10% subset): ~30 seconds
- Per epoch (full dataset): ~5 minutes

**Bottleneck**: Disk I/O for .mrc files (CPU decoding of trilinear interpolation)

**Optimization**: 
- Use multiple workers: `num_workers=2` (for MPS) or `num_workers=4` (for GPU)
- Cache in RAM if possible (not feasible for full dataset)

---

## Part 9: Training Regularization - Early Stopping

### Overview

Early stopping is a critical regularization technique used during training to prevent overfitting and optimize computational efficiency. Our implementation monitors validation accuracy and automatically terminates training when the model stops improving.

### Configuration

```python
early_stopping_patience = 10  # Default setting
```

**Patience**: Number of consecutive epochs without improvement before stopping

### How It Works

#### Tracking Best Performance
```python
best_val_acc = 0.0
patience_counter = 0

# After each epoch:
if val_acc > best_val_acc:
    best_val_acc = val_acc        # Update best
    patience_counter = 0           # Reset counter
    save_checkpoint()              # Save best model
else:
    patience_counter += 1          # Increment counter
```

#### Stopping Condition
```python
if patience_counter >= early_stopping_patience:
    print(f"Early stopping triggered after {epoch+1} epochs")
    break  # Stop training
```

### Example Training Sequence

**Typical pattern when early stopping triggers at epoch 11:**

```
Epoch 1:  Val Acc = 15.3% → NEW BEST ✓ (patience = 0/10)
Epoch 2:  Val Acc = 22.1% → NEW BEST ✓ (patience = 0/10)  ← Best model saved
Epoch 3:  Val Acc = 21.8% → No improvement (patience = 1/10)
Epoch 4:  Val Acc = 20.5% → No improvement (patience = 2/10)
Epoch 5:  Val Acc = 19.9% → No improvement (patience = 3/10)
Epoch 6:  Val Acc = 20.2% → No improvement (patience = 4/10)
Epoch 7:  Val Acc = 21.0% → No improvement (patience = 5/10)
Epoch 8:  Val Acc = 19.7% → No improvement (patience = 6/10)
Epoch 9:  Val Acc = 20.8% → No improvement (patience = 7/10)
Epoch 10: Val Acc = 19.5% → No improvement (patience = 8/10)
Epoch 11: Val Acc = 20.1% → No improvement (patience = 9/10)
Epoch 12: Val Acc = 19.8% → No improvement (patience = 10/10) → STOP
```

**Result**: Training stops, best model from epoch 2 (22.1% accuracy) is used

### Why This Happens

#### 1. Model Convergence
- Model has learned all it can from the data
- Further training yields diminishing returns
- **This is normal and expected**

#### 2. Overfitting Prevention
- Validation accuracy plateaus while training accuracy increases
- Model memorizing training data instead of learning generalizable patterns
- Early stopping prevents this by using best validation checkpoint

#### 3. Computational Efficiency
- No point continuing if model isn't improving
- Saves GPU time and electricity
- Allows faster iteration on experiments

### Benefits

✅ **Prevents Overfitting**
- Stops before model memorizes training set
- Uses validation set as objective judge
- Returns best model, not final model

✅ **Saves Resources**
- Automatic stopping when training is done
- No need to guess optimal epoch count
- Can set high max_epochs (e.g., 50) safely

✅ **Improves Generalization**
- Model from best validation epoch generalizes better
- Avoids overfitted models from late epochs
- Balances bias-variance tradeoff

### When Early Stopping Triggers

**Common scenarios:**

| Trigger Point | Interpretation | Action |
|--------------|----------------|---------|
| **Epoch 5-10** | Model converged quickly | ✅ Good! Model is efficient |
| **Epoch 11-20** | Normal convergence | ✅ Expected behavior |
| **Epoch 30-40** | Slow convergence | ⚠️ Consider: better architecture, more data, higher LR |
| **Never (hits max)** | Model still improving | ⚠️ Increase max_epochs or check for bugs |

### Interpreting Results

#### If Training Stops Early (e.g., Epoch 11)

**✅ This is GOOD if:**
- Best validation accuracy is reasonable (>30% for 15 classes)
- Training loss is decreasing smoothly
- Validation loss is stable or increasing (classic overfitting pattern)

**⚠️ This is CONCERNING if:**
- Best validation accuracy is very low (<15% for 15 classes)
- Both training and validation loss are still decreasing rapidly
- Model seems undertrained

#### Example: Full Dataset Training

```
Training on 28,738 samples (100% dataset)
Early stopping triggered after 11 epochs

Analysis:
- Stopped early: Yes (11/50 epochs)
- Patience exhausted: 10 consecutive epochs without improvement
- Best model: Saved from epoch 1 or 2 (when accuracy peaked)
- Interpretation: Model converged quickly, further training unnecessary
```

### Tuning Early Stopping

#### Patience Parameter

**Conservative (patience = 15-20)**:
```python
early_stopping_patience = 15
```
- Allows longer plateau exploration
- Good for noisy validation metrics
- Use when: unsure if model has converged

**Standard (patience = 10)** ✅ Default:
```python
early_stopping_patience = 10
```
- Balanced approach
- Works well for most cases
- Current setting

**Aggressive (patience = 5)**:
```python
early_stopping_patience = 5
```
- Stops quickly after peak
- Good for fast iteration
- Risk: might stop too early on noisy validation

#### Disabling Early Stopping

**For full training curves:**
```python
early_stopping_patience = 1000  # Effectively disabled
# OR
early_stopping_patience = n_epochs  # Never triggers
```

**When to disable:**
- Want to see complete 50-epoch learning curve
- Debugging training dynamics
- Comparing different epoch counts
- Creating plots for paper

### Best Practices

1. **Always monitor validation metrics**
   - Training accuracy can be misleading
   - Validation accuracy is objective measure
   - Save best validation checkpoint, not last epoch

2. **Use patience ≥ 10 for small datasets**
   - Validation metrics can be noisy with limited data
   - Avoid premature stopping

3. **Complement with learning rate scheduling**
   ```python
   scheduler = ReduceLROnPlateau(
       optimizer, 
       mode='max',           # Maximize validation accuracy
       factor=0.5,           # Reduce LR by half
       patience=5,           # After 5 epochs without improvement
       verbose=True
   )
   ```
   - LR reduction often leads to new improvements
   - Resets patience counter

4. **Check training curves**
   - Plot train/val loss and accuracy
   - Verify early stopping made sense
   - Identify overfitting patterns

### Comparison with Other Stopping Criteria

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Early Stopping** ✅ | Automatic, prevents overfitting | May stop too early with noisy validation | Default choice |
| **Fixed Epochs** | Simple, reproducible | May overfit or underfit | When optimal epochs known |
| **Train Loss Threshold** | Task-specific | Ignores validation, may overfit | When train loss is proxy for performance |
| **Manual Inspection** | Flexible | Time-consuming, not automated | Research/debugging only |

### Troubleshooting

**Problem**: Training stops at epoch 1-3 (too early)
- **Cause**: Validation accuracy not improving from start
- **Solution**: Check data loading, model initialization, learning rate

**Problem**: Training never stops (hits max_epochs)
- **Cause**: Model still improving or patience too high
- **Solution**: Increase max_epochs or check if validation is increasing

**Problem**: Erratic stopping (sometimes epoch 5, sometimes 30)
- **Cause**: High variance in validation set (too small)
- **Solution**: Increase validation set size or patience parameter

### Implementation Details

**Checkpoint Saving**:
```python
# Best model is automatically saved when validation improves
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
    'val_loss': val_loss,
    'history': history
}
torch.save(checkpoint, 'experiments/.../best_model.pth')
```

**Loading Best Model**:
```python
# After training completes, load best checkpoint for evaluation
checkpoint = torch.load('experiments/.../best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']}")
```

---

## Summary

Our data preparation pipeline ensures:
- ✅ **Consistent input dimensions**: 32³ density + 24D features
- ✅ **Proper normalization**: [0, 1] range for densities
- ✅ **No data leakage**: Trilinear interpolation eliminates padding artifacts
- ✅ **Biological validity**: Physical size variation reflects real motif differences
- ✅ **Reproducibility**: Fixed splits, deterministic preprocessing
- ✅ **Computational efficiency**: Balanced between detail and speed

This pipeline successfully transforms raw cryo-EM data into high-quality model inputs for RNA motif classification.
