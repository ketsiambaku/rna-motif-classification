# Baseline Code Analysis Report
**Phase 1.1 Deliverable**  
**Date:** November 27, 2025  
**Author:** RNA Motif Classification Project

## Executive Summary

Analyzed the baseline U-Net classifier implementation from `example/TrainingForClassification/`. The model achieves **100% validation accuracy from epoch 1**, indicating a fundamental data leakage issue rather than genuine classification performance. The root cause is that **PDB feature statistics (phosphate-phosphate distance matrices) are highly discriminative** due to different motif sizes, allowing trivial classification without learning meaningful structural patterns.

## Baseline Architecture

### Model: UNetClassifier
**Total Parameters:** 135,667 (trainable)

#### Architecture Components

1. **Encoder Path**
   ```
   Conv3d(1→32, k=3, pad=1) + ReLU + MaxPool3d(2)
   Conv3d(32→64, k=3, pad=1) + ReLU + MaxPool3d(2)
   ```
   - Input: [B, 1, 64, 64, 64] density volumes
   - Output: [B, 64, 16, 16, 16] encoded features
   - Downsampling: 4x reduction in spatial dimensions

2. **Decoder Path**
   ```
   ConvTranspose3d(64→32, k=2, stride=2) + ReLU
   ConvTranspose3d(32→16, k=2, stride=2) + ReLU
   ```
   - Upsampling back to original resolution
   - Output: [B, 16, 64, 64, 64]

3. **Classifier Head**
   ```
   AdaptiveAvgPool3d(1) → Flatten
   Linear(16 + 900 → 64) + ReLU + Dropout(0.2)
   Linear(64 → 3)
   ```
   - Combines volumetric features (16-dim) with PDB features (900-dim)
   - 3-way classification: bulge, hairpin, internal

#### Code Issues Found

**Original `unet_classifier.py` has a bug:**
```python
# Line 23: References undefined self.pool
x = self.pool(x)  # ❌ self.pool not defined in __init__
```

**Fixed in `unet_classifier_fixed.py`:**
- Added `self.pool = nn.AdaptiveAvgPool3d(1)` to constructor
- Moved pooling operation before classifier
- Added Dropout(0.2) for regularization

## Dataset Analysis

### Data Distribution
- **Training:** 2,751 samples (1,940 internal, 445 hairpin, 366 bulge)
- **Validation:** 688 samples (485 internal, 112 hairpin, 91 bulge)
- **Class Imbalance:** 70.5% internal, 16.2% hairpin, 13.3% bulge
- **No Data Leakage:** Verified zero overlap between train/val splits

### Data Processing Pipeline

1. **MRC Density Maps** (`rna_dataset.py:extract_density`)
   - Load .mrc file with `mrcfile.open()`
   - Min-max normalization: `(data - min) / (max - min)`
   - Resize to [64, 64, 64] via trilinear interpolation
   - Output: Single-channel volume [1, 64, 64, 64]

2. **PDB Features** (`rna_dataset.py:extract_pdb_features`)
   - Extract phosphate (P) atom coordinates from PDB structure
   - Fixed limit: 30 atoms maximum (`max_atoms=30`)
   - Compute pairwise distance matrix: `dist[i,j] = ||coords[i] - coords[j]||`
   - Flatten to 900-dimensional vector (30×30)
   - **Critical Issue:** Zero-padding for structures with <30 atoms

### Feature Statistics by Class

| Class    | Mean Distance | Std Dev | Max Distance | Non-zero % | Interpretation |
|----------|--------------|---------|--------------|------------|----------------|
| Bulge    | 0.23 Å       | 1.65    | 20.05 Å      | **2.22%**  | Very few atoms (~1-2) |
| Hairpin  | 0.56 Å       | 2.75    | 21.47 Å      | **4.67%**  | Few atoms (~4-6) |
| Internal | 3.13 Å       | 6.64    | 28.16 Å      | **20.22%** | Many atoms (~13-20) |

**Key Finding:** The non-zero percentage directly correlates with motif size, creating a trivial classification signal.

## Training Results

### Hyperparameters
- **Optimizer:** Adam (lr=1e-3)
- **Loss:** CrossEntropyLoss
- **Batch Size:** 4
- **Epochs:** 10
- **Device:** CPU (no GPU available)
- **Regularization:** Dropout(0.2) in classifier

### Performance Metrics

| Epoch | Train Loss | Val Loss | Val Accuracy | Time/Epoch |
|-------|-----------|----------|--------------|------------|
| 1     | 0.0104    | 0.0000   | **100.00%**  | 457s       |
| 2     | 0.0064    | 0.0000   | **100.00%**  | 487s       |
| 3     | 0.0032    | 0.0000   | **100.00%**  | 456s       |
| 4     | 0.0001    | 0.0000   | **100.00%**  | 481s       |

**Observations:**
- Perfect accuracy achieved immediately (epoch 1)
- Training loss decreases rapidly (0.0104 → 0.0001)
- Validation loss rounds to 0.0000
- Average batch processing: ~0.66s/batch (688 batches)
- Training extremely slow on CPU (~8 minutes/epoch)

## Critical Issues Identified

### 1. Feature Leakage via Motif Size
**Root Cause:** The 30×30 phosphate distance matrix encodes motif size through sparsity:
- Small motifs (bulge) → mostly zeros (2% non-zero)
- Large motifs (internal) → dense matrix (20% non-zero)

**Why This is a Problem:**
- Model learns to classify based on **feature sparsity**, not structural patterns
- Does not generalize to motifs normalized for size
- Biologically invalid: size alone doesn't determine motif type
- Defeats the purpose of learning from cryo-EM density maps

### 2. Class Imbalance
- Internal loops: 70.5% of dataset
- Hairpin: 16.2%
- Bulge: 13.3%
- **Impact:** Model could achieve ~70% accuracy by always predicting "internal"
- Current 100% accuracy masks potential bias toward majority class

### 3. Architectural Concerns
- **U-Net decoder unused:** Global pooling discards spatial reconstruction
- **Feature fusion imbalance:** 900 PDB features + 16 volume features (98% PDB)
- **Shallow network:** Only 2 encoder/decoder blocks may underutilize volumetric data
- **No data augmentation:** No rotations, flips, or noise injection

### 4. Training Infrastructure
- **CPU-only training:** 8 min/epoch is prohibitively slow for experimentation
- **No early stopping:** Runs all 10 epochs despite convergence at epoch 1
- **Limited metrics:** No per-class accuracy, precision, recall, or F1-scores
- **No visualization:** No confusion matrix, loss curves, or feature maps

## Recommendations for Improvement

### Immediate Actions (Phase 1-2)

1. **Fix Feature Extraction**
   - Normalize PDB features by motif size (divide distances by sqrt(num_atoms))
   - Or use fixed-size sampling with padding indicators
   - Extract additional features: torsion angles, base-pairing patterns

2. **Balance Dataset**
   - Apply class weights to loss function: `weight = [1/n_samples_per_class]`
   - Or use stratified sampling with class balancing

3. **Enhance Evaluation**
   - Compute per-class metrics (precision, recall, F1)
   - Generate confusion matrix
   - Visualize feature importance (which features contribute most?)

4. **Enable GPU Training**
   - Train on CUDA device to reduce epoch time from 8min → ~30s

5. **Add Sequence Features**
   - Extract RNA sequence from PDB files (A, C, G, U)
   - Compute physicochemical properties (GC content, purine/pyrimidine ratio) - 10 features
   - Compute dinucleotide composition (AA, AC, AG, AU, ..., UU) - 16 features
   - Total: 26 additional sequence features
   - Recommended: Normalized hybrid (900 P-P distances + 26 sequence = 926 features)

### Phase 2-3 Improvements

6. **Implement 3D CNN Baseline**
   - Compare with U-Net performance
   - Simpler architecture to validate findings

7. **Multi-Channel U-Net**
   - Create separate channels for ribose/phosphate/base masks
   - Reduce reliance on explicit PDB features

8. **Data Augmentation**
   - Random rotations (90°, 180°, 270°)
   - Random flips (x, y, z axes)
   - Gaussian noise injection (σ=0.01)

9. **Curriculum Learning**
   - Phase 1: Train on size-normalized data
   - Phase 2: Introduce size variability gradually

## Biological Interpretation

### Why Size Differs by Motif Type

- **Bulge Loops:** Unpaired nucleotides on ONE strand (1-5 nt typical)
  - Example: 5'-CGCGAACGCG-3' with bulge A (1 nucleotide)
  - Dataset avg: 5 nucleotides
  
- **Hairpin Loops:** Unpaired nucleotides closing back on same strand (4-10 nt)
  - Example: 5'-CGCG[UUCG]CGCG-3' with 4 nt loop
  - Dataset avg: 7 nucleotides
  
- **Internal Loops:** Unpaired on BOTH strands (2-30 nt combined)
  - Example: 5'-CG[AAA]CG-3' paired with 3'-GC[UU]GC-5' (5 nt total)
  - Dataset avg: 14 nucleotides

**Implication:** Size is partially correlated with motif type but NOT definitively:
- Large hairpins (8-10 nt) exist
- Small internal loops (2+2 nt) exist
- Model should learn 3D structure, not just count atoms

### Sequence Composition Patterns (New Finding)

Analysis of sequence features reveals additional discriminative patterns:

| Motif Type | Avg Length | GC Content | Purine Content | Dinuc Entropy |
|------------|-----------|------------|----------------|---------------|
| Bulge      | 5 nt      | 52%        | 48%            | 1.25          |
| Hairpin    | 7 nt      | 49%        | 37%            | 1.59          |
| Internal   | 14 nt     | 70%        | 59%            | 2.03          |

**Key observations:**
- Internal loops have significantly higher GC content (70% vs ~50%)
- Internal loops are more purine-rich (59% vs 37-48%)
- Sequence complexity (entropy) increases with motif size
- These features could complement size-normalized structural features

## Conclusion

The baseline implementation demonstrates a **false positive result** - achieving 100% accuracy through inadvertent feature leakage rather than learning biologically meaningful patterns. The primary issue is that phosphate distance matrices encode motif size, which is highly correlated with (but not equivalent to) motif type.

**Next Steps:**
1. Implement size-normalized feature extraction
2. Re-run training with corrected features
3. Expect accuracy to drop significantly (likely 60-80% range)
4. Establish true baseline for comparison with Phase 2+ improvements

**Phase 1.1 Status:** ✅ **COMPLETE**
- Identified critical data leakage issue
- Documented architecture and performance
- Provided actionable recommendations

**Estimated Time:** 6 hours (actual: ~7 hours including debugging)

---

## Appendix: Code Files Analyzed

1. **train.py** (original) - 61 lines
2. **train_baseline.py** (fixed) - 183 lines with enhanced logging
3. **unet_classifier.py** (original, buggy) - 28 lines
4. **unet_classifier_fixed.py** - 58 lines with fixes and documentation
5. **rna_dataset.py** - 56 lines
6. **preprocess.py** - 17 lines (unused in training loop)
7. **generate_csv.sh** - 17 lines (shell script for label generation)

**Repository:** [github.com/ketsiambaku/rna-motif-classification](https://github.com/ketsiambaku/rna-motif-classification)
