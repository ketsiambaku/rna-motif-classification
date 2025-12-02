# Training Guide - Phase 2.2 (3-Branch Model)

## Quick Reference

### Model Architecture
- **Phase 2.2**: 3-branch hybrid model
- **Branches**: 
  1. Density encoder (3D U-Net) → 256 features
  2. Sequence encoder (MLP) → 256 features (24 size-invariant sequence features)
  3. Pairing encoder (MLP) → 64 features (10 size-invariant base pairing features)
- **Fusion**: 256 + 256 + 64 = 576 features
- **Classifier**: 576 → 256 → 128 → 6 classes
- **Parameters**: 3.75M (~14.32 MB)

### Dataset Splits
- **Total**: 28,738 samples (dataset2, 6-class consolidated)
- **Train**: 20,116 samples (70%)
- **Val**: 5,747 samples (20%)
- **Test**: 2,875 samples (10%)

### Training Configuration
```python
n_epochs: 100             # Maximum epochs
early_stopping: 7         # Stop if no improvement for 7 epochs
batch_size: 32            # Optimal for T4/M1 GPU
learning_rate: 0.001      # Adam learning rate
weight_decay: 1e-4        # L2 regularization
lr_scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
```

---

## Local Training (M1 Mac)

### Full Dataset Training
```bash
python src/train_hybrid.py \
  --consolidate \
  --use-subset 1.0 \
  --batch-size 32 \
  --device mps \
  --num-workers 0
```

**Expected time**: 8-12 hours on M1 Pro/Max

### Quick Test (5%)
```bash
python src/train_hybrid.py \
  --consolidate \
  --use-subset 0.05 \
  --batch-size 16 \
  --device mps \
  --num-workers 0
```

**Expected time**: 30-45 minutes on M1

---

## Google Colab Training

### Setup Steps

1. **Open notebook**: `colab_training.ipynb`
2. **Enable GPU**: Runtime → Change runtime type → T4 GPU
3. **Run cells in order**:
   - Cell 1: Mount Drive & Check GPU
   - Cell 2: Clone repository (base branch)
   - Cell 3: Extract dataset2.zip from Drive
   - Cell 4: Install dependencies
   - Cell 5: **Main training** (100% dataset)
   - Cell 6: Download results to Drive

### Full Training Command
```python
python src/train_hybrid.py \
  --consolidate \
  --use-subset 1.0 \
  --batch-size 32 \
  --device cuda \
  --num-workers 2
```

**Expected time**: 20-30 minutes on T4 GPU

### Quick Test Command (5%)
```python
python src/train_hybrid.py \
  --consolidate \
  --use-subset 0.05 \
  --batch-size 32 \
  --device cuda \
  --num-workers 2
```

**Expected time**: 3-5 minutes on T4 GPU

---

## Output Files

Training saves to `experiments/hybrid_phase2_1_6class_YYYYMMDD_HHMMSS/`:

### Model Checkpoints
- `best_model.pth` - Best model based on val accuracy
- `checkpoint_epoch_10.pth` - Checkpoint every 10 epochs
- `checkpoint_epoch_20.pth`
- ...

### Predictions & Labels
- `best_val_preds.npy` - Validation predictions (best epoch)
- `best_val_labels.npy` - Validation labels
- `best_val_probs.npy` - Validation class probabilities (for ROC curves)
- `test_preds.npy` - Test set predictions (final)
- `test_labels.npy` - Test set labels
- `test_probs.npy` - Test class probabilities (for ROC curves)

### Metrics & Results
- `config.json` - Training configuration
- `training_history.json` - Loss/accuracy per epoch
- `final_val_metrics.json` - Validation metrics (best model)
- `final_test_metrics.json` - **Test metrics (final unbiased estimate)**

### Generated Visualizations (via `analyze_results.py`)
- `training_curves.png` - Loss, accuracy, and LR over epochs
- `confusion_matrices.png` - Heatmaps for val & test sets
- `roc_curves.png` - **ROC curves with AUC for each class**
- `per_class_comparison.png` - Val vs test performance bars

---

## Interpreting Results

### Validation Metrics
- Used for **model selection** (early stopping)
- Reported from best epoch (highest val accuracy)
- May slightly overestimate due to early stopping on val set

### Test Metrics
- **Final unbiased estimate** for the paper
- Evaluated on held-out 10% never seen during training
- This is the accuracy to report in Results section

### Per-Class Metrics
Both metrics files contain:
```json
{
  "precision": [...],        // Per-class precision (6 values)
  "recall": [...],           // Per-class recall (6 values)
  "f1": [...],              // Per-class F1 scores
  "support": [...],         // Samples per class
  "macro_precision": 0.XX,  // Average precision
  "macro_recall": 0.XX,     // Average recall
  "macro_f1": 0.XX,         // Average F1 (main metric)
  "confusion_matrix": [[...]]  // 6x6 confusion matrix
}
```

### Class Names (6-class consolidation)
0. `small_internal` - 1x1, 2x2 loops
1. `large_internal` - 3x3, 4x4, 5x5 loops
2. `small_bulge` - bulge1, bulge2
3. `large_bulge` - bulge3, bulge4, bulge5
4. `small_hairpin` - hairpin3, hairpin4, hairpin5
5. `large_hairpin` - hairpin6, hairpin7

---

## Monitoring Training

### Real-time Progress
Training prints:
```
Epoch 1/100 [Train]: 100%|████| 629/629 [02:15<00:00, loss=2.34, acc=42.5%]
Epoch 1/100 [Val]:   100%|████| 180/180 [00:30<00:00, loss=1.89, acc=51.2%]

Epoch 1/100 Summary:
  Train Loss: 2.3456 | Train Acc: 42.50%
  Val Loss:   1.8934 | Val Acc:   51.20%
  LR: 0.001000
  ✓ New best model saved! Val Acc: 51.20%
```

### Early Stopping
```
Epoch 45/100 Summary:
  Train Loss: 0.3421 | Train Acc: 89.23%
  Val Loss:   0.8765 | Val Acc:   82.34%
  LR: 0.000125
  No improvement (7/7)

⚠ Early stopping triggered after 45 epochs
```

### Final Evaluation
```
================================================================================
Final Evaluation on Validation Set (Best Model)
================================================================================

Per-Class Metrics:
Class              Precision     Recall         F1    Support
------------------------------------------------------------
small_internal        0.8956     0.8234     0.8580      1426
large_internal        0.7823     0.7905     0.7864       506
small_bulge           0.8645     0.8934     0.8787      1831
large_bulge           0.7234     0.6890     0.7058       476
small_hairpin         0.8123     0.8456     0.8286       940
large_hairpin         0.7456     0.7623     0.7539       568
------------------------------------------------------------
Macro Avg            0.7873     0.7840     0.7852

================================================================================
Final Evaluation on Test Set (Best Model)
================================================================================
[Similar output for test set]

✓ Training complete! Results saved to: experiments/hybrid_phase2_1_6class_...
✓ Best validation accuracy: 82.34%
✓ Final test accuracy: 81.56%
```

---

## Troubleshooting

### Out of Memory (OOM)
- **Reduce batch size**: `--batch-size 16` or `--batch-size 8`
- **Reduce workers**: `--num-workers 0`

### MPS Backend Issues (Mac)
- **Fallback to CPU**: `--device cpu`
- Update PyTorch: `pip install --upgrade torch torchvision`

### Slow Training
- **Check device**: Should show "Device: mps" or "Device: cuda"
- **Increase workers** (if not OOM): `--num-workers 4`

### BioPython Import Error
```bash
pip install biopython
```

### Missing dataset2
- Download from Google Drive
- Extract to project root: `dataset2/`
- Verify structure: `dataset2/1x1/`, `dataset2/bulge1/`, etc.

---

## Next Steps After Training

### 1. Analyze Results
```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load metrics
with open('experiments/hybrid_phase2_1_6class_XXX/final_test_metrics.json') as f:
    metrics = json.load(f)

# Plot confusion matrix
cm = np.array(metrics['confusion_matrix'])
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Test Set Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

### 2. Update Paper
Document in `docs/paper.md` Results section:
- Test accuracy: XX.XX%
- Per-class performance
- Confusion matrix analysis
- Compare with Phase 2.1 (if available)

### 3. Phase 2.3 Decision
If test accuracy < 72%:
- Consider adding torsion angles (210 more features)
- Phase 2.3: 4-branch model (density + seq + pairing + torsion)

If test accuracy ≥ 72%:
- ✅ Phase 2 complete!
- Proceed to Phase 3: Visualization & Analysis
- Write paper Results & Discussion sections

---

## Performance Targets

| Metric | Phase 1 | Phase 2.1 Target | Phase 2.2 Target | Paper Goal |
|--------|---------|------------------|------------------|------------|
| Task | 3-class | 15-class | 6-class | 6-class |
| Accuracy | 58% | TBD | 70-75% | 72%+ |
| Features | 0 | 24 (seq) | 34 (seq+pair) | 34-244 |
| Test Acc | N/A | N/A | **Report this** | **≥72%** |

**Note**: Phase 2.2 uses 6-class consolidation for better performance and interpretability.
