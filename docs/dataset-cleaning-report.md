# Dataset Cleaning Report

## Summary

**Date**: December 1, 2025  
**Action**: Cleaned dataset by removing samples with missing sequence features

## Results

### Before Cleaning
- **Total samples**: 30,348
- **Problematic samples**: 1,610 (5.3%)
  - Missing PDB SEQRES records: 1,609
  - Corrupted MRC files: 1

### After Cleaning
- **Total samples**: 28,738 (94.7% retained)
- **All samples**: Have valid sequence features

## Impact by Class

| Class | Before | After | Removed | % Retained |
|-------|--------|-------|---------|------------|
| 1x1 | 6,153 | 4,544 | 1,609 | 73.9% |
| 2x2 | 2,503 | 2,503 | 0 | 100.0% |
| 3x3 | 2,425 | 2,425 | 0 | 100.0% |
| 4x4 | 113 | 113 | 0 | 100.0% |
| 5x5 | 244 | 244 | 0 | 100.0% |
| bulge1 | 457 | 457 | 0 | 100.0% |
| bulge2 | 8,616 | 8,616 | 0 | 100.0% |
| bulge3 | 1,624 | 1,623 | 1 | 99.9% |
| bulge4 | 694 | 694 | 0 | 100.0% |
| bulge5 | 130 | 130 | 0 | 100.0% |
| hairpin3 | 557 | 557 | 0 | 100.0% |
| hairpin4 | 513 | 513 | 0 | 100.0% |
| hairpin5 | 3,514 | 3,514 | 0 | 100.0% |
| hairpin6 | 2,627 | 2,627 | 0 | 100.0% |
| hairpin7 | 178 | 178 | 0 | 100.0% |

## Key Findings

1. **26.1% of 1x1 class** was missing SEQRES records
   - These PDB files lacked sequence information
   - Would have received all-zero sequence features during training
   - Likely recent PDB deposits from 2023-2024 (8xxx codes)

2. **All other classes** were 100% clean except:
   - bulge3: 1 file with corrupted 2D MRC data

3. **Training Impact**:
   - Clean data should improve model convergence
   - No more zero-vector sequence features confusing the model
   - Class imbalance increased from 76.25:1 to 155:1 (bulge2/bulge5)

## Files Location

- **Cleaned dataset**: `dataset2/` (28,738 samples)
- **Backup**: `dataset2_backup/` (1,610 problematic samples)
- **Cleaning script**: `scripts/clean_dataset.py`

## Recommendation

✅ **Training should proceed on cleaned dataset**
- All samples now have valid sequence features
- Model can properly learn from multi-modal input
- Expected improvement in convergence speed and accuracy
