# Baseline Analysis - Dataset2 (Fine-Grained Classification)
**Phase 1.1 Extended Analysis**  
**Date:** November 27-28, 2025  
**Dataset:** dataset2/ - Fine-grained motif size classification

## ⚠️ CRITICAL UPDATE: DATA LEAKAGE DETECTED (Nov 28, 2025)

**RESULTS SUMMARY:**
- **PDB-only accuracy**: 98.8% → Proves leakage via size-dependent sparsity
- **Density-only (14-class)**: 21.99% → Fine-grained task too hard
- **Density-only (3-class)**: **58.51%** → ✅ **Model learns topology successfully!**

Analysis reveals **severe data leakage** identical to Dataset1. The PDB phosphate distance matrix encodes motif size through sparsity patterns (31.78% range), enabling trivial classification without learning structural features.

**Key Finding**: When limited to density maps only, the model achieves 58.51% accuracy on 3-class topology classification (bulge/hairpin/internal), proving it CAN learn structural patterns. The 76.8 percentage point drop (98.8% → 21.99%) on 14-class task proves PDB features were causing leakage.

See detailed analysis: [`dataset2-leakage-analysis.md`](./dataset2-leakage-analysis.md)

**Status**: ✅ Leakage confirmed and quantified | ✅ Realistic baseline established (58.51%)  
**Recommendation**: Use 3-class coarse task for Phase 2 (scientifically valid)

---

## Executive Summary

Dataset2 provides **14 fine-grained motif classes** based on exact loop sizes, compared to the 3 coarse-grained classes in dataset1. This creates a significantly more challenging classification problem with **76:1 class imbalance** (largest: bulge2 with 8,616 samples, smallest: 4x4 with 113 samples). The fine-grained classification tests whether models can distinguish motifs of the same type but different sizes.

**~~Key Challenge:~~ Will the model learn structural differences between similar-sized motifs, or simply memorize size patterns?**

**UPDATE:** Model exploits size patterns via PDB feature sparsity (leakage confirmed).

## Dataset2 Structure

### Overall Statistics
- **Total Samples:** 24,195 (vs 3,439 in dataset1)
- **Number of Classes:** 14 (vs 3 in dataset1)
- **Files per Sample:** 2 (.mrc density map + .pdb structure)
- **Classification Granularity:** Exact loop size within motif type

### Class Distribution

| Class     | Type          | Samples | Percentage | Description |
|-----------|---------------|---------|------------|-------------|
| bulge2    | Bulge         | 8,616   | 35.61%     | 2 unpaired nt |
| hairpin5  | Hairpin       | 3,514   | 14.52%     | 5 nt loop |
| hairpin6  | Hairpin       | 2,627   | 10.86%     | 6 nt loop |
| 2x2       | Internal      | 2,503   | 10.35%     | 2×2 asymmetric |
| 3x3       | Internal      | 2,425   | 10.02%     | 3×3 asymmetric |
| bulge3    | Bulge         | 1,624   | 6.71%      | 3 unpaired nt |
| bulge4    | Bulge         | 694     | 2.87%      | 4 unpaired nt |
| hairpin4  | Hairpin       | 513     | 2.12%      | 4 nt loop |
| hairpin3  | Hairpin       | 557     | 2.30%      | 3 nt loop |
| bulge1    | Bulge         | 457     | 1.89%      | 1 unpaired nt |
| 5x5       | Internal      | 244     | 1.01%      | 5×5 asymmetric |
| hairpin7  | Hairpin       | 178     | 0.74%      | 7 nt loop |
| bulge5    | Bulge         | 130     | 0.54%      | 5 unpaired nt |
| 4x4       | Internal      | 113     | 0.47%      | 4×4 asymmetric |

### Distribution by Motif Type

| Motif Type    | Classes | Total Samples | Percentage | Avg per Class | Std Dev |
|---------------|---------|---------------|------------|---------------|---------|
| Bulge         | 5       | 11,521        | 47.6%      | 2,304         | 3,572   |
| Hairpin       | 5       | 7,389         | 30.5%      | 1,478         | 1,495   |
| Internal Loop | 4       | 5,285         | 21.8%      | 1,321         | 1,321   |

## Class Imbalance Analysis

### Severity
- **Imbalance Ratio:** 76.2:1 (bulge2 vs 4x4)
- **Largest Class:** bulge2 (8,616 samples, 35.6% of dataset)
- **Smallest Class:** 4x4 (113 samples, 0.5% of dataset)
- **Classes <1% of data:** 3 classes (4x4, bulge5, hairpin7)
- **Classes >10% of data:** 4 classes (bulge2, hairpin5, hairpin6, 2x2)

### Impact on Training
1. **Majority Class Bias:** Model may predict "bulge2" for everything → 35.6% accuracy
2. **Rare Class Underrepresentation:** 4x4, bulge5, hairpin7 have <300 samples each
3. **Within-Type Imbalance:** 
   - Bulge: 1,624× variation (bulge2: 8,616 vs bulge5: 130)
   - Hairpin: 19.7× variation (hairpin5: 3,514 vs hairpin7: 178)
   - Internal: 22.2× variation (2x2: 2,503 vs 4x4: 113)

### Mitigation Strategies
1. **Class Weights:** Weight loss inversely proportional to class frequency
   ```python
   weights = total_samples / (n_classes * samples_per_class)
   # bulge2: 0.14, 4x4: 10.67
   ```

2. **Stratified Sampling:** Ensure all classes represented in each batch
   
3. **Oversampling:** Duplicate rare class samples (SMOTE for tabular features)
   
4. **Focal Loss:** Focus on hard-to-classify examples
   ```python
   FL(p_t) = -α(1-p_t)^γ log(p_t)
   ```

## Comparison with Dataset1

| Metric                    | Dataset1      | Dataset2      | Change    |
|---------------------------|---------------|---------------|-----------|
| Total Samples             | 3,439         | 24,195        | +7.0×     |
| Number of Classes         | 3             | 14            | +4.7×     |
| Classification Task       | Coarse-grained| Fine-grained  | Harder    |
| Class Imbalance (max:min) | 5.3:1         | 76.2:1        | 14.4× worse |
| Avg Samples per Class     | 1,146         | 1,728         | +51%      |

### Key Differences

**Dataset1 (Coarse):**
- Classify motif *type* only (bulge vs hairpin vs internal)
- Size variation within each class
- Easier problem: structural differences between types are large
- Example: All bulges (1-5 nt) grouped together

**Dataset2 (Fine):**
- Classify motif type *and* exact size (bulge2 vs bulge3 vs bulge4...)
- Within-type discrimination required
- Harder problem: structural differences between adjacent sizes are subtle
- Example: Must distinguish bulge2 (2 nt) from bulge3 (3 nt)

## Expected Challenges

### 1. Size Leakage (Still Present)
The same fundamental issue from dataset1 persists:
- PDB phosphate distance matrix encodes size through sparsity
- **Prediction:** Model will achieve high accuracy by counting non-zero distances
- **But:** Now it must learn precise size thresholds (2 vs 3 vs 4 nt)

### 2. Subtle Structural Differences
Within-type discrimination is harder:
- **Easy:** Distinguish hairpin5 (5 nt) from bulge2 (2 nt) - different types
- **Hard:** Distinguish hairpin5 (5 nt) from hairpin6 (6 nt) - same type, adjacent sizes
- **Critical Question:** Are structural differences between hairpin5 and hairpin6 detectable in cryo-EM density?

### 3. Rare Class Learning
Classes with <300 samples may not learn well:
- 4x4: 113 samples (0.5%)
- bulge5: 130 samples (0.5%)
- hairpin7: 178 samples (0.7%)

### 4. Within-Type Confusion
Expected confusion matrix patterns:
- **Low confusion:** bulge ↔ hairpin ↔ internal (different types)
- **High confusion:** bulge2 ↔ bulge3 ↔ bulge4 (adjacent sizes)
- **Highest confusion:** 2x2 ↔ 3x3 ↔ 4x4 (internal loops are most variable)

## Prediction: Expected Baseline Performance

Based on dataset1 analysis (100% accuracy due to size leakage):

### Optimistic Scenario (Size Leakage Works Perfectly)
- **Overall Accuracy:** 85-95%
- **Per-Type Accuracy:**
  - Bulge classification: 90-95% (clear size patterns)
  - Hairpin classification: 85-90% (moderate variation)
  - Internal classification: 70-80% (highest within-type variability)

### Realistic Scenario (Size Leakage + Noise)
- **Overall Accuracy:** 60-75%
- **Well-performing classes:** bulge2, hairpin5, hairpin6, 2x2, 3x3 (>2000 samples)
- **Poor-performing classes:** 4x4, bulge5, hairpin7 (<200 samples)
- **Confusion:** Adjacent size classes (bulge2↔bulge3, hairpin5↔hairpin6)

### Pessimistic Scenario (Size Discrimination Fails)
- **Overall Accuracy:** 35-45%
- **Model defaults to majority class:** Predicts "bulge2" frequently
- **Rare classes never learned:** 4x4, bulge5, hairpin7 accuracy ≈0%

## Recommended Experiments

### Experiment 1: Baseline with Existing Features
- Use same architecture as dataset1 (900 P-P distance features)
- Train with CrossEntropyLoss
- Establish performance ceiling due to size leakage

### Experiment 2: Class-Weighted Training
- Apply class weights inversely proportional to frequency
- Compare per-class performance with Experiment 1
- Goal: Improve rare class accuracy

### Experiment 3: Size-Normalized Features
- Normalize P-P distances by sqrt(num_atoms)
- Remove size signal, force structural learning
- Expected: Large accuracy drop (60% → 30-40%)

### Experiment 4: Hierarchical Classification
- **Stage 1:** Classify motif type (bulge/hairpin/internal) - 3-way
- **Stage 2:** Classify size within type - 4-7 way per type
- Hypothesis: Easier to learn hierarchy than flat 14-way classification

### Experiment 5: Transfer Learning
- **Pre-train:** On dataset1 (3-way coarse classification)
- **Fine-tune:** On dataset2 (14-way fine classification)
- Hypothesis: Coarse features help fine-grained discrimination

## Evaluation Metrics

Given severe class imbalance, accuracy alone is insufficient:

### Primary Metrics
1. **Macro-Averaged F1:** Treats all classes equally (unweighted mean of per-class F1)
2. **Balanced Accuracy:** Average of per-class recall
3. **Per-Class Precision/Recall/F1:** Identify which classes fail

### Secondary Metrics
4. **Confusion Matrix:** Visualize within-type confusion patterns
5. **Top-2 Accuracy:** Is correct class in top 2 predictions?
6. **Micro-Averaged F1:** Overall accuracy (biased toward majority classes)

### Analysis Questions
- Do adjacent size classes confuse more than distant sizes?
- Is confusion higher within types or across types?
- Do rare classes (<300 samples) achieve >50% recall?
- Does class weighting improve rare class performance?

## Data Splits for Dataset2

### Recommended Split Strategy
```python
# Stratified 80/10/10 split ensuring all classes represented
train: 19,356 samples (80%)
val:    2,420 samples (10%)
test:   2,419 samples (10%)

# Minimum samples per class in test set:
# 4x4: 11 samples (10% of 113)
# bulge5: 13 samples (10% of 130)
# hairpin7: 18 samples (10% of 178)
```

**Critical:** Use stratified splitting to ensure rare classes in all splits.

### Alternative: Size-Stratified Split
For testing generalization to unseen sizes:
- **Train:** bulge1, bulge3, bulge5 + hairpin3, hairpin5, hairpin7 + 2x2, 4x4
- **Test:** bulge2, bulge4 + hairpin4, hairpin6 + 3x3, 5x5
- **Goal:** Test if model interpolates between training sizes

## Next Steps (Phase 1.2+)

1. **Generate CSV labels for dataset2**
   ```bash
   python generate_labels.py ../../dataset2 rna_labels_dataset2.csv
   ```

2. **Create stratified train/val/test splits**
   - Ensure all 14 classes represented
   - Maintain class distribution in each split

3. **Run baseline training with class weights**
   - Compare weighted vs unweighted loss
   - Track per-class metrics

4. **Analyze confusion matrix**
   - Identify within-type confusion patterns
   - Quantify adjacent-size confusion

5. **Compare dataset1 vs dataset2 performance**
   - Does fine-grained classification work?
   - What accuracy is lost going from 3→14 classes?

## Biological Interpretation

### Why Size Matters for Function

RNA loop sizes affect:
1. **Flexibility:** Smaller loops more constrained geometrically
2. **Binding Sites:** Specific sizes accommodate different ligands/proteins
3. **Thermodynamic Stability:** Loop entropy scales with size
4. **Tertiary Contacts:** Larger loops can form long-range interactions

**Example - Hairpin Loops:**
- **GNRA tetraloops (4 nt):** Highly stable, common structural motif
- **UNCG tetraloops (4 nt):** Different structure than GNRA despite same size
- **Larger loops (6-7 nt):** More conformational flexibility, harder to predict

### Structural Predictions

**Bulge Loops:**
- 1 nt: Minimal distortion, base may stack or flip out
- 2 nt: More flexibility, can form non-canonical pairs
- 3+ nt: Increased conformational space, may form internal structure

**Hairpin Loops:**
- 3 nt: Sterically constrained, limited conformations
- 4-5 nt: Optimal for stable tetraloop/pentaloop motifs
- 6-7 nt: Greater flexibility, structure depends on sequence

**Internal Loops (NxM):**
- 2×2: Compact, often forms sheared base pairs
- 3×3: Moderate flexibility, can form novel base pairs
- 4×4+: Large cavities, potential protein binding sites

## Conclusion

Dataset2 presents a **significantly harder classification challenge** than dataset1:
- 4.7× more classes (3 → 14)
- 14× worse class imbalance (5:1 → 76:1)
- Requires within-type size discrimination
- Same size leakage vulnerability

**Prediction:** Baseline will achieve 60-75% accuracy with high performance on frequent classes (bulge2, hairpin5) and poor performance on rare classes (4x4, bulge5). Success will require:
1. Class-weighted loss or sampling strategies
2. Careful evaluation using macro-F1, not just accuracy
3. Analysis of within-type confusion patterns
4. Potentially hierarchical classification approach

**Status:** Ready for Phase 1.2 (Dataset Statistics) and baseline training experiments.

---

## Appendix: Class Size Mapping

### Bulge Loops (5 classes, 11,521 samples)
| Class  | Size (nt) | Samples | % of Bulges | % of Total |
|--------|-----------|---------|-------------|------------|
| bulge1 | 1         | 457     | 4.0%        | 1.9%       |
| bulge2 | 2         | 8,616   | 74.8%       | 35.6%      |
| bulge3 | 3         | 1,624   | 14.1%       | 6.7%       |
| bulge4 | 4         | 694     | 6.0%        | 2.9%       |
| bulge5 | 5         | 130     | 1.1%        | 0.5%       |

### Hairpin Loops (5 classes, 7,389 samples)
| Class    | Size (nt) | Samples | % of Hairpins | % of Total |
|----------|-----------|---------|---------------|------------|
| hairpin3 | 3         | 557     | 7.5%          | 2.3%       |
| hairpin4 | 4         | 513     | 6.9%          | 2.1%       |
| hairpin5 | 5         | 3,514   | 47.5%         | 14.5%      |
| hairpin6 | 6         | 2,627   | 35.5%         | 10.9%      |
| hairpin7 | 7         | 178     | 2.4%          | 0.7%       |

### Internal Loops (4 classes, 5,285 samples)
| Class | Size     | Samples | % of Internal | % of Total |
|-------|----------|---------|---------------|------------|
| 2x2   | 2×2 (4)  | 2,503   | 47.4%         | 10.3%      |
| 3x3   | 3×3 (6)  | 2,425   | 45.9%         | 10.0%      |
| 4x4   | 4×4 (8)  | 113     | 2.1%          | 0.5%       |
| 5x5   | 5×5 (10) | 244     | 4.6%          | 1.0%       |

**Note:** Internal loop size = sum of unpaired nucleotides on both strands (e.g., 3×3 = 6 total unpaired nt)
