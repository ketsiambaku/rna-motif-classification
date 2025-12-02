# Development Phases - Detailed Implementation Plan

## Overview

This document breaks down the development into actionable phases with specific tasks, acceptance criteria, and technical details. Each phase builds upon the previous, ensuring steady progress toward the final deliverable.

---

## Phase 0: Foundation (Completed)

### Timeline: Week 1-2 ✓
### Status: COMPLETED

#### Tasks Completed
- [x] Repository setup and initialization
- [x] Dataset organization (train/test split by class)
- [x] Environment setup (Python, PyTorch, dependencies)
- [x] Example code review and understanding
- [x] Project documentation structure
- [x] GitHub Copilot instructions
- [x] Codebase Context Specification (CCS)

#### Deliverables
- ✓ Git repository with proper structure
- ✓ Dataset organized in `dataset2/label/` 
- ✓ Documentation: proposal, guidelines, roadmap
- ✓ `.context/` directory with CCS v1.1 compliance

---

## Phase 1: Research & Understanding

### Timeline: Week 3-4 (2 weeks)
### Status: ✅ PHASE 1 COMPLETED
### Prerequisites: Phase 0 completed

### Objectives
1. ✅ Build theoretical foundation for the work
2. ✅ Understand baseline implementation thoroughly
3. ✅ Identify data leakage and establish realistic baselines
4. ✅ Generate comprehensive dataset statistics
5. ✅ Prepare literature review for paper

### Tasks

#### 1.1 Baseline Code Analysis & Data Leakage Detection ✅
**Status**: ✅ COMPLETED (November 28, 2025)  
**Time**: Completed over multiple iterations

**Files Analyzed**:
- ✅ `example/TrainingForClassification/train.py`
- ✅ `example/TrainingForClassification/unet_classifier.py`
- ✅ `example/TrainingForClassification/rna_dataset.py`
- ✅ `example/TrainingForClassification/preprocess.py`

**Actions Completed**:
- ✅ Ran baseline code on dataset1 (3-class: bulge/hairpin/internal)
- ✅ Achieved 100% accuracy → suspected data leakage
- ✅ Extended analysis to dataset2 (14-class: fine-grained size classification)
- ✅ Achieved 100% validation accuracy → confirmed severe leakage
- ✅ **Discovered root cause**: PDB phosphate distance matrix sparsity encodes motif size
  - Small motifs (bulge1): 97.78% sparse
  - Large motifs (5x5): 66% sparse
  - Sparsity range: 31.78% (enables trivial size classification)
- ✅ Trained PDB-only classifier: 98.8% accuracy (proof of leakage)
- ✅ Trained density-only model (14-class): 21.99% accuracy (76.8pp drop proves leakage)
- ✅ Trained density-only model (3-class coarse): **58.51% accuracy** (proves topology learning)
- ✅ Generated confusion matrices showing leakage patterns
- ✅ Documented complete analysis in `docs/dataset2-leakage-analysis.md`

**Key Findings**:
- Dataset1 & Dataset2 both have **severe size leakage** via PDB P-P distance sparsity
- Current 100% accuracy results are **INVALID** for scientific evaluation
- Density-only model achieves 58.51% on 3-class task (proves model CAN learn topology)
- Fine-grained 14-class too hard: only 21.99% accuracy with density alone
- **Recommendation**: Use 3-class coarse task (bulge/hairpin/internal) for Phase 2

**Deliverables**: 
- ✅ `docs/baseline-analysis.md` (dataset1 analysis)
- ✅ `docs/baseline-analysis-dataset2.md` (dataset2 initial analysis)
- ✅ `docs/dataset2-leakage-analysis.md` (comprehensive 500+ line report)
- ✅ Training scripts: `train_density_only.py`, `train_density_3class.py`
- ✅ Analysis scripts: `analyze_dataset2_results.py`
- ✅ Trained models: `best_density_3class_model.pth` (58.51% accuracy)
- ✅ Realistic baseline established: **58.51%** without leakage

---

### 1.1.1 Phase 1 Completion Summary & Phase 2 Plan

**Date**: November 28, 2025  
**Status**: Phase 1.1 ✅ COMPLETED | Phase 2 🚀 READY TO START

#### What Was Accomplished

**Baseline Analysis & Data Leakage Detection**

1. **Discovered Critical Data Leakage** (Dataset1 & Dataset2)
   - PDB phosphate distance matrix (900 features) has size-dependent sparsity
   - Sparsity range: 31.78% (bulge1: 97.78% → 5x5: 66%)
   - This enables trivial size classification without learning structure

2. **Quantified Leakage Impact**
   - PDB-only classifier: **98.8% accuracy** (proof of leakage)
   - Combined model: **100% accuracy** (completely reliant on leakage)
   - Density-only (14-class): **21.99% accuracy** (76.8pp drop proves leakage)
   - Density-only (3-class): **58.51% accuracy** (proves model CAN learn topology)

3. **Established Realistic Baseline**
   - **58.51%** on 3-class coarse task (bulge/hairpin/internal)
   - This is 75% above random baseline (33.3%)
   - Proves density-only model successfully learns structural topology
   - Fine-grained 14-class too hard: only 21.99% with density alone

4. **Documentation & Analysis**
   - Created comprehensive 500+ line leakage analysis report
   - Generated confusion matrices showing leakage patterns
   - Documented training on 10% subset (1,936 samples) for M1 GPU
   - Trained multiple models: PDB-only, density-only (14-class), density-only (3-class)

#### Key Files Created

**Documentation:**
- `docs/baseline-analysis.md` - Dataset1 analysis
- `docs/baseline-analysis-dataset2.md` - Dataset2 initial analysis  
- `docs/dataset2-leakage-analysis.md` - Comprehensive leakage report (500+ lines)
- `docs/sequence-features-analysis.md` - Phase 2 sequence feature specification

**Training Scripts:**
- `example/TrainingForClassification/train_density_only.py` - 14-class density-only
- `example/TrainingForClassification/train_density_3class.py` - 3-class density-only
- `example/TrainingForClassification/analyze_dataset2_results.py` - Leakage detection

**Models:**
- `example/TrainingForClassification/unet_density_only.py` - 3D U-Net (3.5M params)
- `example/TrainingForClassification/dataset_density_only.py` - Density-only dataset loader
- `example/TrainingForClassification/dataset_coarse.py` - Coarse label dataset

**Results:**
- `best_density_3class_model.pth` - Best 3-class model (58.51% accuracy)
- `confusion_matrix_dataset2_pdb_only.png` - PDB-only results visualization
- Training logs: `density_only_training_results.csv`, `density_3class_training_results.csv`

#### Critical Insights

**What We Learned:**
1. Current 100% accuracy is **invalid** - completely due to size leakage
2. PDB features contain valuable information BUT current representation leaks size
3. Density-only model works (58.51%) but needs structural guidance
4. Need **size-invariant PDB features** to combine benefits without leakage
5. 3-class coarse task (topology) is more realistic than 14-class fine-grained

**Scientific Contribution:**
- First to identify and quantify size leakage in RNA motif classification
- Established methodology for detecting feature leakage (sparsity analysis, PDB-only training)
- Demonstrated proper baseline without leakage (58.51%)

---

## Phase 2: Size-Invariant Feature Engineering

### Timeline: Week 5-7
### Status: 🔄 PHASE 2.1 IN PROGRESS (Started Dec 1, 2025)
### Priority: CRITICAL
### Prerequisites: Phase 1 completed

### Overall Goal
Achieve 70-75% accuracy using density + size-invariant features, improving on 58.51% baseline.

### Strategy Overview
Three-step approach with conditional progression:
1. **Phase 2.1** (3-4 days): Density + Sequence (24 features) → Target: 65-70%
2. **Phase 2.2** (3-4 days): Add Base Pairing (~10 features) → Target: 70-75% ← **SUFFICIENT FOR PAPER**
3. **Phase 2.3** (OPTIONAL): Add Torsion Angles (210 features) → Only if Phase 2.2 < 72%

---

### Phase 2.1: Density + Sequence Features 🔄 IN PROGRESS

**Status**: STARTED December 1, 2025 | Training in progress
**Timeline**: 3-4 days  
**Expected Result**: Establish 15-class baseline + improve with sequence features

#### What We're Building

**Goal**: Hybrid model combining 3D density with 24 size-invariant sequence features

**Task**: 15-class classification (1x1, 2x2, 3x3, 4x4, 5x5, bulge1-5, hairpin3-7)

**Architecture**:
```
Input 1: MRC density (32×32×32) → 3D U-Net Encoder → 256 features
Input 2: PDB sequence (24 features) → MLP (24→256) → 256 features
                                    ↓
                        Concatenate [256 + 256] → 512
                                    ↓
                            Classifier → 15 classes
```

#### Progress Tracker

**✅ COMPLETED**:
1. **Sequence Feature Extractor** (`src/features/sequence_features.py`) ✅ DONE
   - ✅ 400+ line implementation
   - ✅ Extracts 24 size-invariant features:
     * 4: A%, U%, G%, C% composition
     * 3: GC content, purine%, pyrimidine%
     * 16: Dinucleotide frequencies (AA, AU, AG, ..., CC)
     * 1: Shannon entropy (normalized)
   - ✅ Parses SEQRES records from PDB
   - ✅ Handles modified nucleotides (PSU, H2U, M2G, 1MA)
   - ✅ Tested on real data (1RY1_114.pdb: 302 nucleotides → 24 features)
   - ✅ Includes size-invariance validation method
   - ✅ **Paper documentation**: Added detailed Methods subsection (5.4.1) with mathematical formulas and biological interpretations
   - ✅ **Status**: Feature extraction pipeline complete and documented

2. **Hybrid Dataset Loader** (`src/data/hybrid_dataset.py`) ✅ DONE
   - ✅ 461+ line implementation (updated)
   - ✅ Loads .mrc density files → (1, 32, 32, 32) normalized tensors
   - ✅ Extracts sequence features from .pdb → (24,) feature arrays
   - ✅ Handles all 15 classes with proper label mapping
   - ✅ **6-class consolidation support** (small/large internal, bulge, hairpin)
   - ✅ Train/val/test splits (70/20/10)
   - ✅ **Class weight calculation bug fixed** (uses self.num_classes instead of hardcoded 15)
   - ✅ Tested on 10% subset:
     * Train: 2,011 samples (6-class consolidated)
     * Val: 574 samples (6-class consolidated)
   - ✅ Sample loading: 18ms/sample
   - ✅ Batch loading: 30ms for batch_size=4

3. **Hybrid U-Net Model** (`src/models/hybrid_unet.py`) ✅ DONE
   - ✅ 350+ line implementation
   - ✅ Dual-branch architecture:
     * Density encoder: 3D U-Net (32³ → 256 features)
     * Sequence encoder: MLP (24 → 256 features)
     * Fusion: Concatenate 512 features
     * Classifier: 512 → 256 → 128 → 6 or 15 classes (configurable)
   - ✅ 3.7M parameters (~14.25 MB)
   - ✅ BatchNorm + Dropout for regularization
   - ✅ Tested with forward pass and feature extraction

4. **Training Script** (`src/train_hybrid.py`) ✅ DONE
   - ✅ 474+ line implementation (updated)
   - ✅ Weighted CrossEntropyLoss (class imbalance handled)
   - ✅ AdamW optimizer with ReduceLROnPlateau scheduler
   - ✅ Early stopping (patience=10)
   - ✅ Checkpoint saving (best model + every 10 epochs)
   - ✅ **CLI args**: --consolidate flag for 6-class mode
   - ✅ **6-class consolidation working** (small/large internal/bulge/hairpin)
   - ✅ Tested locally with 10% subset
   - ✅ **Colab notebook ready** (`colab_training.ipynb`)

5. **Google Colab Integration** ✅ DONE
   - ✅ Created `colab_training.ipynb` in proper JSON format
   - ✅ Extracts dataset from `dataset2.zip` (tar.gz had 50% corruption)
   - ✅ 6-step workflow: mount, clone, extract, install, train, download
   - ✅ Fixed class weights bug (commit 3173ae9 pushed to dev-consolidate)
   - ✅ Ready for full training on T4 GPU (~15-20 min expected)

**🔄 IN PROGRESS**: Training & Evaluation on Google Colab
1. ✅ Setup complete: Colab notebook ready with dataset2.zip extraction
2. ✅ Bug fix: Class weights dimension mismatch resolved (commit 3173ae9)
3. 🔄 **Next**: Re-run quick test (10% subset) on Colab T4 GPU
4. 🔄 **Next**: Full 6-class training (100% dataset, ~28,738 samples)
5. ⏸️ **Pending**: Download results and analyze accuracy/confusion matrix
6. ⏸️ **Pending**: Compare 6-class vs 15-class performance
7. ⏸️ **Pending**: Validate size-invariance (correlation < 0.3)

#### Deliverables for Phase 2.1
- [x] `src/features/sequence_features.py` - Feature extractor (400+ lines) ✅
- [x] `src/data/hybrid_dataset.py` - Dataset loader (461+ lines) ✅
- [x] `src/models/hybrid_unet.py` - Hybrid architecture (350+ lines) ✅
- [x] `src/train_hybrid.py` - Training script (474+ lines) ✅
- [x] `colab_training.ipynb` - Colab notebook for GPU training ✅
- [x] `docs/colab-class-weights-fix.md` - Bug fix documentation ✅
- [x] Paper Methods section 5.4.1 - Sequence feature extraction ✅
- [ ] Training logs and validation report (awaiting Colab results)
- [ ] Final metrics and confusion matrix (awaiting Colab results)
- [ ] Trained model checkpoint (awaiting Colab download)
- [ ] Size-invariance validation report (awaiting results)
- [ ] Performance comparison: baseline → 6-class accuracy

#### Success Criteria
- ✅ Sequence features extract correctly (24 values per sample)
- ✅ Dataset loading works with 6-class consolidation
- ✅ Class weights computed correctly for n_classes (bug fixed)
- ✅ Model architecture supports configurable n_classes (6 or 15)
- ✅ Training script accepts --consolidate flag
- ✅ Colab setup complete with GPU allocation
- 🔄 Model trains successfully on full dataset (in progress)
- ⏸️ All features size-invariant (|correlation| < 0.3 with motif size)
- ⏸️ **6-class accuracy ≥ 50%** (target: 60-70%)
- ⏸️ Model handles class imbalance (weighted loss working)
- ⏸️ Inference time reasonable (<100ms per sample)

---

### Phase 2.2: Add Base Pairing Features ✅ COMPLETED

**Status**: ✅ COMPLETED (December 1, 2025)  
**Timeline**: Completed in parallel with Phase 2.1 training  
**Result**: 3-branch model ready for training

#### What Was Built

**Additional Input**: 10 size-invariant base pairing features

**Features Implemented**:
1. Pairing ratio (% residues in pairs)
2. Pairing density (pairs per nucleotide, normalized)
3. Average pairing distance (Å, normalized by sqrt(n_residues))
4. Pairing distance std dev (normalized)
5. Stem length statistics (mean, max normalized by size)
6. Loop closure indicator (binary)
7. GC pairing percentage (among all pairs)
8. AU pairing percentage (among all pairs)
9. Non-canonical pair percentage
10. Paired vs unpaired ratio

**Detection Method**: N1 (purine) or N3 (pyrimidine) distance < 3.5 Å → paired

**Updated Architecture**:
```
Density → 256 features ┐
Sequence → 256 features├→ Concatenate [256+256+64] → 576 → Classifier (6 classes)
Pairing → 64 features  ┘
```

#### Implementation Completed

**✅ Files Created/Updated**:
1. **`src/features/base_pairing.py`** (550+ lines)
   - BasePairingExtractor class
   - parse_pdb_structure() - BioPython integration
   - get_nucleotide_atoms() - extract N1/N3 coordinates
   - detect_base_pairs() - Watson-Crick and non-canonical detection
   - compute_stem_lengths() - consecutive base pair analysis
   - compute_pairing_features() - 10 size-invariant features
   - validate_size_invariance() - correlation checking
   - Tested on real PDB files (7-9ms extraction time)

2. **`src/data/hybrid_dataset.py`** (updated)
   - Added pairing_extractor initialization
   - Added _extract_pairing_features() method
   - Updated __getitem__ to return pairing tensor (10,)
   - Handles extraction failures gracefully (zeros fallback)

3. **`src/models/hybrid_unet.py`** (updated)
   - Added PairingEncoder class (10→32→64 MLP)
   - Updated HybridUNet to 3-branch architecture
   - Fusion: 256 + 256 + 64 = 576 features
   - Classifier: 576 → 256 → 128 → 6 classes
   - Total parameters: 3,753,606 (~14.32 MB)

4. **`src/train_hybrid.py`** (updated)
   - Updated train_epoch() to extract pairing features
   - Updated validate() to use 3-input forward pass
   - Compatible with existing CLI args and logging

5. **`test_phase2_2.py`** (integration test)
   - End-to-end pipeline test
   - Validates data loading, model forward pass, feature extraction
   - All tests passing ✅

#### Test Results

**Base Pairing Extraction** (tested on real samples):
- **1x1**: 80% pairing ratio, loop closure detected, 60% GC pairs
- **bulge1**: 40% pairing, no loop closure (characteristic), 100% non-canonical
- **hairpin3**: 57% pairing, loop closure present, 100% GC pairs
- **Extraction time**: 2-9ms per sample

**Integration Test** (Phase 2.2):
- ✅ Dataset loads all 3 feature types (density, sequence, pairing)
- ✅ Model processes 3-branch input correctly
- ✅ Output shape: (batch_size, 6) for consolidated classes
- ✅ Feature representations: density(256), sequence(256), pairing(64), fused(576)
- ✅ Model size: 3.75M parameters (~14.32 MB)

#### Success Criteria Met

- ✅ Pairing features extract correctly (10 values per sample)
- ✅ All features use size-invariant formulas (normalized by motif size)
- ✅ 3-branch model architecture functional
- ✅ Training script updated for 3 inputs
- ✅ Integration test passing
- ✅ Ready for training and evaluation

#### Next Steps

- 🔄 Train on Colab with full dataset (awaiting Phase 2.1 results)
- ⏸️ Compare accuracy: Phase 2.1 (seq only) vs Phase 2.2 (seq + pairing)
- ⏸️ Target: 10-15pp improvement with pairing features
- ⏸️ If accuracy < 72%: Consider Phase 2.3 (torsion angles)

---

### Phase 2.3: Add Torsion Angles ⏸️ CONDITIONAL

**Status**: Not planned (only if Phase 2.2 < 72%)  
**Timeline**: 5-7 days  
**Expected Result**: 75-80% accuracy

Complex implementation with 7 backbone angles × 30 residues = 210 features.
Only pursue if base pairing insufficient.

---

### Phase 2 Success Metrics Summary

| Metric | Phase 1 Baseline | Phase 2.1 Target | Phase 2.2 Target | Paper Goal |
|--------|------------------|------------------|------------------|------------|
| **Task** | 3-class (old) | 15-class | 15-class | 15-class |
| **Accuracy** | 58.51% (3-class) | TBD (new baseline) | +10-15pp | 70%+ |
| **Classes** | bulge/hairpin/internal | All 15 classes | All 15 classes | All 15 |
| **Features** | 0 (density) | 24 (seq) | 34 (seq+pair) | Ready |
| **Training Time** | ~12hrs | ~15hrs | ~18hrs | Scale to 100% |
| **Dataset** | 10% (~2.4K old) | 10% (~3K) | 10% (~3K) | Full (30.3K) |

---

#### 1.2 Dataset Statistics ✅ UPDATED
**Status**: ✅ COMPLETED (December 1, 2025)  
**Time**: Re-run with 15 classes

**Actions Completed**:
- ✅ Updated to include new 1x1 class (15 classes total)
- ✅ Re-counted samples: **30,348 total samples** (was 24,195)
- ✅ New class: 1x1 with 6,153 samples (20.27%) - 2nd largest!
- ✅ Class imbalance: 76.25:1 ratio unchanged (bulge2: 28.39% vs bulge5: 0.43%)
- ✅ Regenerated all visualizations

**Script Updated**: `src/utils/dataset_stats.py` (now includes 1x1 class)

**Key Findings**:
- **Total samples**: 30,348 (increased by 6,153)
- **Number of classes**: 15 (was 14)
- **Class distribution**:
  - Size classes (1x1-5x5): 11,438 samples (37.68%)
  - Bulge loops: 11,521 samples (37.96%)
  - Hairpin loops: 7,389 samples (24.35%)
- **Largest classes**: bulge2 (28.39%), 1x1 (20.27%), hairpin5 (11.58%)
- **Severe class imbalance**: Still 76.25:1 ratio requires weighted loss
- **MRC dimensions**: 683 unique dimensions (more variability)
- **PDB structures**: Mean 4,360 ± 2,093 residues, GC content 53.45% ± 8.40%
  - GC content: 53.41% ± 8.58% (balanced purine/pyrimidine composition)
- **Nucleotide composition**: Balanced (A: 24.7%, U: 20.8%, G: 30.7%, C: 23.8%)

**Deliverables**: 
- ✅ `docs/dataset-statistics.md` (comprehensive statistics report)
- ✅ `docs/figures/class_distribution.png` (class and category distributions)
- ✅ `docs/figures/mrc_statistics.png` (MRC dimensions and density analysis)
- ✅ `docs/figures/pdb_statistics.png` (residue counts and sequence composition)

#### 1.3 Literature Review ✅
**Status**: ✅ CORE COMPLETED (November 28, 2025)  
**Time**: 8-10 hours

**Reading List** (minimum 5-7 papers):
- ✅ RNA secondary structure prediction methods
- ✅ Deep learning for cryo-EM density map analysis
- ✅ 3D U-Net applications in biomedical imaging
- ✅ Feature extraction from molecular structures
- ⏸️ Additional RNA loop classification papers (optional if needed)

**Papers Reviewed**:
1. ✅ **Sato et al. (2021)** - RNA secondary structure prediction with deep learning + thermodynamics
   - Nature Communications, 474 citations
   - **Key insight**: Combining data-driven learning with domain knowledge prevents overfitting
   - **Relevance**: Validates our multi-modal approach (density + structural features)

2. ✅ **Si et al. (2020)** - Deep learning for cryo-EM backbone structure prediction
   - Scientific Reports, 125 citations
   - **Key insight**: 3D CNNs successfully learn structural patterns from density maps
   - **Relevance**: Validates our density-only model approach (58.51% baseline)

3. ✅ **Çiçek et al. (2016)** - 3D U-Net: learning dense volumetric segmentation
   - MICCAI, 10,000+ citations (foundational paper)
   - **Key insight**: Skip connections + encoder-decoder crucial for volumetric data
   - **Relevance**: Direct architectural basis for our model, weighted loss for imbalance

4. ✅ **Li et al. (2016)** - Deep CNNs for detecting secondary structures in cryo-EM
   - IEEE BIBM, 75 citations
   - **Key insight**: First application of deep learning to cryo-EM structure detection
   - **Relevance**: Pioneering work validating CNNs on cryo-EM for classification

5. ✅ **Zhang et al. (2022)** - CR-I-TASSER: protein structure assembly with deep CNN
   - Nature Methods, 67 citations
   - **Key insight**: Hybrid CNN + physics-based modeling outperforms either alone
   - **Relevance**: Validates our multi-modal fusion philosophy

6. ✅ **Matsumoto et al. (2021)** - Extraction of protein dynamics from cryo-EM using DL
   - Nature Machine Intelligence, 93 citations
   - **Key insight**: 3D-CNN can extract hidden information beyond structure
   - **Relevance**: Shows density contains topological information we're classifying

7. ✅ **Mu et al. (2021)** - Segmentation tool for secondary structures with 3D CNN
   - Frontiers in Bioinformatics, 17 citations
   - **Key insight**: Most similar problem - classifying structural motifs from density
   - **Relevance**: Validates feasibility, but no multi-modal fusion or leakage detection

**Actions Completed**:
- ✅ Created `docs/related-work.md` with structured template
- ✅ Searched and identified 7+ highly relevant papers
- ✅ Summarized all papers with problem/method/findings/relevance
- ✅ Extracted 7 BibTeX citations for final paper
- ✅ Identified research gaps and our novel contributions

**Key Takeaways for Implementation**:
1. **Multi-modal fusion** (density + structure) is validated approach (Papers 1, 5)
2. **3D U-Net architecture** is proven for volumetric biomedical data (Paper 3)
3. **Weighted loss** essential for class imbalance (Paper 3: 76:1 in our dataset)
4. **Skip connections** preserve spatial information in encoder-decoder (Paper 3)
5. **Medium resolution** (6-8Å) sufficient for structural classification (Papers 2, 4)
6. **Data augmentation** (rotations, flips, elastic) improves generalization (Paper 3)
7. **Transfer learning** from simulation to experiment viable (Paper 6)

**Research Gaps Identified**:
1. **No prior work on RNA motif classification from cryo-EM** - all papers focus on proteins
2. **Size leakage** not addressed in any structural biology ML papers - our novel contribution
3. **Multi-modal fusion** rare in cryo-EM (mostly density-only or structure-only)
4. **Topology classification** (bulge vs hairpin vs internal) unexplored in literature
5. **Size-invariant feature engineering** framework - methodological contribution

**Our Novel Contributions**:
1. **First RNA motif classification** from cryo-EM density maps
2. **Size leakage detection** methodology (PDB-only training, sparsity analysis)
3. **Size-invariant features** (sequence %, base pairing ratio, normalized torsions)
4. **Multi-modal hybrid architecture** combining density and structural features
5. **Realistic baseline** without leakage (58.51%) for 3-class topology

**Deliverable**: 
- ✅ `docs/related-work.md` (comprehensive literature review with 7 papers)

#### 1.4 Paper Writing - Introduction & Related Work ✅
**Status**: ✅ COMPLETED (November 29-30, 2025)  
**Time**: 6-8 hours

**Actions Completed**:
- ✅ Created IEEE format paper skeleton (`docs/paper.md`)
- ✅ **Section 1.1: Background and Motivation** - Completed with RNA motif context, cryo-EM, DeepTracer, DAIS group
- ✅ **Section 1.2: Problem Statement** - Formal problem definition with inputs/outputs, 4 key challenges, clear objective
- ✅ **Section 1.3: Contributions** - Placeholder for final contributions
- ✅ **Section 1.4: Paper Organization** - Complete section roadmap
- ✅ **Section 2.1: RNA Secondary Structure Prediction** - MXfold2 (thermodynamic + DL integration)
- ✅ **Section 2.2: Deep Learning for Cryo-EM Analysis** - DeepTracer 1.0 & 2.0 (protein/nucleic acid reconstruction)
- ✅ **Section 2.3: Unified Biomolecular Structure Prediction** - AlphaFold 3 (diffusion models, multi-modal)
- ✅ **Section 2.4: RNA-Specific Cryo-EM Reconstruction** - DeepCryoRNA (18 atom types, multi-scale, sequence integration)
- ✅ **Section 2.5: RNA Motif Similarity** - RNAMotifComp (overlapping families, 60-70% accuracy challenge)
- ✅ **Section 2.6: Integrated RNA Folding** - CaCoFold-R3D (probabilistic grammar, 96 motif variants)
- ✅ **Section 2.7: Research Gaps** - 5 comprehensive gaps identified with clear positioning

**Papers Integrated into Section 2**:
1. MXfold2 (Sato et al. 2021) - Section 2.1
2. DeepTracer 1.0 (Pfab et al. 2020) - Section 2.2
3. DeepTracer 2.0 (Pfab et al. 2022) - Section 2.2
4. AlphaFold 3 (Abramson et al. 2024) - Section 2.3
5. DeepCryoRNA (Li & Chen 2025) - Section 2.4
6. RNAMotifComp (Petrov et al. 2013) - Section 2.5
7. CaCoFold-R3D (Karan & Rivas 2025) - Section 2.6

**Key Features of Completed Sections**:
- All citations properly formatted with \cite{} commands
- Quantitative results included (accuracy numbers, RMSD, coverage percentages)
- Clear connections to our work in each subsection
- Research gaps systematically identified (cryo-EM motif classification, size-invariance, multi-modal fusion, overlapping families, local vs global context)
- Problem statement with formal mathematical notation
- 4 key challenges documented (size-topology disentanglement, overlapping similarity, multi-modal fusion, limited data)

**Reference Files Created**:
- ✅ `docs/references/si2020deep.md` - DeepTracer 1.0 details
- ✅ `docs/references/deeptracer2.md` - DeepTracer 2.0 details
- ✅ `docs/references/sato2021rna.md` - MXfold2 details
- ✅ `docs/references/alphafold3.md` - AlphaFold 3 details
- ✅ `docs/references/deepcryorna` - DeepCryoRNA details
- ✅ `docs/references/rnamotifcomp.md` - RNAMotifComp details
- ✅ `docs/references/cacofold.md` - CaCoFold-R3D details

**Paper Status**:
- ✅ Section 1 (Introduction): 1.1-1.4 complete
- ✅ Section 2 (Related Work): 2.1-2.7 complete
- ⏸️ Section 3 (Dataset): Statistics ready, needs narrative
- ⏸️ Section 4 (Data Leakage): Phase 1.1 findings ready, needs writing
- ⏸️ Section 5 (Methodology): Awaiting Phase 2 implementation
- ⏸️ Section 6-9: Awaiting experimental results

**Deliverables**:
- ✅ `docs/paper.md` - 700+ line IEEE format paper with Sections 1-2 complete
- ✅ `docs/references/` - 7 reference summary files
- ✅ All BibTeX citations prepared for final References section

### Acceptance Criteria
- ✅ At least 5 papers read and summarized (completed 7)
- ✅ Baseline code runs successfully
- ✅ Baseline accuracy documented (58.51%)
- ✅ Dataset statistics report completed
- ✅ Literature review ready for paper Introduction
- ✅ Paper sections 1 (Introduction) and 2 (Related Work) written and complete

### Risks & Mitigation
- **Risk**: Related work too similar, reduces novelty
- **Mitigation**: Focus on multi-modal fusion and multi-channel aspects as novel contributions

---

## Phase 2: Feature Engineering (Size-Invariant)

### Timeline: Week 5-7 (2-3 weeks)
### Status: 🔄 PHASE 2.1 IN PROGRESS
### Priority: CRITICAL
### Prerequisites: Phase 1 completed

### Objectives
1. ✅ Implement RNA sequence feature extraction (24 features)
2. 🔄 Create hybrid dataset loader (density + sequence)
3. 🔄 Implement hybrid U-Net architecture
4. ⏸️ Add base pairing features (~10 features) 
5. ⏸️ Optional: Add torsion angles (210 features)

### Phase 2.1: RNA Sequence Features 🔄 IN PROGRESS

**Status**: 🔄 STARTED (December 1, 2025)
**Timeline**: 3-4 days
**Expected Result**: 65-70% accuracy with density + sequence

**Tasks Completed**:
- ✅ Implemented `SequenceFeatureExtractor` class (24 features)
  - ✅ Nucleotide composition (A%, U%, G%, C%)
  - ✅ Chemical composition (GC content, purine%, pyrimidine%)
  - ✅ Dinucleotide frequencies (16 features)
  - ✅ Shannon entropy (normalized)
- ✅ SEQRES record parsing from PDB files
- ✅ Modified nucleotide mapping (PSU, H2U, M2G, 1MA)
- ✅ Tested on sample PDB file (1RY1_114.pdb)
  - Extracted 302-nucleotide sequence
  - Generated 24 features successfully
  - GC content: 61.92%, entropy: 0.977

**Files Created**:
- ✅ `src/features/__init__.py` - Package initialization
- ✅ `src/features/sequence_features.py` - Complete feature extractor (400+ lines)

**Remaining Tasks**:
- [ ] Create hybrid dataset loader (`src/data/hybrid_dataset.py`)
- [ ] Implement hybrid U-Net model (`src/models/hybrid_unet.py`)
- [ ] Create training script for hybrid model
- [ ] Validate size-invariance (correlation with motif size < 0.1)
- [ ] Train on 10% subset (3,034 samples from 30,348 total)
- [ ] Evaluate and compare to 58.51% baseline

**Next Steps**:
1. Create `HybridDataset` class that loads both density and sequence features
2. Implement `HybridUNet` with density encoder + sequence MLP branches
3. Train and validate model

---

### Original Phase 2 Plan (Archive)

**Note**: Original plan focused on P-P distances and torsion angles. 
Updated plan prioritizes size-invariant sequence features first, then base pairing.

#### 2.1 P-P Distance Matrix (ARCHIVED - Has size leakage)
**Time**: 4-5 hours

**Implementation**: `src/features/phosphate_distances.py`

```python
def extract_pp_distances(
    pdb_path: str,
    max_residues: int = 30
) -> np.ndarray:
    """
    Extract phosphate-phosphate distance matrix.
    
    Returns:
        Flattened distance matrix of shape (900,)
    """
```

**Steps**:
1. Parse PDB file using BioPython
2. Extract all phosphate (P) atom coordinates
3. Compute pairwise Euclidean distances
4. Handle missing P atoms (zero padding)
5. Flatten matrix to feature vector

**Testing**:
- [ ] Test on sample PDB with known structure
- [ ] Verify output shape is (900,)
- [ ] Check distance values are reasonable (3-30 Å)
- [ ] Handle edge cases (missing atoms, short sequences)

**Deliverable**: Working P-P distance extractor with tests

#### 2.2 Torsion Angles
**Time**: 8-10 hours (most complex)

**Implementation**: `src/features/torsion_angles.py`

**Angles to Compute** (per residue):
1. **α (alpha)**: P(i-1) - O5' - C5' - C4'
2. **β (beta)**: O5' - C5' - C4' - C3'
3. **γ (gamma)**: C5' - C4' - C3' - O3'
4. **δ (delta)**: C4' - C3' - O3' - P(i+1)
5. **ε (epsilon)**: C3' - O3' - P(i+1) - O5'(i+1)
6. **ζ (zeta)**: O3' - P(i+1) - O5'(i+1) - C5'(i+1)
7. **χ (chi)**: O4' - C1' - N9 - C4 (purines) or O4' - C1' - N1 - C2 (pyrimidines)

**Helper Function**:
```python
def compute_dihedral(
    p1: np.ndarray,
    p2: np.ndarray, 
    p3: np.ndarray,
    p4: np.ndarray
) -> float:
    """
    Compute dihedral angle using standard formula.
    Returns angle in degrees [-180, 180].
    """
```

**Steps**:
1. Implement dihedral angle calculation
2. For each residue, extract required atoms
3. Handle missing atoms (fill with 0.0)
4. Compute all 7 angles
5. Validate angle ranges [-180°, 180°]

**Testing**:
- [ ] Test dihedral calculation on known angles
- [ ] Verify output shape (30 × 7 = 210)
- [ ] Check angle ranges
- [ ] Handle nucleotide type differences (purine vs pyrimidine)

**Deliverable**: Working torsion angle extractor with tests

#### 2.3 Base Pairing Matrix
**Time**: 4-5 hours

**Implementation**: `src/features/base_pairing.py`

```python
def extract_base_pairing(
    pdb_path: str,
    max_residues: int = 30,
    threshold: float = 3.5
) -> np.ndarray:
    """
    Detect base pairing using N1/N3 distance criterion.
    
    Args:
        threshold: Distance threshold in Ångströms (default: 3.5)
        
    Returns:
        Binary matrix of shape (900,) indicating pairs
    """
```

**Watson-Crick Pairing Rules**:
- A-U: N1(A) ... N3(U) distance < threshold
- G-C: N1(G) ... N3(C) distance < threshold

**Steps**:
1. Extract N1 atoms (purines: A, G)
2. Extract N3 atoms (pyrimidines: U, C)
3. Compute pairwise distances
4. Apply threshold to create binary matrix
5. Ensure symmetry (if i-j paired, j-i paired)

**Testing**:
- [ ] Test on structure with known pairs
- [ ] Verify output shape (900,)
- [ ] Check binary values (0 or 1)
- [ ] Test different thresholds

**Deliverable**: Working base pairing detector with tests

#### 2.4 Unified Feature Extraction Pipeline
**Time**: 3-4 hours

**Implementation**: `src/features/feature_extractor.py`

```python
class PDBFeatureExtractor:
    """Unified pipeline for all PDB feature extraction."""
    
    def __init__(self, max_residues: int = 30):
        self.max_residues = max_residues
    
    def extract_all_features(self, pdb_path: str) -> dict:
        """
        Extract all features from PDB file.
        
        Returns:
            dict with keys:
                'pp_distances': (900,)
                'torsion_angles': (210,)
                'base_pairing': (900,)
                'concatenated': (2010,)
        """
```

**Actions**:
- [ ] Combine all extractors
- [ ] Add error handling and logging
- [ ] Validate feature dimensions
- [ ] Check for NaN/Inf values
- [ ] Create feature normalization/standardization

**Deliverable**: Complete feature extraction pipeline

### Acceptance Criteria
- ✅ All three features extract correctly
- ✅ Unit tests pass for each feature
- ✅ Unified pipeline handles edge cases
- ✅ Documentation with examples
- ✅ Features validated on known structures

### Risks & Mitigation
- **Risk**: Torsion angle calculation errors
- **Mitigation**: Validate against reference implementations, use known structures for testing

- **Risk**: Missing atoms in PDB files
- **Mitigation**: Implement robust error handling, padding strategies

---

## Phase 3: Dataset & Data Loading

### Timeline: Week 6-7 (1.5 weeks)
### Priority: HIGH
### Prerequisites: Phase 2 completed

### Objectives
1. Create robust PyTorch Dataset class
2. Implement data augmentation
3. Setup data loaders with proper splitting

### Tasks

#### 3.1 Dataset Class Implementation
**Time**: 5-6 hours

**Implementation**: `src/data/rna_dataset.py`

```python
class RNAMotifDataset(Dataset):
    """
    PyTorch Dataset for RNA motif classification.
    
    Loads .mrc density maps and extracts features from .pdb files.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        target_shape: Tuple[int, int, int] = (64, 64, 64),
        max_residues: int = 30,
        transform: Optional[Callable] = None,
        cache_features: bool = True
    ):
```

**Features**:
- [ ] Load and preprocess MRC files
- [ ] Extract features from PDB files
- [ ] Cache extracted features (save time)
- [ ] Handle train/val/test splits
- [ ] Return (density, features, label) tuples

**Testing**:
- [ ] Test on small subset
- [ ] Verify output shapes
- [ ] Check label encoding
- [ ] Test caching mechanism

**Deliverable**: Robust Dataset class

#### 3.2 Data Augmentation
**Time**: 4-5 hours

**Implementation**: `src/data/augmentation.py`

**Augmentation Techniques**:
1. **Spatial**:
   - Random 90° rotations (on x, y, z axes)
   - Random flips (along each axis)
   - Random crops (if applicable)

2. **Intensity**:
   - Gaussian noise injection (std=0.05)
   - Brightness adjustment
   - Contrast adjustment

3. **Elastic Deformation** (optional, advanced):
   - Small deformations preserving structure

**Implementation**:
```python
class Augment3D:
    """3D data augmentation for density maps."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
```

**Important**: Augmentation applied only to density maps, not PDB features (geometric consistency)

**Deliverable**: Augmentation pipeline with visual verification

#### 3.3 Data Splitting Strategy
**Time**: 2-3 hours

**Implementation**: `src/data/split_data.py`

**Strategy**:
- Training: 70% of available data
- Validation: 20% of available data
- Test: 10% OR use provided test folder

**Actions**:
- [ ] Create stratified split (maintain class balance)
- [ ] Save split indices for reproducibility
- [ ] Create CSV files: `train.csv`, `val.csv`, `test.csv`
- [ ] Document split in README

**File Format** (CSV):
```csv
filename,label,mrc_path,pdb_path
5APN_14347,bulge,dataset/train/bulge/5APN_14347.mrc,dataset/train/bulge/5APN_14347.pdb
```

**Deliverable**: Split strategy implemented and documented

#### 3.4 DataLoader Setup
**Time**: 2 hours

**Implementation**: `src/data/create_dataloaders.py`

```python
def create_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    """
```

**Configuration**:
- Batch size: 4 (GPU memory constraint)
- Shuffle: True for train, False for val/test
- Num workers: 4 (adjust based on CPU)
- Pin memory: True (faster GPU transfer)

**Deliverable**: DataLoader factory function

### Acceptance Criteria
- ✅ Dataset class loads all samples correctly
- ✅ Augmentation works without errors
- ✅ Data splits are reproducible
- ✅ DataLoaders iterate correctly
- ✅ Feature caching speeds up loading

### Risks & Mitigation
- **Risk**: Slow data loading
- **Mitigation**: Feature caching, multi-processing, prefetching

---

## Phase 4: Model Implementation

### Timeline: Week 7-9 (2.5 weeks)
### Priority: CRITICAL
### Prerequisites: Phase 3 completed

### Objectives
1. Implement improved U-Net classifier
2. Implement 3D CNN baseline
3. Implement multi-channel U-Net
4. Ensure consistent interfaces

### Tasks

#### 4.1 Enhanced U-Net Classifier
**Time**: 6-8 hours

**Implementation**: `src/models/unet.py`

**Architecture Improvements** over example:
- Deeper network (more layers)
- Batch normalization
- Skip connections properly implemented
- Better feature fusion strategy

```python
class UNetClassifier(nn.Module):
    """
    3D U-Net for RNA motif classification.
    
    Args:
        in_channels: Input channels (1 for single density map)
        pdb_feat_dim: PDB feature dimension (2010)
        num_classes: Number of output classes (3)
        base_filters: Number of filters in first layer (default: 32)
    """
```

**Encoder**:
- Conv3d → BatchNorm3d → ReLU → Conv3d → BatchNorm3d → ReLU → MaxPool3d
- Repeat 3-4 times with increasing channels

**Decoder**:
- ConvTranspose3d → BatchNorm3d → ReLU
- Concatenate with skip connection
- Conv3d → BatchNorm3d → ReLU
- Repeat to match encoder depth

**Classification Head**:
- Global pooling → Flatten
- Concatenate with PDB features
- FC layers with dropout
- Output layer (3 classes)

**Deliverable**: Complete U-Net implementation

#### 4.2 3D CNN Baseline
**Time**: 4-5 hours

**Implementation**: `src/models/cnn.py`

**Architecture**:
```python
class CNN3DClassifier(nn.Module):
    """
    Simple 3D CNN baseline for comparison.
    
    Simpler than U-Net:
    - No skip connections
    - No decoder path
    - Direct classification from encoded features
    """
```

**Structure**:
- 3-4 convolutional blocks
- Each block: Conv3d → BatchNorm → ReLU → MaxPool3d
- Global pooling
- Concatenate with PDB features
- FC layers → output

**Purpose**: Establishes baseline to demonstrate U-Net advantage

**Deliverable**: Working 3D CNN baseline

#### 4.3 Multi-Channel U-Net
**Time**: 6-8 hours

**Implementation**: `src/models/multi_channel_unet.py`

**Key Difference**: Accepts 4-channel input
- Channel 0: Original density map
- Channel 1: Ribose mask
- Channel 2: Phosphate mask
- Channel 3: Base mask

**First Layer**:
```python
self.conv1 = nn.Conv3d(4, base_filters, 3, padding=1)  # 4 input channels
```

**Rest**: Same as single-channel U-Net

**Note**: Requires Phase 4.4 (mask generation) to be completed first

**Deliverable**: Multi-channel U-Net implementation

#### 4.4 Mask Generation (Fix Bugs)
**Time**: 8-10 hours

**Implementation**: `src/preprocessing/mask_generator.py`

**Bug to Fix** (from example/label/):
- Ribose and sugar masks are identical
- Need proper separation: backbone, ribose, base

**Approach**:
1. Load density map and PDB coordinates
2. For each atom in PDB:
   - Determine atom type (phosphate, ribose, base)
   - Find corresponding voxels in density map
   - Assign to appropriate mask
3. Normalize and threshold masks

```python
class ComponentMaskGenerator:
    """Generate component masks from density maps and PDB structures."""
    
    def generate_masks(
        self,
        mrc_path: str,
        pdb_path: str,
        target_shape: Tuple[int, int, int] = (64, 64, 64)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                'density': Original density
                'ribose_mask': Ribose sugar mask
                'phosphate_mask': Backbone mask
                'base_mask': Nucleotide base mask
        """
```

**Testing**:
- [ ] Visual inspection: overlay masks on density
- [ ] Verify masks capture correct components
- [ ] Check mask alignment with density
- [ ] Test on multiple structures

**Deliverable**: Working mask generator with bug fixes

### Acceptance Criteria
- ✅ All three models implemented and tested
- ✅ Models have consistent input/output interfaces
- ✅ Forward pass works correctly
- ✅ Mask generation produces valid outputs
- ✅ Visual validation of masks completed

### Risks & Mitigation
- **Risk**: Mask generation remains buggy
- **Mitigation**: Have single-channel U-Net as fallback; masks are "nice-to-have"

---

## Phase 5: Training Pipeline

### Timeline: Week 9-10 (1.5 weeks)
### Priority: HIGH
### Prerequisites: Phase 4 completed

### Objectives
1. Implement training loop
2. Implement validation and metrics
3. Setup checkpointing and logging
4. Handle class imbalance

### Tasks

#### 5.1 Training Loop
**Time**: 6-8 hours

**Implementation**: `src/training/trainer.py`

```python
class Trainer:
    """Training pipeline for RNA motif classification."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.Device,
        checkpoint_dir: str = 'checkpoints'
    ):
```

**Features**:
- [ ] Training loop with progress bar
- [ ] Validation after each epoch
- [ ] Learning rate scheduling
- [ ] Gradient clipping
- [ ] Mixed precision training (optional)
- [ ] Early stopping
- [ ] Best model checkpointing

**Deliverable**: Complete training pipeline

#### 5.2 Metrics & Evaluation
**Time**: 4-5 hours

**Implementation**: `src/training/evaluator.py`

**Metrics to Track**:
- Loss (train and validation)
- Accuracy (overall and per-class)
- Precision, Recall, F1-Score (per-class)
- Confusion matrix
- ROC curves (optional during training)

```python
class Evaluator:
    """Evaluation metrics for classification."""
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.Device
    ) -> Dict[str, float]:
        """
        Returns dictionary of metrics.
        """
```

**Deliverable**: Evaluation module with all metrics

#### 5.3 Checkpointing & Logging
**Time**: 3-4 hours

**Implementation**: `src/training/checkpoint.py`

**Checkpoint Format**:
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_acc,
    'train_losses': train_losses,
    'val_accuracies': val_accs,
    'config': config_dict
}
```

**Logging**:
- [ ] TensorBoard integration (optional)
- [ ] CSV logging of metrics
- [ ] Console progress bars
- [ ] Save training curves as plots

**Deliverable**: Checkpointing and logging utilities

#### 5.4 Handle Class Imbalance
**Time**: 2-3 hours

**Strategies**:
1. **Weighted Loss**:
```python
class_weights = compute_class_weights(train_dataset)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

2. **Weighted Sampling**:
```python
sampler = WeightedRandomSampler(sample_weights, num_samples)
train_loader = DataLoader(dataset, sampler=sampler)
```

3. **Oversampling** minority classes

**Action**: Analyze dataset, choose strategy, implement

**Deliverable**: Class balancing implemented

### Acceptance Criteria
- ✅ Training loop runs without errors
- ✅ Validation happens after each epoch
- ✅ Checkpoints save correctly
- ✅ Metrics are logged
- ✅ Early stopping works
- ✅ Class imbalance handled

### Risks & Mitigation
- **Risk**: Training unstable (NaN loss)
- **Mitigation**: Gradient clipping, check data for NaN, lower learning rate

---

## Phase 6: Experimentation & Ablation Studies

### Timeline: Week 11-12 (2 weeks)
### Priority: CRITICAL
### Prerequisites: Phase 5 completed

### Objectives
1. Train all model configurations
2. Perform ablation studies
3. Find best hyperparameters
4. Document all experiments

### Tasks

#### 6.1 Baseline Experiments
**Time**: Varies (mostly compute time)

**Experiments to Run**:

1. **U-Net with only density** (no PDB features)
2. **U-Net with density + P-P distances**
3. **U-Net with density + all PDB features** (full model)
4. **3D CNN with density + all features** (baseline)
5. **Multi-channel U-Net** (if masks ready)

**For Each Experiment**:
- Train for 50-100 epochs
- Use same train/val split
- Save checkpoints
- Log all metrics
- Generate training curves

**Deliverable**: Results for all configurations

#### 6.2 Ablation Studies
**Time**: 8-10 hours (analysis time)

**Studies to Perform**:

1. **Feature Ablation**:
   - Density only
   - Density + P-P distances
   - Density + torsion angles
   - Density + base pairing
   - Density + all features

2. **Architecture Ablation**:
   - U-Net depth (3 vs 4 vs 5 levels)
   - Number of filters (16, 32, 64 base)
   - With/without skip connections

3. **Data Augmentation Ablation**:
   - No augmentation
   - Only rotations
   - Only flips
   - All augmentations

**Analysis**:
- Create comparison tables
- Statistical significance tests
- Identify most important features

**Deliverable**: Ablation study results and analysis

#### 6.3 Hyperparameter Tuning
**Time**: 6-8 hours (mostly compute)

**Parameters to Tune**:
- Learning rate: [1e-4, 1e-3, 1e-2]
- Batch size: [2, 4, 8] (limited by GPU)
- Dropout rate: [0.3, 0.5, 0.7]
- Weight decay: [0, 1e-5, 1e-4]

**Approach**: Grid search or random search

**Deliverable**: Best hyperparameters identified

#### 6.4 Experiment Tracking
**Time**: Ongoing

**Tool Options**:
- Manual: CSV + plots
- TensorBoard
- Weights & Biases (wandb)

**Document**:
- All experiment configurations
- Results (accuracy, loss, metrics)
- Training time
- Model size
- Failure cases

**Create**: `experiments/experiment-log.md`

**Deliverable**: Complete experiment log

### Acceptance Criteria
- ✅ All planned experiments completed
- ✅ Results documented systematically
- ✅ Ablation studies show feature importance
- ✅ Best configuration identified
- ✅ Statistical analysis performed

### Risks & Mitigation
- **Risk**: Limited compute time
- **Mitigation**: Prioritize key experiments, use smaller subset for quick iterations

---

## Phase 7: Comprehensive Evaluation

### Timeline: Week 13 (1 week)
### Priority: HIGH
### Prerequisites: Phase 6 completed

### Objectives
1. Evaluate best models on test set
2. Generate all publication-quality figures
3. Perform error analysis
4. Statistical comparison of models

### Tasks

#### 7.1 Test Set Evaluation
**Time**: 3-4 hours

**Actions**:
- [ ] Load best checkpoint for each model
- [ ] Evaluate on held-out test set
- [ ] Compute all metrics
- [ ] Generate predictions for all test samples
- [ ] Save results to file

**Deliverable**: Test set results for all models

#### 7.2 Generate Figures
**Time**: 8-10 hours

**Figures Needed** (publication quality, 300 DPI):

1. **Training Curves**:
   - Loss curves (train vs val)
   - Accuracy curves (train vs val)
   - One plot per model

2. **Confusion Matrices**:
   - Heat map visualization
   - One per model
   - Include percentages and counts

3. **Per-Class Performance**:
   - Bar charts: precision, recall, F1 per class
   - Grouped by model

4. **ROC Curves**:
   - One-vs-rest for each class
   - AUC scores

5. **Ablation Study Results**:
   - Bar chart or table visualization
   - Feature importance ranking

6. **Architecture Diagrams** (optional):
   - Visual representation of U-Net
   - Show feature fusion

**Implementation**: `src/utils/visualization.py`

**Deliverable**: All figures in high resolution

#### 7.3 Statistical Analysis
**Time**: 4-5 hours

**Tests to Perform**:

1. **McNemar's Test**: Compare model predictions pairwise
2. **T-Test**: Compare accuracy distributions (if using k-fold CV)
3. **Confidence Intervals**: Bootstrap confidence intervals for metrics

```python
from scipy.stats import mcnemar, ttest_rel
from sklearn.utils import resample

# McNemar test for model comparison
# Tests if models make different errors
```

**Deliverable**: Statistical test results and p-values

#### 7.4 Error Analysis
**Time**: 6-8 hours

**Analysis**:
- [ ] Identify misclassified samples
- [ ] Visualize density maps of errors
- [ ] Look for patterns in errors
- [ ] Analyze per-class error rates
- [ ] Examine feature distributions for errors

**Questions to Answer**:
- Which classes are most confused?
- Do errors correlate with certain features?
- Are there outliers in the dataset?
- What biological insights can we gain?

**Deliverable**: Error analysis report with visualizations

### Acceptance Criteria
- ✅ Test results for all models
- ✅ All figures publication-ready
- ✅ Statistical tests performed
- ✅ Error analysis completed
- ✅ Results interpretable and explained

---

## Phase 8: Paper Writing

### Timeline: Week 14-15 (2 weeks)
### Priority: CRITICAL
### Prerequisites: Phase 7 completed

### Objectives
1. Write complete 15-page paper in ACM format
2. Follow instructor guidelines precisely
3. Ensure all figures and tables are polished
4. Proofread thoroughly

### Tasks

#### 8.1 Paper Structure & Drafting
**Time**: 20-25 hours

**Section Breakdown**:

**Abstract (200-250 words)** - 2 hours
- Problem statement
- Method overview
- Key results
- Significance

**1. Introduction (2 pages)** - 6 hours
- Background on RNA structure and cryo-EM
- Motivation for classification
- Related work (cite 10-15 papers)
- Contributions of this work
- Paper organization

**2. Methods (3-4 pages)** - 8 hours
- 2.1 Dataset Description
- 2.2 Feature Engineering (P-P, torsion, base pairing)
- 2.3 Model Architectures (U-Net, CNN, multi-channel)
- 2.4 Training Procedure
- Include flowchart/architecture diagram

**3. Experiments (3-4 pages)** - 6 hours
- 3.1 Experimental Setup
- 3.2 Implementation Details
- 3.3 Baseline Comparisons
- 3.4 Ablation Studies
- 3.5 Hyperparameter Selection

**4. Results (2-3 pages)** - 6 hours
- 4.1 Classification Performance (main results table)
- 4.2 Comparison of Models (statistical tests)
- 4.3 Ablation Study Results
- 4.4 Feature Importance Analysis
- Include all key figures

**5. Discussion (1-2 pages)** - 4 hours
- Interpretation of results
- Biological insights
- Comparison with related work (if exists)
- Limitations
- Why approach works or doesn't work

**6. Conclusion (0.5-1 page)** - 2 hours
- Summary of contributions
- Key findings
- Future work
- Broader impact

**References** - 2 hours
- Use BibTeX
- 15-25 scientific papers
- Proper ACM citation format

**Deliverable**: Complete first draft

#### 8.2 Figure & Table Preparation
**Time**: 6-8 hours

**Requirements**:
- High resolution (300 DPI minimum)
- Clear captions (below figures, above tables)
- Color-blind friendly palettes
- Professional appearance
- Consistent style across all figures

**Tables**:
- Results table (accuracy, precision, recall, F1)
- Ablation study table
- Dataset statistics table
- Hyperparameter table

**Deliverable**: All figures and tables camera-ready

#### 8.3 Writing Quality Checks
**Time**: 6-8 hours

**Checklist** (per instructor advice):
- [ ] Abstract clearly summarizes everything
- [ ] Introduction has clear motivation
- [ ] Related work properly cited (no Wikipedia)
- [ ] Method describes novelty
- [ ] Flowchart/architecture diagram included
- [ ] Pseudocode for key algorithms (optional)
- [ ] Results use clear visualizations
- [ ] Discussion interprets findings
- [ ] Conclusion summarizes contributions
- [ ] References properly formatted
- [ ] All formulas professionally typeset (LaTeX)
- [ ] Grammar and spelling checked
- [ ] Consistent terminology throughout
- [ ] Figures referenced in text
- [ ] No orphaned sections

**Deliverable**: Polished paper draft

#### 8.4 Peer Review & Revision
**Time**: 4-6 hours

**Actions**:
- [ ] Get feedback from colleague/friend
- [ ] Address all comments
- [ ] Read paper aloud (catch errors)
- [ ] Check paper against ACM template
- [ ] Verify all claims match experimental results
- [ ] Final proofread

**Deliverable**: Final paper ready for submission

### Acceptance Criteria
- ✅ Paper is exactly 15 pages (excluding references)
- ✅ Follows ACM format precisely
- ✅ All sections complete and polished
- ✅ Figures and tables publication quality
- ✅ No grammatical errors
- ✅ All claims supported by results
- ✅ References properly formatted

### Risks & Mitigation
- **Risk**: Running out of time for writing
- **Mitigation**: Start early, write Methods section alongside Phase 4-6

---

## Phase 9: Code Documentation & Packaging

### Timeline: Week 15-16 (1.5 weeks)
### Priority: HIGH
### Prerequisites: Paper near completion

### Objectives
1. Clean up and document all code
2. Create comprehensive README
3. Ensure reproducibility
4. Package for submission

### Tasks

#### 9.1 Code Cleanup
**Time**: 6-8 hours

**Actions**:
- [ ] Remove debug code and unused imports
- [ ] Add docstrings to all functions/classes
- [ ] Add type hints
- [ ] Format code consistently (black, isort)
- [ ] Remove hardcoded paths
- [ ] Add configuration file (YAML/JSON)

**Deliverable**: Clean, well-documented codebase

#### 9.2 Comprehensive README
**Time**: 6-8 hours

**README.md Structure**:

```markdown
# RNA Motif Classification

## Overview
Brief description of the project

## Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU)
- 16GB RAM minimum
- GPU with 8GB+ VRAM (recommended)

## Installation
```bash
# Step-by-step installation
```

## Dataset Preparation
1. Download dataset from [link]
2. Organize as follows: ...
3. Run preprocessing: ...

## Training
```bash
# Command to train U-Net
python src/train.py --config configs/unet.yaml

# Command to train CNN
python src/train.py --config configs/cnn.yaml
```

## Evaluation
```bash
# Evaluate on test set
python src/evaluate.py --checkpoint checkpoints/best_unet.pth
```

## Reproducing Paper Results
1. Train all models: ...
2. Generate figures: ...
3. Expected results: Accuracy ~XX%

## Project Structure
Description of folders

## Citation
If you use this code, please cite: ...

## License
MIT (or appropriate)
```

**Deliverable**: Complete README with screenshots

#### 9.3 Reproducibility Verification
**Time**: 4-6 hours

**Actions**:
- [ ] Create fresh conda/virtualenv
- [ ] Follow README from scratch
- [ ] Run training
- [ ] Verify results match paper
- [ ] Document any issues
- [ ] Fix issues

**Deliverable**: Verified reproducible pipeline

#### 9.4 Final Packaging
**Time**: 3-4 hours

**Create .zip with**:
1. Final paper PDF
2. Source code (entire `src/` directory)
3. README.md
4. requirements.txt or environment.yml
5. Configuration files
6. Trained model checkpoints (optional)
7. Dataset download instructions
8. LICENSE file

**File Structure**:
```
rna-motif-classification-submission.zip
├── paper.pdf
├── README.md
├── requirements.txt
├── src/
├── configs/
├── checkpoints/ (or download link)
└── LICENSE
```

**Deliverable**: Final submission package

### Acceptance Criteria
- ✅ Code is clean and well-documented
- ✅ README is comprehensive and accurate
- ✅ Pipeline is reproducible (verified)
- ✅ Submission package is complete
- ✅ No hardcoded paths or credentials

---

## Phase 10: Presentation Preparation

### Timeline: Week 16 (Final week)
### Priority: MEDIUM
### Prerequisites: Paper and code complete

### Objectives
1. Create presentation slides
2. Prepare demo (if applicable)
3. Practice presentation

### Tasks

#### 10.1 Presentation Slides
**Time**: 6-8 hours

**Slides** (10-15 slides for 10-15 min talk):

1. Title slide
2. Motivation & Problem Statement
3. Dataset Overview
4. Method: Feature Engineering
5. Method: Model Architectures
6. Experimental Setup
7. Main Results (accuracy table)
8. Ablation Studies
9. Visualizations (confusion matrix, ROC)
10. Discussion & Insights
11. Conclusion & Future Work
12. Thank You / Questions

**Design**:
- Clean, professional template
- Not too much text per slide
- High-quality figures from paper
- Consistent color scheme

**Deliverable**: Presentation slides

#### 10.2 Demo Preparation (Optional)
**Time**: 3-4 hours

**Demo Options**:
- Live inference on test sample
- Interactive visualization
- Jupyter notebook walkthrough

**Deliverable**: Working demo

#### 10.3 Practice & Timing
**Time**: 3-4 hours

**Actions**:
- [ ] Practice full presentation 3+ times
- [ ] Time yourself (stay within limit)
- [ ] Prepare for common questions
- [ ] Have backup slides for Q&A

**Deliverable**: Polished presentation delivery

### Acceptance Criteria
- ✅ Slides are professional and clear
- ✅ Presentation within time limit
- ✅ Demo works smoothly (if included)
- ✅ Prepared for questions

---

## Success Metrics Summary

### Minimum Requirements (Must Have)
- ✅ Both U-Net and 3D CNN implemented
- ✅ All three PDB features extracted
- ✅ Training pipeline functional
- ✅ Test accuracy > 70%
- ✅ 15-page paper in ACM format
- ✅ Reproducible code with documentation

### Target Goals (Should Have)
- 🎯 Multi-channel U-Net implemented
- 🎯 Comprehensive ablation studies
- 🎯 Test accuracy > 85%
- 🎯 Statistical model comparison
- 🎯 Publication-quality figures
- 🎯 Error analysis with insights

### Stretch Goals (Nice to Have)
- 🚀 Ensemble methods
- 🚀 Attention mechanisms
- 🚀 Additional features beyond proposal
- 🚀 Comparison with existing methods
- 🚀 Conference-ready paper

---

## Emergency Fallback Plans

### If Behind Schedule

**Week 10 (Progress Report Deadline)**:
- Minimum: Show baseline results, plan for rest
- Skip multi-channel U-Net if not ready
- Focus on core features (P-P distances minimum)

**Week 13 (Experiments)**:
- Prioritize main comparison (U-Net vs CNN)
- Reduce ablation studies to essentials
- Use smaller validation set for faster iteration

**Week 14-15 (Paper)**:
- Start writing early (don't wait for all experiments)
- Methods section can be written anytime
- Results section needs only key experiments

### If Technical Issues

**GPU Access**:
- Use Google Colab (free GPU)
- AWS/GCP credits for students
- Reduce model size and batch size

**Implementation Bugs**:
- Use example code as fallback
- Skip multi-channel U-Net (nice-to-have)
- Focus on solid implementation of core models

**Data Issues**:
- Use smaller subset for development
- Document and handle missing data gracefully
- Class imbalance can be addressed post-hoc

---

**Document Version**: 1.0  
**Last Updated**: November 20, 2025  
**Status**: Active Development  
**Current Phase**: Phase 1 (Research & Understanding)
