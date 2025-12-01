---
module-name: "RNA Motif Classification"
version: "0.4.0"
description: "Deep learning system for classifying RNA secondary structure motif types (bulge loops, internal loops, hairpin loops) from cryo-EM density maps with size-invariant structural features"
status: "Phase 1 Complete | Paper Sections 1-2 Complete | Phase 2 Ready"
phase: "Feature Engineering (Size-Invariant PDB Features) + Paper Writing"
related-modules:
  - name: Example Training Code
    path: ./example/TrainingForClassification
  - name: Project Documentation
    path: ./docs
  - name: Development Phases (includes Phase 1.4 paper writing)
    path: ./docs/development-phases.md
  - name: Leakage Analysis Report
    path: ./docs/dataset2-leakage-analysis.md
  - name: Dataset Statistics Report
    path: ./docs/dataset-statistics.md
  - name: Literature Review
    path: ./docs/related-work.md
  - name: IEEE Format Paper (Sections 1-2 Complete)
    path: ./docs/paper.md
  - name: Reference Summaries
    path: ./docs/references/
architecture:
  style: "Hybrid Deep Learning with Multi-Modal Fusion (Density + Size-Invariant Features)"
  components:
    - name: "Data Processing"
      description: "MRC density map loading, size-invariant PDB feature extraction"
    - name: "Size-Invariant Feature Engineering"
      description: "RNA sequence features, base pairing topology, torsion angles (optional)"
      status: "In Development"
    - name: "Model Architectures"
      description: "Density-only U-Net (baseline: 58.51%) + Hybrid U-Net (target: 70-75%)"
    - name: "Training Pipeline"
      description: "PyTorch training with M1 GPU optimization, class weighting, early stopping"
    - name: "Evaluation"
      description: "Leakage detection, classification metrics, ablation studies"
  patterns:
    - name: "Multi-Modal Fusion"
      usage: "Combines 3D volumetric density with size-invariant sequence and pairing features"
    - name: "Leakage Detection"
      usage: "PDB-only training, sparsity analysis, correlation with problem-specific artifacts"
    - name: "Data Augmentation"
      usage: "3D rotations, flips, noise injection (planned for Phase 3)"
key-findings:
  - name: "Data Leakage Discovery"
    description: "PDB phosphate distance matrix has 31.78% sparsity range encoding motif size"
    impact: "76.8pp accuracy inflation (98.8% → 21.99%)"
    solution: "Size-invariant features: sequence + base pairing + torsion angles (optional)"
  - name: "Realistic Baseline"
    description: "Density-only 3D U-Net achieves 58.51% on 3-class topology classification"
    significance: "Proves model CAN learn structural patterns without leakage"
  - name: "Task Selection"
    description: "3-class coarse (bulge/hairpin/internal) is realistic vs 14-class fine-grained (21.99%)"
  - name: "Literature Review Complete"
    description: "7 papers integrated covering RNA structure, cryo-EM, deep learning, motif analysis"
    papers: "MXfold2, DeepTracer 1.0/2.0, AlphaFold 3, DeepCryoRNA, RNAMotifComp, CaCoFold-R3D"
    impact: "Identified 5 research gaps positioning our work (cryo-EM motif classification, size-invariance, multi-modal fusion, overlapping families, local/global context)"
  - name: "Paper Progress"
    description: "IEEE format paper Sections 1 (Introduction) and 2 (Related Work) complete"
    content: "Problem statement with formal notation, 4 key challenges, 7 comprehensive subsections covering state-of-the-art"
    status: "700+ lines written, ready for Methodology section after Phase 2 implementation"
development:
  stack:
    - PyTorch (with MPS for Apple Silicon M1)
    - BioPython
    - mrcfile
    - NumPy
    - pandas
    - scikit-learn
  setup:
    - "Python 3.8+"
    - "Apple Silicon M1 GPU (MPS) or CUDA-capable GPU"
    - "Install: pip install torch biopython mrcfile numpy pandas scikit-learn"
  current-phase:
    phase: "2.1"
    task: "Implementing RNA sequence feature extraction"
    timeline: "Week 5 (3-4 days)"
    expected-result: "65-70% accuracy with density + sequence features"
---

# RNA Motif Classification

## Project Overview

This project implements a deep learning classification system for predicting RNA secondary structure motif types from cryo-EM (cryogenic electron microscopy) density maps and their corresponding structural data.

### Problem Statement

**Input:**
- 3D density map segment (`.mrc` file) - volumetric electron density from cryo-EM
- Structural coordinates (`.pdb` file) - atomic positions of RNA nucleotides

**Output:**
- Classification into one of 3 RNA motif types:
  1. **Bulge loops** - unpaired nucleotides on one strand
  2. **Internal loops** - unpaired nucleotides on both strands between helices
  3. **Hairpin loops** - single-stranded loop at helix terminus

### Academic Context

This is a term project for an ML course requiring:
- Implementation of at least 2 algorithms for comparison
- Real-world biomedical data application
- ACM conference paper format documentation (max 15 pages)
- Reproducible results with complete source code
- Statistical analysis and performance comparison

## Architecture Overview

### Data Flow

```
.mrc file → Preprocessing → 3D Tensor [1, 64, 64, 64]
                                ↓
.pdb file → Feature Extraction → Feature Vector [2010]
                                ↓
                    Combined Input to Model
                                ↓
                    U-Net or 3D CNN Classifier
                                ↓
                    Softmax → Class Prediction
```

### Models

#### 1. U-Net Classifier (Primary)
- **Architecture**: 3D U-Net with encoder-decoder structure
- **Input**: Volumetric density [B, 1, 64, 64, 64] + PDB features [B, 2010]
- **Fusion**: Late fusion - concatenate encoded volume with PDB features
- **Output**: 3-class probabilities

#### 2. Multi-Channel U-Net (Enhanced)
- **Architecture**: 4-channel input U-Net
- **Channels**:
  1. Original density map
  2. Ribose mask (sugar component)
  3. Phosphate mask (backbone)
  4. Base mask (nucleotide bases)
- **Advantage**: Explicitly models RNA components

#### 3. Simple 3D CNN (Baseline)
- **Architecture**: Convolutional encoder with pooling
- **Purpose**: Baseline comparison for U-Net performance
- **Design**: Simpler than U-Net, no skip connections

### Feature Engineering

From PDB files, we extract:

1. **Phosphate-Phosphate (P-P) Distances** (900 features)
   - Pairwise distances between all phosphate atoms
   - Captures backbone geometry

2. **Torsion Angles** (210 features)
   - 7 backbone angles per residue (α, β, γ, δ, ε, ζ, χ)
   - Describes local conformational state

3. **Base Pairing Matrix** (900 features)
   - Binary matrix indicating Watson-Crick pairing
   - Detection via N1/N3 atom proximity

**Total**: 2010 structural features per sample

## Dataset Organization

```
dataset/
├── train/
│   ├── bulge/       # Training samples for bulge loops
│   │   ├── {PDB_ID}_{number}.mrc
│   │   └── {PDB_ID}_{number}.pdb
│   ├── hairpin/     # Training samples for hairpin loops
│   └── internal/    # Training samples for internal loops
└── test/
    ├── bulge/       # Test samples
    ├── hairpin/
    └── internal/
```

**File Naming**: `{PDB_ID}_{segment_number}.{mrc|pdb}`
- PDB_ID: 4-character Protein Data Bank identifier
- segment_number: Unique identifier for extracted region

## Key Implementation Details

### Data Processing
1. **MRC Loading**: Use `mrcfile` library with permissive mode
2. **Normalization**: Scale density to [0, 1] range
3. **Resizing**: Standardize to 64×64×64 via trilinear interpolation
4. **PDB Parsing**: Use BioPython's PDBParser

### Training Configuration
- **Batch Size**: 4 (GPU memory constrained)
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss (multi-class)
- **Epochs**: 50-100 with early stopping
- **Validation**: 20% of training data

### Evaluation Metrics
- Overall classification accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- ROC curves (one-vs-rest)
- Statistical significance tests for model comparison

## Current Status

**Baseline Implementation**: Available in `example/TrainingForClassification/`
- Simple U-Net with P-P distance features
- Training loop with validation
- Basic preprocessing and dataset loading

**Phase 1 - Research & Understanding**: IN PROGRESS
1. Baseline code analysis and performance measurement
2. Dataset statistics generation
3. Literature review for paper background

**Planned Enhancements**:
1. Add torsion angle extraction
2. Add base pairing detection
3. Implement multi-channel U-Net
4. Implement 3D CNN baseline
5. Add data augmentation (rotations, flips, noise)
6. Comprehensive evaluation suite
7. Model comparison and ablation studies

## Getting Started

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install torch torchvision biopython mrcfile numpy pandas scikit-learn matplotlib seaborn
```

### Quick Start
```bash
# Navigate to example code
cd example/TrainingForClassification

# Generate dataset CSV
bash generate_csv.sh

# Train baseline model
python train.py
```

### Project Structure
```
rna-motif-classification/
├── .context/              # This documentation
├── dataset/               # Training and test data
├── example/               # Reference implementation
├── src/                   # Main source code (to be developed)
│   ├── data/             # Dataset loaders, augmentation
│   ├── models/           # U-Net, CNN implementations
│   ├── features/         # PDB feature extraction
│   ├── training/         # Training and evaluation
│   └── utils/            # Helper functions
└── docs/                  # Project proposals and guidelines
```

## Known Issues

1. **Mask Generation Bug**: Code in `example/label/` has identical ribose and sugar labels - needs fixing before multi-channel U-Net implementation

2. **Missing Atoms**: Some PDB files may have incomplete atom records - need graceful error handling

3. **Class Imbalance**: Dataset may have unequal class distribution - consider class weights or oversampling

## Development Workflow

1. **Baseline Analysis**: Understand and document current implementation
2. **Feature Implementation**: Add PDB feature extractors (torsion angles, base pairing)
3. **Model Development**: Implement CNN baseline, then multi-channel U-Net
4. **Experimentation**: Train models with different feature combinations
5. **Evaluation**: Compare performance with statistical tests
6. **Documentation**: Maintain ACM paper format throughout
7. **Reproducibility**: Document all hyperparameters and random seeds

## Contributing Guidelines

- Follow NumPy/SciPy docstring style
- Use type hints for function signatures
- Document biological/structural concepts when relevant
- Test feature extraction on known structures
- Validate dimensions at each pipeline stage
- Keep academic paper context in mind for all documentation

## References

- Cryo-EM density maps: EMDB (Electron Microscopy Data Bank)
- PDB structures: RCSB Protein Data Bank
- RNA secondary structure: Leontis & Westhof classification
- Deep learning for structural biology: AlphaFold, RoseTTAFold approaches
