---
module-name: "RNA Motif Classification"
version: "0.1.0"
description: "Deep learning system for classifying RNA secondary structure motif types (bulge loops, internal loops, hairpin loops) from cryo-EM density maps and structural features"
related-modules:
  - name: Example Training Code
    path: ./example/TrainingForClassification
  - name: Project Documentation
    path: ./docs
architecture:
  style: "Deep Learning Pipeline with Multi-Modal Fusion"
  components:
    - name: "Data Processing"
      description: "MRC density map loading, PDB feature extraction, data augmentation"
    - name: "Feature Engineering"
      description: "P-P distances, torsion angles, base pairing matrix extraction from PDB structures"
    - name: "Model Architectures"
      description: "U-Net classifier and 3D CNN for volumetric density classification"
    - name: "Training Pipeline"
      description: "PyTorch training loop with validation, checkpointing, and metrics tracking"
    - name: "Evaluation"
      description: "Classification metrics, confusion matrices, model comparison"
  patterns:
    - name: "Multi-Modal Fusion"
      usage: "Combines 3D volumetric density data with structural features from PDB files"
    - name: "Transfer Learning"
      usage: "Pre-trained feature extractors adapted for RNA structure classification"
    - name: "Data Augmentation"
      usage: "3D rotations, flips, and noise injection to increase effective dataset size"
development:
  stack:
    - PyTorch
    - BioPython
    - mrcfile
    - NumPy
    - pandas
    - scikit-learn
  setup:
    - "Python 3.8+"
    - "CUDA-capable GPU recommended"
    - "Install: pip install torch biopython mrcfile numpy pandas scikit-learn"
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

1. **Feature Implementation**: Start with additional PDB feature extractors
2. **Model Development**: Implement CNN baseline, then multi-channel U-Net
3. **Experimentation**: Train models with different feature combinations
4. **Evaluation**: Compare performance with statistical tests
5. **Documentation**: Maintain ACM paper format throughout
6. **Reproducibility**: Document all hyperparameters and random seeds

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
