# GitHub Copilot Instructions for RNA Motif Classification Project

## Project Overview
This is a machine learning classification project for predicting RNA secondary structure motif types (bulge loops, internal loops, and hairpin loops) from cryo-EM density maps (.mrc files) and their corresponding structural features (.pdb files).

## Project Context & Goals

### Primary Objective
Given a density map segment (.mrc file) and features extracted from a deposited structure segment (.pdb file), predict which of 3 motif classes the segment belongs to:
- Bulge loops
- Internal loops  
- Hairpin loops

### Models Being Implemented
1. **U-Net** (3D segmentation architecture)
2. **Simple 3D CNN** (baseline comparison)

### Academic Context
- This is a term project following ACM conference paper format
- Must apply at least 2 algorithms and compare performance
- Focus on classification problem with real-world biomedical data
- Final deliverable: 15-page ACM format paper + code + reproducible results

## Technical Architecture

### Input Data Structure
```
dataset/
├── train/
│   ├── bulge/     # .mrc and .pdb file pairs
│   ├── hairpin/   # .mrc and .pdb file pairs
│   └── internal/  # .mrc and .pdb file pairs
└── test/
    ├── bulge/
    ├── hairpin/
    └── internal/
```

### Feature Engineering Requirements

#### Current Features (from example code)
- Backbone atom distances from PDB structures

#### Required New Features to Implement
1. **P–P distances** (phosphate-phosphate, 900 features)
2. **Torsion angles** (7 angles per residue, 7×30 = 210 features)
3. **Base pairing matrix** (900 features) using simple pairing detector:
   ```
   For each residue pair (i,j):
   if N1/N3 atoms distance < threshold → paired
   else unpaired
   ```

#### Multi-Channel U-Net Enhancement
Convert single-channel U-Net to **4-channel input**:
1. Original density map (.mrc)
2. Ribose mask
3. Phosphate mask
4. Base mask

**Note**: Example mask extraction code in `example/label/` has bugs:
- Ribose and sugar labels are identical (should be different)
- Should create: backbone, ribose, and base masks
- Logic needs correction before integration

## Code Style & Development Guidelines

### Python Development
- Use PyTorch for deep learning models
- Follow scientific computing best practices (NumPy, SciPy for feature extraction)
- Use BioPython or similar for PDB file parsing
- Type hints for function signatures
- Docstrings following NumPy/SciPy style

### Data Processing
- Implement data augmentation techniques for improving accuracy
- Use proper train/validation/test splits
- Implement cross-validation if dataset is small
- Handle .mrc files with mrcfile library
- Handle .pdb files with BioPython or MDAnalysis

### Model Development
- Implement both U-Net and 3D CNN for comparison
- Track metrics: accuracy, precision, recall, F1-score, confusion matrix
- Save model checkpoints during training
- Implement early stopping
- Log experiments (consider using tensorboard or wandb)

### Code Organization
```
src/
├── data/           # Dataset loaders, augmentation
├── models/         # U-Net, 3D CNN implementations
├── features/       # PDB feature extraction (distances, angles, pairing)
├── preprocessing/  # Mask generation, normalization
├── training/       # Training loops, evaluation
└── utils/          # Helper functions
```

## When Writing Code

### Feature Extraction
- When extracting P-P distances, torsion angles, or base pairing: reference structural biology conventions
- Validate extracted features have correct dimensions (900 for distances/pairing, 210 for torsions)
- Handle missing atoms gracefully in PDB files

### Model Architecture
- For U-Net: implement 3D version suitable for volumetric data
- For multi-channel U-Net: ensure proper concatenation of 4 input channels
- Document architecture choices (kernel sizes, pooling, activation functions)
- Implement proper 3D convolutions for volumetric .mrc data

### Mask Generation
- Fix bugs in existing mask extraction code before reusing
- Ensure masks are properly aligned with density maps
- Validate mask quality (proper separation of ribose/phosphate/base)

### Training & Evaluation
- Implement proper class balancing (3 classes may be imbalanced)
- Use appropriate loss function (CrossEntropyLoss for 3-class classification)
- Compare model performance with statistical significance tests
- Generate visualizations: confusion matrices, ROC curves, training curves

### Reproducibility
- Set random seeds for reproducibility
- Document all hyperparameters
- Save preprocessing parameters
- Version control dataset splits

## Documentation Standards

### Code Comments
- Explain WHY, not just WHAT
- Reference papers or biological concepts when relevant
- Document assumptions (e.g., distance thresholds for base pairing)

### README Requirements
- Installation instructions with exact package versions
- Step-by-step data preparation
- Training commands with all flags explained
- Evaluation and reproduction instructions
- Expected results/benchmarks

### Paper Writing Support
When generating documentation or analysis:
- Use scientific/academic tone
- Include method justifications
- Compare with related work
- Provide quantitative results with error bars/confidence intervals
- Suggest visualizations appropriate for academic papers

## Optimization & Performance

### Suggested Improvements to Consider
- Ensemble methods combining U-Net and CNN predictions
- Additional features: secondary structure annotations, solvent accessibility, electrostatic properties
- Transfer learning from pre-trained molecular models
- Attention mechanisms for focusing on important regions
- Data augmentation: rotations, flips, noise injection in density maps

### Performance Metrics Priority
1. Classification accuracy (primary metric)
2. Per-class precision/recall (handle class imbalance)
3. Confusion matrix analysis
4. Model size and inference time (secondary)

## Important Conventions

### File Naming
- Dataset files follow pattern: `{PDB_ID}_{number}.{mrc|pdb}`
- Model checkpoints: `{model_name}_epoch{n}_acc{accuracy}.pth`
- Results: use timestamps and descriptive names

### Units & Scales
- Distances in Ångströms (Å)
- Angles in degrees or radians (be consistent, document choice)
- Normalize/standardize features appropriately

### Validation
- Always validate feature extraction against known structures
- Sanity check dimensions at each pipeline stage
- Visualize intermediate results when debugging

## Project-Specific Reminders
- This is a **classification** problem with **3 classes**
- Must implement **at least 2 algorithms** (U-Net + 3D CNN)
- Dataset is organized by class in separate folders
- Example code exists in `example/TrainingForClassification/` - understand it before extending
- Fix bugs in mask generation code before using
- Focus on **feature engineering** and **model comparison** for academic contribution

## Communication Style
- Explain biological/structural biology concepts when they arise
- Suggest relevant papers or techniques from bioinformatics/structural biology
- When suggesting improvements, consider both ML performance and biological interpretability
- Be explicit about trade-offs (e.g., model complexity vs. interpretability)
