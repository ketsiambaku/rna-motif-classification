# Related Work - Literature Review

**Date**: November 28, 2025  
**Status**: In Progress  
**Papers Reviewed**: 0/7 minimum

---

## Overview

This literature review covers relevant work in:
1. RNA secondary structure prediction and classification
2. Deep learning for cryo-EM density map analysis
3. 3D convolutional neural networks for volumetric data
4. Feature extraction from molecular structures
5. Multi-modal fusion in biomedical imaging

## Search Strategy

**Databases**: PubMed, arXiv, Google Scholar, IEEE Xplore  
**Keywords**: 
- "RNA secondary structure prediction"
- "cryo-EM deep learning"
- "3D U-Net biomedical"
- "RNA loop classification"
- "molecular structure CNN"
- "multi-modal fusion protein structure"

**Inclusion Criteria**:
- Published in peer-reviewed venues (journals or conferences)
- Relevant to RNA structure, cryo-EM, or deep learning on molecular data
- Recent work (2018-2025 preferred)
- Methodologically sound

---

## Paper Summaries

### 1. RNA Secondary Structure Prediction with Deep Learning

**Paper**: RNA secondary structure prediction using deep learning with thermodynamic integration

**Authors**: K. Sato, M. Akiyama, Y. Sakakibara

**Venue**: Nature Communications

**Year**: 2021

**Problem**: Predicting accurate RNA secondary structures while avoiding overfitting in highly parameterized machine learning models.

**Method**: 
- Integrates deep neural network learned folding scores with Turner's nearest-neighbor free energy parameters
- Uses thermodynamic regularization to prevent overfitting
- Combines data-driven learning with physics-based constraints

**Key Findings**: 
- Thermodynamic integration significantly reduces overfitting
- Achieved state-of-the-art accuracy in RNA structure prediction
- Physics-based regularization improves generalization
- Model learns meaningful biological patterns rather than dataset artifacts

**Relevance to Our Work**: 
- Demonstrates importance of combining learned features with domain knowledge
- Our approach similarly combines learned density features with structural PDB features
- Shows value of regularization to prevent overfitting (relevant for our leakage concerns)
- Validates multi-modal approach (data + physics)

**Limitations**: 
- Focused on sequence-based prediction, not volumetric data
- Computational cost of thermodynamic integration
- Requires high-quality training data with known structures

**Citation (BibTeX)**:
```bibtex
@article{sato2021rna,
  title={RNA secondary structure prediction using deep learning with thermodynamic integration},
  author={Sato, Kengo and Akiyama, Manato and Sakakibara, Yasubumi},
  journal={Nature Communications},
  volume={12},
  number={1},
  pages={941},
  year={2021},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-021-21194-4}
}
```

---

### 2. Deep Learning for Cryo-EM Density Maps

**Paper**: Deep learning to predict protein backbone structure from high-resolution cryo-EM density maps

**Authors**: D. Si, S.A. Moritz, J. Pfab, J. Hou, R. Cao, L. Wang, T. Wu, J. Cheng

**Venue**: Scientific Reports

**Year**: 2020

**Problem**: Predicting protein backbone structure (Cα positions) directly from cryo-EM density maps without requiring initial atomic models.

**Method**: 
- C-CNN (Cascaded-CNN) architecture with 3D convolutional layers
- Takes 3D density maps as input, outputs predicted Cα coordinates
- Multi-scale feature extraction at different density resolutions
- End-to-end learning without manual feature engineering

**Key Findings**: 
- Achieved high accuracy (RMSD < 2Å) on 50 experimental density maps
- Performance improves with map resolution (better at 3-4Å than 6-8Å)
- 3D CNNs can learn structural patterns directly from density
- Fully automated approach outperforms semi-automated methods

**Relevance to Our Work**: 
- Validates 3D CNN/U-Net for cryo-EM density analysis
- Demonstrates that volumetric deep learning works on real experimental data
- Our density-only model (58.51%) proves concept works for RNA classification
- Shows importance of resolution quality (relevant for our variable dimensions)

**Limitations**: 
- Requires high-resolution maps (< 4Å) for best performance
- Computationally expensive for large structures
- Limited to backbone prediction, not full atomic detail
- Protein-focused, RNA has different characteristics

**Citation (BibTeX)**:
```bibtex
@article{si2020deep,
  title={Deep learning to predict protein backbone structure from high-resolution cryo-EM density maps},
  author={Si, Dong and Moritz, Spencer A and Pfab, Jonas and Hou, Jie and Cao, Renzhi and Wang, Liguo and Wu, Tianqi and Cheng, Jianlin},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={4282},
  year={2020},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-020-60598-y}
}
```

---

### 3. 3D U-Net Applications

**Paper**: [To be filled]

**Authors**: 

**Venue**: 

**Year**: 

**Problem**: 

**Method**: 

**Key Findings**: 

**Relevance to Our Work**: 

**Limitations**: 

**Citation (BibTeX)**:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  doi={}
}
```

---

### 4. Feature Extraction from PDB Structures

**Paper**: [To be filled]

**Authors**: 

**Venue**: 

**Year**: 

**Problem**: 

**Method**: 

**Key Findings**: 

**Relevance to Our Work**: 

**Limitations**: 

**Citation (BibTeX)**:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  doi={}
}
```

---

### 5. RNA Loop Classification

**Paper**: [To be filled]

**Authors**: 

**Venue**: 

**Year**: 

**Problem**: 

**Method**: 

**Key Findings**: 

**Relevance to Our Work**: 

**Limitations**: 

**Citation (BibTeX)**:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  doi={}
}
```

---

### 6. Multi-Modal Fusion in Molecular Analysis

**Paper**: [To be filled]

**Authors**: 

**Venue**: 

**Year**: 

**Problem**: 

**Method**: 

**Key Findings**: 

**Relevance to Our Work**: 

**Limitations**: 

**Citation (BibTeX)**:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  doi={}
}
```

---

### 7. Deep Learning for Volumetric Medical Imaging

**Paper**: [To be filled]

**Authors**: 

**Venue**: 

**Year**: 

**Problem**: 

**Method**: 

**Key Findings**: 

**Relevance to Our Work**: 

**Limitations**: 

**Citation (BibTeX)**:
```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={},
  pages={},
  doi={}
}
```

---

## Comparative Analysis

### Common Approaches

| Approach | Papers | Advantages | Disadvantages |
|----------|--------|------------|---------------|
| Traditional ML | | | |
| CNNs | | | |
| U-Net variants | | | |
| Sequence-based | | | |
| Structure-based | | | |
| Hybrid methods | | | |

### Performance Benchmarks

| Method | Dataset | Metric | Performance | Year |
|--------|---------|--------|-------------|------|
| | | | | |

---

## Research Gaps & Our Contribution

### Identified Gaps

1. **Gap 1**: [To be filled after reading]
   - **Description**: 
   - **How we address**: 

2. **Gap 2**: 
   - **Description**: 
   - **How we address**: 

3. **Gap 3**: 
   - **Description**: 
   - **How we address**: 

### Our Novel Contributions

1. **Multi-modal fusion** of cryo-EM density maps with size-invariant PDB features
   - Most work uses either density OR structure, not both
   - Our hybrid approach combines complementary information

2. **Size-invariant feature engineering**
   - First to identify and quantify size leakage in RNA motif classification
   - Developed methodology for detecting feature leakage
   - Created size-invariant features (sequence, base pairing, torsion)

3. **Leakage detection framework**
   - Novel approach: PDB-only training to detect leakage
   - Sparsity analysis as leakage indicator
   - Applicable to other structural biology ML problems

4. **3-class coarse topology classification**
   - Focus on learnable structural patterns vs size
   - Realistic baseline (58.51%) established

---

## Key Takeaways for Implementation

### Best Practices

1. **Data preprocessing**:
   - 
   
2. **Model architecture**:
   - 

3. **Training strategies**:
   - 

4. **Feature engineering**:
   - 

5. **Evaluation**:
   - 

### Lessons Learned from Prior Work

1. 

2. 

3. 

---

## Bibliography

### Full BibTeX Export

```bibtex
% RNA Secondary Structure
@article{paper1,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}

% Cryo-EM Deep Learning
@article{paper2,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}

% 3D U-Net
@article{paper3,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}

% PDB Feature Extraction
@article{paper4,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}

% RNA Loop Classification
@article{paper5,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}

% Multi-Modal Fusion
@article{paper6,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}

% Volumetric Medical Imaging
@article{paper7,
  title={},
  author={},
  journal={},
  year={},
  doi={}
}
```

---

## Papers to Read (Priority Queue)

### High Priority
- [ ] RNA secondary structure prediction methods (foundational)
- [ ] Deep learning for cryo-EM (directly relevant)
- [ ] 3D U-Net original paper (architecture basis)

### Medium Priority
- [ ] PDB feature extraction methods
- [ ] RNA loop classification
- [ ] Multi-modal fusion approaches

### Low Priority (if time permits)
- [ ] Attention mechanisms for molecular data
- [ ] Transfer learning in structural biology
- [ ] Ensemble methods for classification

---

## Reading Notes Template

For each paper, take notes on:

1. **Problem formulation**
   - What exactly are they trying to predict/classify?
   - Input/output types?
   - Dataset characteristics?

2. **Method overview**
   - Architecture details
   - Feature engineering
   - Training procedure
   - Novel contributions

3. **Experimental setup**
   - Dataset size and splits
   - Evaluation metrics
   - Baseline comparisons
   - Ablation studies

4. **Results**
   - Main performance numbers
   - Comparison with baselines
   - Statistical significance
   - Failure cases discussed?

5. **Critical analysis**
   - Strengths
   - Limitations
   - Reproducibility concerns
   - Relevance to our work

6. **Ideas to borrow**
   - Techniques we should implement
   - Evaluation strategies
   - Presentation approaches

---

**Status**: Template created, ready to begin literature search and reading

**Next Steps**:
1. Search for papers using defined keywords
2. Read and summarize minimum 5-7 papers
3. Complete comparative analysis
4. Identify research gaps
5. Finalize BibTeX citations for paper

**Estimated Time**: 8-10 hours (1-2 hours per paper + analysis)

---

### 3. 3D U-Net for Volumetric Segmentation

**Paper**: 3D U-Net: learning dense volumetric segmentation from sparse annotation

**Authors**: Ö. Çiçek, A. Abdulkadir, S.S. Lienkamp, T. Brox, O. Ronneberger

**Venue**: International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)

**Year**: 2016

**Problem**: Performing volumetric segmentation of 3D biomedical images with limited annotated training data.

**Method**: 
- Extends 2D U-Net to 3D with volumetric convolutions and pooling
- Encoder-decoder architecture with skip connections
- Weighted loss function to handle class imbalance
- Data augmentation through elastic deformations
- Can train on sparsely annotated volumes

**Key Findings**: 
- Successfully segments 3D structures with minimal annotations
- Skip connections crucial for preserving spatial information
- Outperforms 2D slice-by-slice approaches
- Generalizes well to different biomedical imaging tasks
- Cited 10,000+ times - highly influential architecture

**Relevance to Our Work**: 
- **Direct architectural basis** for our density-only U-Net model
- Validates encoder-decoder with skip connections for volumetric data
- Weighted loss strategy applicable to our severe class imbalance (76:1)
- Data augmentation techniques transferable to our MRC volumes
- Proves 3D convolutions learn meaningful volumetric features

**Limitations**: 
- Computationally expensive (3D convolutions >> 2D)
- Requires substantial GPU memory
- May lose fine details in deeper layers
- Designed for segmentation, not classification (we adapt it)

**Citation (BibTeX)**:
```bibtex
@inproceedings{cicek20163d,
  title={3D U-Net: learning dense volumetric segmentation from sparse annotation},
  author={{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S and Brox, Thomas and Ronneberger, Olaf},
  booktitle={International conference on medical image computing and computer-assisted intervention},
  pages={424--432},
  year={2016},
  organization={Springer},
  doi={10.1007/978-3-319-46723-8_49}
}
```

---

