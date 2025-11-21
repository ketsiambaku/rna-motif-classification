# Detailed Documentation

## Table of Contents
1. [Development Guidelines](#development-guidelines)
2. [Biological Background](#biological-background)
3. [Technical Implementation](#technical-implementation)
4. [Feature Extraction Details](#feature-extraction-details)
5. [Model Architecture Details](#model-architecture-details)
6. [Training Best Practices](#training-best-practices)
7. [Evaluation and Metrics](#evaluation-and-metrics)
8. [Troubleshooting](#troubleshooting)
9. [Academic Paper Guidelines](#academic-paper-guidelines)

---

## Development Guidelines

### Code Style

#### Python Conventions
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use meaningful variable names (no single letters except loop indices)

#### Documentation Style
```python
def extract_torsion_angles(pdb_path: str, max_residues: int = 30) -> np.ndarray:
    """
    Extract backbone torsion angles from RNA PDB structure.
    
    Computes 7 torsion angles (α, β, γ, δ, ε, ζ, χ) for each residue
    following standard structural biology conventions.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file containing RNA structure
    max_residues : int, default=30
        Maximum number of residues to process
        
    Returns
    -------
    np.ndarray
        Array of shape (max_residues, 7) containing torsion angles in degrees.
        Missing angles are filled with 0.0.
        
    Notes
    -----
    Torsion angle definitions:
    - α (alpha): P(i-1) - O5' - C5' - C4'
    - β (beta): O5' - C5' - C4' - C3'
    - γ (gamma): C5' - C4' - C3' - O3'
    - δ (delta): C4' - C3' - O3' - P(i+1)
    - ε (epsilon): C3' - O3' - P(i+1) - O5'(i+1)
    - ζ (zeta): O3' - P(i+1) - O5'(i+1) - C5'(i+1)
    - χ (chi): O4' - C1' - N9 - C4 (purines) or O4' - C1' - N1 - C2 (pyrimidines)
    
    References
    ----------
    .. [1] Saenger, W. (1984). Principles of Nucleic Acid Structure.
    """
    pass  # Implementation here
```

#### Import Organization
```python
# Standard library imports
import os
import sys
from typing import Tuple, List, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser

# Local imports
from src.utils.geometry import compute_dihedral_angle
from src.features.pdb_utils import get_atom_coords
```

### Project Organization

#### Module Structure
```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset classes
│   ├── augmentation.py      # 3D augmentation functions
│   └── preprocessing.py     # MRC/PDB preprocessing
├── features/
│   ├── __init__.py
│   ├── pdb_extractor.py     # Feature extraction from PDB
│   ├── mask_generator.py    # Generate component masks
│   └── geometry.py          # Geometric calculations
├── models/
│   ├── __init__.py
│   ├── unet.py             # U-Net implementation
│   ├── cnn.py              # 3D CNN baseline
│   └── multi_channel_unet.py  # 4-channel U-Net
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training loop logic
│   ├── evaluator.py        # Evaluation and metrics
│   └── checkpoint.py       # Save/load utilities
└── utils/
    ├── __init__.py
    ├── visualization.py    # Plotting functions
    └── config.py           # Configuration management
```

### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/torsion-angles
# Make changes, commit frequently with descriptive messages
git commit -m "feat: add torsion angle extraction with proper angle conventions"
git push origin feature/torsion-angles
# Merge to main after review
```

### Testing Strategy
```python
# Unit tests for feature extraction
import pytest
import numpy as np

def test_pp_distances_shape():
    """Test P-P distance matrix has correct shape."""
    distances = extract_pp_distances('test_data/sample.pdb', max_residues=30)
    assert distances.shape == (900,)
    
def test_pp_distances_range():
    """Test P-P distances are in reasonable range."""
    distances = extract_pp_distances('test_data/sample.pdb')
    assert np.all(distances >= 0)
    assert np.all(distances <= 100)  # Ångströms

def test_torsion_angles_range():
    """Test torsion angles are in [-180, 180] degrees."""
    angles = extract_torsion_angles('test_data/sample.pdb')
    assert np.all(angles >= -180)
    assert np.all(angles <= 180)
```

---

## Biological Background

### RNA Secondary Structure

#### Structure Hierarchy
1. **Primary Structure**: Nucleotide sequence (A, U, G, C)
2. **Secondary Structure**: Base pairing patterns (what we're classifying)
3. **Tertiary Structure**: 3D spatial arrangement

#### Motif Types

**Bulge Loops**
- Definition: Unpaired nucleotides on one strand while opposite strand remains paired
- Biological Role: Flexible regions allowing conformational changes, protein binding sites
- Structural Feature: Asymmetric with continuous helix on one side
- Example: 5'-AAAA-3' bulge with 4 unpaired adenines

**Hairpin Loops**
- Definition: Single-stranded loop capping the end of a helix
- Biological Role: Recognition sites, catalytic centers, regulatory elements
- Structural Feature: Terminal loop with characteristic U-turn
- Example: GNRA tetraloops (G-N-R-A where N=any, R=purine)

**Internal Loops**
- Definition: Unpaired nucleotides on both strands between helices
- Biological Role: Hinge regions, recognition motifs
- Structural Feature: Symmetric or asymmetric interruption of helix
- Example: 2×2 internal loops common in ribosomal RNA

### Cryo-EM and Density Maps

#### What is Cryo-EM?
- Cryogenic Electron Microscopy: imaging biomolecules at near-atomic resolution
- Sample is frozen rapidly (vitrification) to preserve native structure
- Electron beam passes through, creating 2D projections
- Computational reconstruction produces 3D density map

#### Density Map Interpretation
- **High Density**: Regions with many electrons (heavy atoms like phosphorus)
- **Low Density**: Water, flexible regions, or empty space
- **Resolution**: Typically 2-4 Å for good quality RNA structures
- **File Format**: MRC (Medical Research Council) binary format

#### RNA Components in Density
1. **Phosphate Backbone**: Highest density (phosphorus atom)
2. **Ribose Sugar**: Moderate density (carbon, oxygen)
3. **Nucleotide Bases**: Varies by base type (purines denser than pyrimidines)

### PDB Structure Files

#### Coordinate System
- Cartesian coordinates (x, y, z) in Ångströms (Å)
- Origin typically at geometric center
- Right-handed coordinate system

#### Important Atoms for RNA

**Backbone Atoms:**
- `P`: Phosphate
- `O5'`, `O3'`: Backbone oxygens
- `C5'`, `C4'`, `C3'`: Ribose carbons

**Ribose Sugar:**
- `C1'`, `C2'`, `C3'`, `C4'`, `O4'`
- `O2'`: Distinguishes RNA from DNA (2'-OH group)

**Base Atoms (Pairing Detection):**
- Purines (A, G): `N9`, `N7`, `N3`, `N1`
- Pyrimidines (U, C): `N1`, `N3`, `O2`, `O4`
- Watson-Crick pairs: A-U (N1...N3), G-C (N1...N3, O6...N4)

---

## Technical Implementation

### Data Loading Pipeline

#### MRC File Processing
```python
import mrcfile
import numpy as np
import torch
import torch.nn.functional as F

def load_mrc_density(filepath: str, target_shape: Tuple[int, int, int] = (64, 64, 64)) -> torch.Tensor:
    """
    Load and preprocess MRC density map.
    
    Steps:
    1. Read binary MRC file
    2. Convert to float32
    3. Normalize to [0, 1]
    4. Resize to target shape
    5. Convert to PyTorch tensor
    """
    # Read MRC file (permissive=True handles minor format violations)
    with mrcfile.open(filepath, permissive=True) as mrc:
        density = mrc.data.astype(np.float32)
    
    # Normalize to [0, 1] range
    density_min = density.min()
    density_max = density.max()
    density = (density - density_min) / (density_max - density_min + 1e-8)
    
    # Add batch and channel dimensions [1, 1, D, H, W]
    density = torch.from_numpy(density).unsqueeze(0).unsqueeze(0)
    
    # Resize using trilinear interpolation
    density = F.interpolate(
        density, 
        size=target_shape, 
        mode='trilinear', 
        align_corners=False
    )
    
    # Return as [1, D, H, W]
    return density.squeeze(0)
```

#### PDB File Parsing
```python
from Bio.PDB import PDBParser
from typing import List, Dict
import numpy as np

def parse_pdb_structure(filepath: str) -> Dict:
    """
    Parse PDB file and extract relevant information.
    
    Returns dictionary with:
    - residues: List of residue objects
    - atoms: Dictionary mapping atom names to coordinates
    - chains: List of chain identifiers
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', filepath)
    
    result = {
        'residues': [],
        'atoms': {},
        'chains': []
    }
    
    for model in structure:
        for chain in model:
            result['chains'].append(chain.id)
            for residue in chain:
                result['residues'].append(residue)
                
                # Extract atom coordinates
                for atom in residue:
                    atom_name = f"{residue.id[1]}_{atom.name}"
                    result['atoms'][atom_name] = atom.get_coord()
    
    return result
```

### Feature Extraction Implementation

#### 1. P-P Distance Matrix
```python
def extract_pp_distances(pdb_path: str, max_residues: int = 30) -> np.ndarray:
    """
    Extract phosphate-phosphate distance matrix.
    
    Returns flattened distance matrix of shape (max_residues^2,).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_path)
    
    # Extract P atom coordinates
    p_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'P' in residue:
                    p_coords.append(residue['P'].get_coord())
                    if len(p_coords) >= max_residues:
                        break
    
    # Pad if necessary
    while len(p_coords) < max_residues:
        p_coords.append(np.zeros(3))
    
    p_coords = np.array(p_coords[:max_residues])
    
    # Compute pairwise distances
    dist_matrix = np.zeros((max_residues, max_residues))
    for i in range(max_residues):
        for j in range(max_residues):
            dist_matrix[i, j] = np.linalg.norm(p_coords[i] - p_coords[j])
    
    return dist_matrix.flatten()
```

#### 2. Torsion Angles
```python
def compute_dihedral_angle(p1: np.ndarray, p2: np.ndarray, 
                          p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Compute dihedral angle between 4 points using standard formula.
    
    Returns angle in degrees [-180, 180].
    """
    # Vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Normals to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    # Normalize
    n1 = n1 / (np.linalg.norm(n1) + 1e-8)
    n2 = n2 / (np.linalg.norm(n2) + 1e-8)
    b2_norm = b2 / (np.linalg.norm(b2) + 1e-8)
    
    # Compute angle
    m1 = np.cross(n1, b2_norm)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    angle = np.arctan2(y, x)
    return np.degrees(angle)

def extract_torsion_angles(pdb_path: str, max_residues: int = 30) -> np.ndarray:
    """
    Extract 7 backbone torsion angles per residue.
    
    Returns array of shape (max_residues, 7).
    """
    structure = parse_pdb_structure(pdb_path)
    residues = structure['residues']
    
    angles = np.zeros((max_residues, 7))
    
    for i, residue in enumerate(residues[:max_residues]):
        # Get atoms for this and adjacent residues
        # ... (detailed implementation of each angle)
        pass
    
    return angles.flatten()
```

#### 3. Base Pairing Matrix
```python
def extract_base_pairing(pdb_path: str, max_residues: int = 30, 
                        threshold: float = 3.5) -> np.ndarray:
    """
    Detect base pairing using N1/N3 distance criterion.
    
    Parameters
    ----------
    threshold : float
        Distance threshold in Ångströms for pairing detection (default: 3.5 Å)
    
    Returns
    -------
    np.ndarray
        Binary matrix of shape (max_residues^2,) indicating pairs
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', pdb_path)
    
    # Extract N1/N3 coordinates based on base type
    n_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                
                # Purines (A, G) use N1
                if resname in ['A', 'G', 'DA', 'DG']:
                    if 'N1' in residue:
                        n_coords.append(residue['N1'].get_coord())
                # Pyrimidines (U, C) use N3
                elif resname in ['U', 'C', 'T', 'DU', 'DC', 'DT']:
                    if 'N3' in residue:
                        n_coords.append(residue['N3'].get_coord())
                
                if len(n_coords) >= max_residues:
                    break
    
    # Pad if necessary
    while len(n_coords) < max_residues:
        n_coords.append(np.zeros(3))
    
    n_coords = np.array(n_coords[:max_residues])
    
    # Compute pairing matrix
    pair_matrix = np.zeros((max_residues, max_residues))
    for i in range(max_residues):
        for j in range(i+1, max_residues):  # Only upper triangle
            distance = np.linalg.norm(n_coords[i] - n_coords[j])
            if distance < threshold:
                pair_matrix[i, j] = 1
                pair_matrix[j, i] = 1  # Symmetric
    
    return pair_matrix.flatten()
```

---

## Model Architecture Details

### U-Net Classifier Implementation

```python
import torch
import torch.nn as nn

class UNetClassifier(nn.Module):
    """
    3D U-Net architecture for RNA motif classification.
    
    Combines volumetric density data with structural features.
    """
    
    def __init__(self, in_channels: int = 1, pdb_feat_dim: int = 2010, 
                 num_classes: int = 3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # 256 due to skip connection
        
        self.upconv2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        
        # Combine volume features with PDB features
        self.classifier = nn.Sequential(
            nn.Linear(32 + pdb_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """3D convolutional block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, volume: torch.Tensor, pdb_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        volume : torch.Tensor
            Shape [B, 1, 64, 64, 64] - density map
        pdb_features : torch.Tensor
            Shape [B, 2010] - structural features
            
        Returns
        -------
        torch.Tensor
            Shape [B, 3] - class logits
        """
        # Encoder with skip connections
        x1 = self.enc1(volume)       # [B, 32, 64, 64, 64]
        x = self.pool1(x1)           # [B, 32, 32, 32, 32]
        
        x2 = self.enc2(x)            # [B, 64, 32, 32, 32]
        x = self.pool2(x2)           # [B, 64, 16, 16, 16]
        
        x3 = self.enc3(x)            # [B, 128, 16, 16, 16]
        x = self.pool3(x3)           # [B, 128, 8, 8, 8]
        
        # Bottleneck
        x = self.bottleneck(x)       # [B, 256, 8, 8, 8]
        
        # Decoder with skip connections
        x = self.upconv3(x)          # [B, 128, 16, 16, 16]
        x = torch.cat([x, x3], dim=1)  # [B, 256, 16, 16, 16]
        x = self.dec3(x)             # [B, 128, 16, 16, 16]
        
        x = self.upconv2(x)          # [B, 64, 32, 32, 32]
        x = torch.cat([x, x2], dim=1)  # [B, 128, 32, 32, 32]
        x = self.dec2(x)             # [B, 64, 32, 32, 32]
        
        x = self.upconv1(x)          # [B, 32, 64, 64, 64]
        x = torch.cat([x, x1], dim=1)  # [B, 64, 64, 64, 64]
        x = self.dec1(x)             # [B, 32, 64, 64, 64]
        
        # Global pooling and classification
        x = self.global_pool(x)      # [B, 32, 1, 1, 1]
        x = self.flatten(x)          # [B, 32]
        
        # Concatenate with PDB features
        x = torch.cat([x, pdb_features], dim=1)  # [B, 32 + 2010]
        
        # Classification
        logits = self.classifier(x)  # [B, 3]
        
        return logits
```

### Multi-Channel U-Net

```python
class MultiChannelUNet(nn.Module):
    """
    4-channel U-Net: density + ribose mask + phosphate mask + base mask.
    """
    
    def __init__(self, pdb_feat_dim: int = 2010, num_classes: int = 3):
        super().__init__()
        
        # First convolution accepts 4 channels
        self.enc1 = self._conv_block(4, 32)  # 4 input channels
        # Rest is same as regular U-Net
        # ...
```

---

## Training Best Practices

### Data Augmentation

```python
import torch
import torch.nn.functional as F

def augment_3d_volume(volume: torch.Tensor, 
                      augment_prob: float = 0.5) -> torch.Tensor:
    """
    Apply random 3D augmentations to density volume.
    
    Augmentations:
    - Random 90° rotations on each axis
    - Random flips
    - Gaussian noise injection
    """
    # Random rotation (90, 180, 270 degrees on each axis)
    if torch.rand(1) < augment_prob:
        k = torch.randint(1, 4, (1,)).item()  # Number of 90° rotations
        axis = torch.randint(2, 5, (1,)).item()  # Which spatial dimension
        volume = torch.rot90(volume, k, dims=[axis-1, axis])
    
    # Random flip
    if torch.rand(1) < augment_prob:
        axis = torch.randint(2, 5, (1,)).item()
        volume = torch.flip(volume, dims=[axis])
    
    # Gaussian noise
    if torch.rand(1) < 0.3:
        noise = torch.randn_like(volume) * 0.05
        volume = volume + noise
        volume = torch.clamp(volume, 0, 1)
    
    return volume
```

### Class Balancing

```python
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(dataset, label_column='label'):
    """Create sampler for balanced class distribution."""
    labels = [dataset[i][2].item() for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
```

### Training Loop

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (volume, pdb_feat, labels) in enumerate(dataloader):
        # Move to device
        volume = volume.to(device).float()
        pdb_feat = pdb_feat.to(device).float()
        labels = labels.to(device).long()
        
        # Forward pass
        outputs = model(volume, pdb_feat)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
```

---

## Evaluation and Metrics

### Comprehensive Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataloader, device, class_names=['bulge', 'hairpin', 'internal']):
    """Comprehensive model evaluation."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for volume, pdb_feat, labels in dataloader:
            volume = volume.to(device).float()
            pdb_feat = pdb_feat.to(device).float()
            
            outputs = model(volume, pdb_feat)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Solutions:
# - Reduce batch size
# - Use gradient accumulation
# - Mixed precision training

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for volume, pdb_feat, labels in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(volume, pdb_feat)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 2. Missing Atoms in PDB
```python
# Handle gracefully with try-except
try:
    atom_coord = residue['P'].get_coord()
except KeyError:
    atom_coord = np.zeros(3)  # Use zero vector
    # Or skip this residue
```

#### 3. NaN Loss During Training
- Check for division by zero in normalization
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Validate input data for NaN/Inf values

---

## Academic Paper Guidelines

### ACM Format Requirements
- Two-column format
- 10pt font
- Maximum 15 pages (including references)
- Use ACM template: https://www.acm.org/publications/proceedings-template

### Recommended Structure

1. **Abstract** (150-250 words)
   - Problem statement
   - Approach
   - Key results
   - Significance

2. **Introduction** (1-2 pages)
   - Biological background
   - Problem motivation
   - Related work
   - Contributions

3. **Methods** (3-4 pages)
   - Dataset description
   - Feature engineering
   - Model architectures
   - Training procedure

4. **Experiments** (3-4 pages)
   - Experimental setup
   - Baseline comparisons
   - Ablation studies
   - Statistical analysis

5. **Results** (2-3 pages)
   - Classification performance
   - Confusion matrices
   - Feature importance
   - Comparison with related work

6. **Discussion** (1-2 pages)
   - Interpretation of results
   - Limitations
   - Biological insights

7. **Conclusion** (0.5-1 page)
   - Summary
   - Future work

### Figures and Tables
- High-resolution (300 DPI minimum)
- Clear captions with detailed descriptions
- Color-blind friendly palettes
- Statistical significance indicators

### Citations
- Use BibTeX for reference management
- Cite: original papers for methods, datasets, related work
- Format: Author-year style for ACM
