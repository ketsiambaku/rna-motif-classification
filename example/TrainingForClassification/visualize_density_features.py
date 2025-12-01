import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def visualize_density_maps():
    """
    Visualize what 3D density maps actually contain and what features
    a CNN could learn from them.
    """
    base_dir = Path('/Users/ketsiambaku/Repositories/rna-motif-classification')
    dataset_dir = base_dir / 'dataset2'
    
    # Load a few examples from different classes (using actual files from dataset2)
    examples = {
        'bulge1': 'bulge1/4V6N_852.mrc',
        'bulge2': 'bulge2/4V50_700.mrc',
        'hairpin5': 'hairpin5/4V50_701.mrc',
        '2x2': '2x2/4V50_702.mrc',
        '4x4': '4x4/6Q97_73.mrc',
    }
    
    print("="*70)
    print("3D DENSITY MAP FEATURE ANALYSIS")
    print("="*70)
    
    fig, axes = plt.subplots(len(examples), 4, figsize=(20, 4*len(examples)))
    
    for idx, (label, filepath) in enumerate(examples.items()):
        full_path = dataset_dir / filepath
        
        if not full_path.exists():
            print(f"⚠️  File not found: {filepath}")
            continue
            
        # Load density map
        with mrcfile.open(full_path, permissive=True) as mrc:
            density = mrc.data.astype(np.float32)
        
        print(f"\n{label.upper()} ({filepath})")
        print(f"  Shape: {density.shape}")
        print(f"  Value range: [{density.min():.3f}, {density.max():.3f}]")
        print(f"  Mean: {density.mean():.3f}, Std: {density.std():.3f}")
        print(f"  Nonzero voxels: {np.count_nonzero(density)} / {density.size} ({100*np.count_nonzero(density)/density.size:.1f}%)")
        
        # Calculate 3D features that CNN could learn
        print(f"\n  3D STRUCTURAL FEATURES (what CNN learns):")
        
        # 1. Density gradients (edges, boundaries)
        gradients = np.gradient(density)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        print(f"    - Edge strength (gradient): mean={gradient_magnitude.mean():.3f}, max={gradient_magnitude.max():.3f}")
        
        # 2. Local connectivity patterns
        from scipy.ndimage import label
        binary = density > density.mean()
        labeled, num_features = label(binary)
        print(f"    - Connected components: {num_features} (spatial clustering)")
        
        # 3. Density distribution
        high_density = np.sum(density > density.mean() + density.std())
        print(f"    - High-density voxels: {high_density} (nucleotide cores)")
        
        # 4. Spatial extent
        nonzero_coords = np.argwhere(density > density.mean())
        if len(nonzero_coords) > 0:
            extent = nonzero_coords.max(axis=0) - nonzero_coords.min(axis=0)
            print(f"    - Spatial extent: {extent} (bounding box)")
        
        # 5. Symmetry/asymmetry
        center = np.array(density.shape) // 2
        if density.shape[0] == density.shape[1] == density.shape[2]:
            # Check rotational patterns
            print(f"    - Geometric arrangement (loop closure, base stacking)")
        
        # Visualize: XY, XZ, YZ slices + 3D projection
        mid_z = density.shape[0] // 2
        mid_y = density.shape[1] // 2
        mid_x = density.shape[2] // 2
        
        # XY slice (top view)
        axes[idx, 0].imshow(density[mid_z, :, :], cmap='hot', interpolation='nearest')
        axes[idx, 0].set_title(f'{label} - XY slice (Z={mid_z})')
        axes[idx, 0].axis('off')
        
        # XZ slice (side view)
        axes[idx, 1].imshow(density[:, mid_y, :], cmap='hot', interpolation='nearest')
        axes[idx, 1].set_title(f'{label} - XZ slice (Y={mid_y})')
        axes[idx, 1].axis('off')
        
        # YZ slice (front view)
        axes[idx, 2].imshow(density[:, :, mid_x], cmap='hot', interpolation='nearest')
        axes[idx, 2].set_title(f'{label} - YZ slice (X={mid_x})')
        axes[idx, 2].axis('off')
        
        # Maximum intensity projection (MIP)
        mip = density.max(axis=0)
        axes[idx, 3].imshow(mip, cmap='hot', interpolation='nearest')
        axes[idx, 3].set_title(f'{label} - Max Projection')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('density_map_features_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: density_map_features_visualization.png")
    plt.close()
    
    # Now explain what 3D CNN learns
    print("\n" + "="*70)
    print("WHAT 3D CNN LEARNS FROM DENSITY MAPS")
    print("="*70)
    
    print("""
1. LOCAL GEOMETRY (via 3D convolutions):
   - Backbone curvature and bending
   - Loop closure angles
   - Base stacking geometry (π-π stacking patterns)
   - Helical twist and rise
   
2. SPATIAL PATTERNS (via feature maps):
   - Ribose-phosphate backbone connectivity
   - Base positions relative to backbone
   - Inter-base distances and orientations
   - Sugar pucker conformations (C2'/C3' endo)
   
3. DENSITY DISTRIBUTIONS (via learned filters):
   - Phosphate density (higher Z, brighter)
   - Base density (aromatic rings, medium density)
   - Ribose density (lower, more diffuse)
   - Solvent excluded volume
   
4. TOPOLOGICAL FEATURES (via hierarchical learning):
   - Loop shape: bulge (asymmetric), hairpin (U-turn), internal (symmetric)
   - Opening size: small bulges vs large internal loops
   - Strand orientation: parallel vs antiparallel
   - Base pairing directionality
   
5. STRUCTURAL MOTIFS (high-level features):
   - GNRA tetraloop signature (common hairpin)
   - E-loop motif (internal loop)
   - Kink-turn geometry (bulge)
   - Cross-strand stacking
   
KEY DIFFERENCE FROM PDB FEATURES:
❌ PDB features: Explicit coordinates → trivial size counting
✓ Density features: Implicit geometry → must learn structural patterns

EXAMPLE DISCRIMINATIVE FEATURES:
- Bulge: Asymmetric density protrusion from helix
- Hairpin: U-turn density with tight loop closure
- Internal: Symmetric widening with base stacking disruption
- 2x2 vs 5x5: Different opening angles and strand separation
""")
    
    print("\n" + "="*70)
    print("EXPECTED PERFORMANCE (DENSITY-ONLY)")
    print("="*70)
    print("""
PREDICTION:
- 3-class (bulge/hairpin/internal): 70-85% accuracy
  → Clear topological differences
  
- 14-class (size-specific): 40-65% accuracy  
  → Similar structures, subtle geometric differences
  → 2x2 vs 3x3 internal: harder to distinguish
  → bulge2 vs bulge3: requires precise size estimation
  
WHY HARDER THAN 100%?
1. Density resolution limits (cryo-EM is ~3-4Å)
2. Structural flexibility (same motif, different conformations)
3. Size ambiguity (2x2 vs 3x3 can look similar at low resolution)
4. Class overlap (bulge3 and hairpin3 similar sizes)

THIS IS REALISTIC SCIENTIFIC PERFORMANCE!
""")

if __name__ == '__main__':
    visualize_density_maps()
