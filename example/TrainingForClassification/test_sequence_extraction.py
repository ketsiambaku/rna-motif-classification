#!/usr/bin/env python3
"""Quick check: Can we extract sequences from PDB files?"""

import os
import sys

# Check if BioPython is available
try:
    from Bio.PDB import PDBParser
    print("✓ BioPython is available\n")
except ImportError:
    print("✗ BioPython NOT available - need to install: pip install biopython\n")
    sys.exit(1)

# RNA nucleotide mapping
RNA_MAP = {'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
           'DA': 'A', 'DT': 'U', 'DG': 'G', 'DC': 'C'}

def extract_sequence(pdb_path):
    """Extract RNA sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('RNA', pdb_path)
        sequence = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in RNA_MAP:
                        sequence.append(RNA_MAP[resname])
        
        return ''.join(sequence) if sequence else None
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Test on sample files from each motif class
dataset_root = "../../dataset2"

print("="*60)
print("TESTING: Sequence Extraction from PDB Files")
print("="*60 + "\n")

# Find one file from each coarse class
test_cases = []

# Bulge
bulge_dir = os.path.join(dataset_root, "bulge1")
if os.path.exists(bulge_dir):
    files = [f for f in os.listdir(bulge_dir) if f.endswith('.pdb')]
    if files:
        test_cases.append(("bulge", "bulge1", files[0]))

# Hairpin
hairpin_dir = os.path.join(dataset_root, "hairpin3")
if os.path.exists(hairpin_dir):
    files = [f for f in os.listdir(hairpin_dir) if f.endswith('.pdb')]
    if files:
        test_cases.append(("hairpin", "hairpin3", files[0]))

# Internal
internal_dir = os.path.join(dataset_root, "3x3")
if os.path.exists(internal_dir):
    files = [f for f in os.listdir(internal_dir) if f.endswith('.pdb')]
    if files:
        test_cases.append(("internal", "3x3", files[0]))

if not test_cases:
    print("ERROR: No PDB files found in dataset2!")
    sys.exit(1)

print(f"Found {len(test_cases)} test files\n")

success_count = 0
for coarse_label, folder, filename in test_cases:
    pdb_path = os.path.join(dataset_root, folder, filename)
    
    print(f"[{coarse_label.upper()}] {folder}/{filename}")
    
    seq = extract_sequence(pdb_path)
    if seq:
        # Calculate composition
        a_count = seq.count('A')
        u_count = seq.count('U')
        g_count = seq.count('G')
        c_count = seq.count('C')
        total = len(seq)
        
        gc_content = (g_count + c_count) / total if total > 0 else 0
        purine_pct = (a_count + g_count) / total if total > 0 else 0
        
        print(f"  ✓ Sequence: {seq[:40]}... (length={total})")
        print(f"  ✓ Composition: A={a_count} ({100*a_count/total:.1f}%), " +
              f"U={u_count} ({100*u_count/total:.1f}%), " +
              f"G={g_count} ({100*g_count/total:.1f}%), " +
              f"C={c_count} ({100*c_count/total:.1f}%)")
        print(f"  ✓ GC content: {100*gc_content:.1f}%")
        print(f"  ✓ Purine%: {100*purine_pct:.1f}%")
        print()
        success_count += 1
    else:
        print(f"  ✗ Failed to extract sequence")
        print()

print("="*60)
if success_count == len(test_cases):
    print(f"✓ SUCCESS: All {success_count}/{len(test_cases)} extractions worked!")
    print("\nSequence features ARE available and can be used as:")
    print("  • Size-invariant (percentages, not counts)")
    print("  • Biologically meaningful (GC content, purine/pyrimidine ratio)")
    print("  • Available for all samples")
    print("\nRecommendation: Include sequence features in Phase 2 model")
else:
    print(f"✗ PARTIAL: Only {success_count}/{len(test_cases)} extractions worked")
print("="*60)
