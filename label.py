import os
import sys
import numpy as np
import mrcfile
from Bio import PDB
from copy import deepcopy

# Normalize an MRC map by its 95th percentile and clips values to [0,1].
def normalize_map(input_mrc_path, output_mrc_path):
    print(f"### Normalizing {input_mrc_path} ###")
    with mrcfile.open(input_mrc_path, mode='r') as clean_map:
        map_data = deepcopy(clean_map.data)

        try:
            percentile = np.percentile(map_data[np.nonzero(map_data)], 95)
            map_data /= percentile
        except IndexError:
            print("Warning: Empty map or invalid data for normalization.")
        
        map_data[map_data < 0] = 0
        map_data[map_data > 1] = 1

        with mrcfile.new(output_mrc_path, overwrite=True) as mrc:
            mrc.set_data(map_data.astype(np.float32))
            mrc.voxel_size = clean_map.voxel_size
            mrc.header.origin = clean_map.header.origin
            mrc.update_header_stats()

    print(f"### Wrote normalized map to {output_mrc_path}")
    return output_mrc_path

#Convert atomic coordinate to MRC voxel indexes since the 3D coordinates of atomic model is in angstroms.
def get_index(coord, origin, voxel_size):
    return round((coord - origin) / voxel_size)


def generate_label_maps(experimental_map, pdb_structure, input_mrc_path):
    error_list = set()

    org_map = mrcfile.open(experimental_map, mode='r')
    shape = org_map.data.shape

    backbone_map = np.zeros(shape, dtype=np.float32)
    ribose_map  = np.zeros(shape, dtype=np.float32)
    base_map    = np.zeros(shape, dtype=np.float32)

    x_origin = org_map.header.origin['x']
    y_origin = org_map.header.origin['y']
    z_origin = org_map.header.origin['z']
    x_voxel = org_map.voxel_size['x']
    y_voxel = org_map.voxel_size['y']
    z_voxel = org_map.voxel_size['z']

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_structure)

    print(f"Labeling map using {pdb_structure}")

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    x, y, z = atom.get_coord()

                    iz = int(get_index(z, z_origin, z_voxel))
                    jy = int(get_index(y, y_origin, y_voxel))
                    kx = int(get_index(x, x_origin, x_voxel))

                    try:
                        atom_name = atom.get_name()

                        if atom_name == "P" or atom_name in ["O5'", "O3'", "O1P", "O2P"]:
                            backbone_map[iz, jy, kx] = 1.0

                        elif atom_name in ["C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O4'"]:
                            ribose_map[iz, jy, kx] = 1.0

                        elif atom_name in [
                            "N1", "N3", "N9", "N2", "N6", "N7",
                            "C2", "C4", "C5", "C6", "C8",
                            "O2", "O4", "O6"
                        ]:
                            base_map[iz, jy, kx] = 1.0

                    except IndexError:
                        error_list.add((iz, jy, kx))

    base_name = os.path.splitext(os.path.basename(input_mrc_path))[0]

    outputs = {
        f"backbone_label_{base_name}.mrc": backbone_map,
        f"ribose_label_{base_name}.mrc": ribose_map,
        f"sugar_label_{base_name}.mrc": base_map
    }

    for fname, vol in outputs.items():
        with mrcfile.new(fname, overwrite=True) as mrc:
            mrc.set_data(vol)
            mrc.voxel_size = org_map.voxel_size
            mrc.header.origin = org_map.header.origin
            mrc.nstart = org_map.nstart
            mrc.update_header_stats()
        print(f"Wrote: {fname}")

    org_map.close()

    if error_list:
        print(f"{len(error_list)} atoms were outside map bounds.")



def main():
    if len(sys.argv) != 3:
        print("Usage: python label.py input.mrc input.pdb")
        sys.exit(1)

    input_mrc, input_pdb = sys.argv[1:3]

    normalized_mrc = os.path.splitext(input_mrc)[0] + "_normalized.mrc"
    normalize_map(input_mrc, normalized_mrc)

    generate_label_maps(normalized_mrc, input_pdb,input_mrc)


if __name__ == "__main__":
    main()