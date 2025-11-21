import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
import mrcfile
import torch.nn.functional as F

class RNADensityDataset(Dataset):
    def __init__(self, dataframe, base_path, target_shape=(64,64,64), max_atoms=30):
        self.df = dataframe
        self.base_path = base_path
        self.target_shape = target_shape
        self.max_atoms = max_atoms
        self.label_map = {label: i for i, label in enumerate(sorted(dataframe['label'].unique()))}

    def extract_density(self, filepath):
        with mrcfile.open(filepath, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min())
        data = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        return F.interpolate(data, size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)

    def extract_pdb_features(self, filepath):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('RNA', filepath)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'P' in residue:
                        coords.append(residue['P'].get_coord())
                    if len(coords) >= self.max_atoms:
                        break
        coords = np.array(coords)
        dist_matrix = np.zeros((self.max_atoms, self.max_atoms))
        for i in range(len(coords)):
            for j in range(len(coords)):
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        return torch.tensor(dist_matrix.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mrc_path = f"{self.base_path}/{row['label']}/{row['filename']}.mrc"
        pdb_path = f"{self.base_path}/{row['label']}/{row['filename']}.pdb"

        density = self.extract_density(mrc_path)
        pdb_feat = self.extract_pdb_features(pdb_path)
        label = torch.tensor(self.label_map[row['label']])
        return density, pdb_feat, label

