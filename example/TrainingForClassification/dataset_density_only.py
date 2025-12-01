import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import mrcfile
import torch.nn.functional as F
from pathlib import Path

class DensityOnlyDataset(Dataset):
    """Dataset that returns ONLY density maps (no PDB features)"""
    def __init__(self, dataframe, base_path, target_shape=(64,64,64)):
        self.df = dataframe
        self.base_path = base_path
        self.target_shape = target_shape
        self.label_map = {label: i for i, label in enumerate(sorted(dataframe['label'].unique()))}
    
    def __len__(self):
        return len(self.df)
    
    def extract_density(self, filepath):
        """Extract and normalize density map"""
        with mrcfile.open(filepath, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
        
        # Check if data is 3D
        if data.ndim != 3:
            return torch.zeros((1, *self.target_shape), dtype=torch.float32)
        
        # Normalize
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Convert to tensor and resize
        data = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        data = F.interpolate(data, size=self.target_shape, mode='trilinear', align_corners=False)
        return data.squeeze(0)  # [1,D,H,W]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Parse filename to get pdb_id and number
        filename = row['filename']  # e.g., "8BSJ_3269"
        
        # Load density map
        mrc_file = f"{filename}.mrc"
        mrc_path = Path(self.base_path) / row['label'] / mrc_file
        density = self.extract_density(mrc_path)
        
        # Get label
        label = self.label_map[row['label']]
        
        return density, label

if __name__ == '__main__':
    # Test the dataset
    import pandas as pd
    df = pd.read_csv('rna_train_dataset2_small.csv')
    dataset = DensityOnlyDataset(df, '../../dataset2')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {sorted(dataset.label_map.keys())}")
    
    # Test loading
    density, label = dataset[0]
    print(f"Density shape: {density.shape}")
    print(f"Label: {label}")
