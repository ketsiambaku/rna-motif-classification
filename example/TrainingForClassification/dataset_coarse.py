import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import mrcfile
import torch.nn.functional as F
from pathlib import Path

class CoarseDataset(Dataset):
    """Dataset that maps coarse labels back to original fine-grained folders"""
    def __init__(self, dataframe, fine_grained_df, base_path, target_shape=(64,64,64)):
        self.df = dataframe
        self.fine_df = fine_grained_df  # Original with fine-grained labels
        self.base_path = base_path
        self.target_shape = target_shape
        self.label_map = {label: i for i, label in enumerate(sorted(dataframe['label'].unique()))}
        
        # Create filename to fine-grained label mapping
        self.filename_to_fine = dict(zip(fine_grained_df['filename'], fine_grained_df['label']))
    
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
        filename = row['filename']
        coarse_label = row['label']
        
        # Get fine-grained label to find correct folder
        fine_label = self.filename_to_fine[filename]
        
        # Load density map from fine-grained folder
        mrc_file = f"{filename}.mrc"
        mrc_path = Path(self.base_path) / fine_label / mrc_file
        density = self.extract_density(mrc_path)
        
        # Map coarse label to index
        label = self.label_map[coarse_label]
        
        return density, label
