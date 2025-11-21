import mrcfile
import numpy as np
import torch
import torch.nn.functional as F

def load_and_preprocess_mrc(filepath, target_shape=(64, 64, 64)):
    with mrcfile.open(filepath, permissive=True) as mrc:
        data = mrc.data.astype(np.float32)

    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)

    data_resized = F.interpolate(data, size=target_shape, mode='trilinear', align_corners=False)

    return data_resized.squeeze(0)  # shape: [1, D, H, W]

