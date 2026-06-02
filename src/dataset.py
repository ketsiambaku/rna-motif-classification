from __future__ import annotations
"""
PyTorch Dataset for RNA motif classification from cryo-EM density volumes.

Reads data/manifest.csv, loads MRC volumes on-the-fly, and applies
95th-percentile normalisation (robust to noise spikes in cryo-EM data).

Usage:
    from src.dataset import RNAMotifDataset
    train_ds = RNAMotifDataset(split="train")
    sample = train_ds[0]
    # sample["volume"]   : FloatTensor [1, 64, 64, 64]
    # sample["l1_idx"]   : int  (0-2)
    # sample["l2_idx"]   : int  (0-3)
    # sample["l3_idx"]   : int  (0-14) or -1 for asymmetric loops
    # sample["filepath"] : str

Multi-channel upgrade path (when PDB label maps are available):
    Pass in_channels=4 and provide label_map_dirs to load
    [density, backbone, ribose, base] as a 4-channel volume.
"""
import csv
from pathlib import Path

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"


class RNAMotifDataset(Dataset):
    """
    Args:
        split:       "train", "val", or "test"
        target_size: expected voxel dimensions; volumes are resized if they differ
        in_channels: 1 for density-only (default); extend to 4 with label maps
        manifest:    path to manifest CSV; defaults to data/manifest.csv
    """

    def __init__(
        self,
        split: str,
        target_size: tuple[int, int, int] = (64, 64, 64),
        in_channels: int = 1,
        manifest: Path = DEFAULT_MANIFEST,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"
        self.root = PROJECT_ROOT
        self.target_size = target_size
        self.in_channels = in_channels
        self.split = split

        with open(manifest, newline="") as f:
            all_rows = list(csv.DictReader(f))

        self.rows = [r for r in all_rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest}")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        volume = self._load_volume(self.root / row["filepath"])
        return {
            "volume":    volume,
            "l1_idx":    int(row["l1_idx"]),
            "l2_idx":    int(row["l2_idx"]),
            "l3_idx":    int(row["l3_idx"]),
            "class_idx": int(row["class_idx"]),
            "filepath":  row["filepath"],
        }

    # ------------------------------------------------------------------
    # Loading and normalisation
    # ------------------------------------------------------------------

    def _load_volume(self, path: Path) -> torch.Tensor:
        with mrcfile.open(path, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)

        # Guard against malformed MRC files that are not 3D volumes.
        # Return a zero tensor so the DataLoader worker doesn't crash —
        # these samples contribute a neutral (zero) signal to training.
        if data.ndim != 3:
            return torch.zeros(1, *self.target_size)

        data = _normalize_95p(data)

        # [1, D, H, W]
        volume = torch.from_numpy(data).unsqueeze(0)

        if tuple(volume.shape[1:]) != tuple(self.target_size):
            volume = F.interpolate(
                volume.unsqueeze(0),
                size=self.target_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        return volume

    # ------------------------------------------------------------------
    # Class-weight utility (used to build weighted loss in training)
    # ------------------------------------------------------------------

    @classmethod
    def class_weights(
        cls,
        level: str,
        split: str = "train",
        manifest: Path = DEFAULT_MANIFEST,
    ) -> torch.Tensor:
        """
        Return inverse-sqrt class weights for CrossEntropyLoss.

        Args:
            level:  "l1" (3 classes), "l2" (4 classes), or "l3" (15 classes;
                    asymmetric samples with l3_idx=-1 are excluded from counts)
            split:  which split to count from (default "train")

        Returns:
            FloatTensor of shape [num_classes], normalised to sum to 1.
        """
        assert level in ("l1", "l2", "l3"), f"Unknown level: {level!r}"
        idx_col = f"{level}_idx"
        n_classes = {"l1": 3, "l2": 4, "l3": 15}[level]

        counts = np.zeros(n_classes, dtype=np.float64)
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                idx = int(row[idx_col])
                if idx == -1:
                    continue  # asymmetric loops have no L3 label
                counts[idx] += 1

        counts = np.where(counts == 0, 1, counts)  # avoid division by zero
        weights = 1.0 / np.sqrt(counts)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------------
# Normalisation (module-level so label.py can also import it)
# ------------------------------------------------------------------

def _normalize_95p(data: np.ndarray) -> np.ndarray:
    """Normalise by 95th percentile of non-zero voxels, clip to [0, 1]."""
    nonzero = data[data != 0]
    if nonzero.size == 0:
        return data
    p95 = float(np.percentile(nonzero, 95))
    if p95 > 0:
        data = data / p95
    return np.clip(data, 0.0, 1.0)
