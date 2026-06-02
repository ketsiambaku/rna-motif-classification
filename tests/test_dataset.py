"""Tests for src/dataset.py — normalisation, class weights, and dataset API.

MRC file I/O is mocked so tests run without touching disk.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import RNAMotifDataset, _normalize_95p


# ---------------------------------------------------------------------------
# _normalize_95p unit tests  (pure function, no I/O)
# ---------------------------------------------------------------------------

class TestNormalize95p:
    def test_output_range_clipped_to_0_1(self):
        data = np.array([0.0, 0.5, 1.0, 2.0, 100.0], dtype=np.float32)
        out  = _normalize_95p(data)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_all_zero_returns_unchanged(self):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        out  = _normalize_95p(data)
        np.testing.assert_array_equal(out, data)

    def test_normalises_by_95th_percentile(self):
        # Construct data where p95 of nonzero values is exactly 2.0
        data = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        p95  = float(np.percentile(data[data != 0], 95))
        out  = _normalize_95p(data)
        # Value 2.0 should map to 2.0/p95 (clipped to 1.0 if p95 < 2.0)
        expected = np.clip(data / p95, 0.0, 1.0)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_negative_values_clipped_to_zero(self):
        data = np.array([-5.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out  = _normalize_95p(data)
        assert (out >= 0).all()

    def test_dtype_preserved_as_float32(self):
        data = np.ones((8,), dtype=np.float32)
        out  = _normalize_95p(data)
        assert out.dtype == np.float32

    def test_volume_shape_preserved(self):
        data = np.random.rand(64, 64, 64).astype(np.float32)
        out  = _normalize_95p(data)
        assert out.shape == (64, 64, 64)


# ---------------------------------------------------------------------------
# class_weights unit tests  (reads manifest CSV, no MRC I/O)
# ---------------------------------------------------------------------------

class TestClassWeights:
    def test_l1_weight_tensor_length(self):
        w = RNAMotifDataset.class_weights("l1")
        assert len(w) == 3

    def test_l2_weight_tensor_length(self):
        w = RNAMotifDataset.class_weights("l2")
        assert len(w) == 4

    def test_l3_weight_tensor_length(self):
        w = RNAMotifDataset.class_weights("l3")
        assert len(w) == 15

    def test_weights_sum_to_one(self):
        for level in ("l1", "l2", "l3"):
            w = RNAMotifDataset.class_weights(level)
            assert abs(w.sum().item() - 1.0) < 1e-5, \
                f"L{level[-1]} weights sum to {w.sum().item():.6f}, expected 1.0"

    def test_weights_all_positive(self):
        for level in ("l1", "l2", "l3"):
            w = RNAMotifDataset.class_weights(level)
            assert (w > 0).all(), f"Some {level} weights are non-positive"

    def test_weights_are_float_tensor(self):
        w = RNAMotifDataset.class_weights("l3")
        assert w.dtype == torch.float32

    def test_invalid_level_raises(self):
        with pytest.raises(AssertionError):
            RNAMotifDataset.class_weights("l4")


# ---------------------------------------------------------------------------
# Dataset API tests  (MRC I/O mocked)
# ---------------------------------------------------------------------------

def _make_fake_mrc():
    """Return a mock mrcfile context manager yielding a 64³ float32 array."""
    mock_mrc  = MagicMock()
    mock_mrc.__enter__ = lambda s: s
    mock_mrc.__exit__  = MagicMock(return_value=False)
    mock_mrc.data      = np.random.rand(64, 64, 64).astype(np.float32)
    return mock_mrc


class TestDatasetAPI:
    @pytest.fixture(autouse=True)
    def patch_mrcfile(self):
        with patch("src.dataset.mrcfile.open", return_value=_make_fake_mrc()):
            yield

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_split_loads_nonzero_rows(self, split):
        ds = RNAMotifDataset(split=split)
        assert len(ds) > 0

    def test_getitem_returns_correct_keys(self):
        ds     = RNAMotifDataset(split="val")
        sample = ds[0]
        assert set(sample.keys()) == {"volume", "l1_idx", "l2_idx", "l3_idx", "class_idx", "filepath"}

    def test_volume_shape(self):
        ds     = RNAMotifDataset(split="val")
        sample = ds[0]
        assert sample["volume"].shape == torch.Size([1, 64, 64, 64])

    def test_volume_dtype(self):
        ds     = RNAMotifDataset(split="val")
        sample = ds[0]
        assert sample["volume"].dtype == torch.float32

    def test_volume_range(self):
        ds     = RNAMotifDataset(split="val")
        sample = ds[0]
        assert sample["volume"].min().item() >= 0.0
        assert sample["volume"].max().item() <= 1.0

    def test_l1_idx_in_valid_range(self):
        ds = RNAMotifDataset(split="val")
        for row in ds.rows[:20]:
            assert int(row["l1_idx"]) in {0, 1, 2}

    def test_l3_idx_minus_one_for_asymmetric(self):
        ds = RNAMotifDataset(split="train")
        asymmetric_rows = [r for r in ds.rows if r["l2"] == "asymmetric"]
        assert len(asymmetric_rows) > 0
        for r in asymmetric_rows[:10]:
            assert int(r["l3_idx"]) == -1

    def test_invalid_split_raises(self):
        with pytest.raises(AssertionError):
            RNAMotifDataset(split="unknown")
