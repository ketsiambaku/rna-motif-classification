"""Tests for scripts/label_map.py — hierarchy encoding correctness."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.label_map import (
    ALL25_CLASSES,
    ASYMMETRIC_CLASSES,
    CLASS_IDX_TO_L1_IDX,
    CLASS_IDX_TO_L2_IDX,
    CLASS_TO_IDX,
    L1_CLASSES,
    L1_TO_IDX,
    L2_CLASSES,
    L2_TO_IDX,
    L3_CLASSES,
    L3_TO_IDX,
    LABEL_MAP,
    encode,
)


class TestSizes:
    def test_l1_has_3_classes(self):
        assert len(L1_CLASSES) == 3

    def test_l2_has_4_classes(self):
        assert len(L2_CLASSES) == 4

    def test_l3_has_15_classes(self):
        assert len(L3_CLASSES) == 15

    def test_all25_has_25_classes(self):
        assert len(ALL25_CLASSES) == 25

    def test_asymmetric_has_10_classes(self):
        assert len(ASYMMETRIC_CLASSES) == 10

    def test_label_map_covers_all_25(self):
        assert set(LABEL_MAP.keys()) == set(ALL25_CLASSES)


class TestInvariant:
    """For L3-eligible classes: class_idx must equal l3_idx."""

    @pytest.mark.parametrize("cls", L3_CLASSES)
    def test_class_idx_equals_l3_idx_for_eligible(self, cls):
        _, _, l3_idx, class_idx = encode(cls)
        assert class_idx == l3_idx, (
            f"{cls}: class_idx={class_idx} != l3_idx={l3_idx}"
        )

    @pytest.mark.parametrize("cls", ASYMMETRIC_CLASSES)
    def test_asymmetric_l3_is_minus_one(self, cls):
        _, _, l3_idx, _ = encode(cls)
        assert l3_idx == -1, f"{cls} should have l3_idx=-1, got {l3_idx}"

    @pytest.mark.parametrize("cls", ASYMMETRIC_CLASSES)
    def test_asymmetric_class_idx_is_15_or_above(self, cls):
        _, _, _, class_idx = encode(cls)
        assert class_idx >= 15, (
            f"{cls}: class_idx={class_idx} should be ≥15 (in asymmetric range)"
        )


class TestHierarchyConsistency:
    @pytest.mark.parametrize("cls", L3_CLASSES[:5])  # sample hairpins
    def test_hairpins_have_l1_hairpin(self, cls):
        l1_idx, _, _, _ = encode(cls)
        assert L1_CLASSES[l1_idx] == "hairpin"

    @pytest.mark.parametrize("cls", ["1x1", "2x2", "3x3", "4x4", "5x5"])
    def test_symmetric_loops_have_l2_symmetric(self, cls):
        _, l2_idx, _, _ = encode(cls)
        assert L2_CLASSES[l2_idx] == "symmetric"

    @pytest.mark.parametrize("cls", ASYMMETRIC_CLASSES)
    def test_asymmetric_loops_have_l2_asymmetric(self, cls):
        _, l2_idx, _, _ = encode(cls)
        assert L2_CLASSES[l2_idx] == "asymmetric"

    @pytest.mark.parametrize("cls", ["bulge1", "bulge2", "bulge3", "bulge4", "bulge5"])
    def test_bulges_have_l1_bulge(self, cls):
        l1_idx, _, _, _ = encode(cls)
        assert L1_CLASSES[l1_idx] == "bulge"


class TestLookupTables:
    def test_l1_map_length(self):
        assert len(CLASS_IDX_TO_L1_IDX) == 25

    def test_l2_map_length(self):
        assert len(CLASS_IDX_TO_L2_IDX) == 25

    def test_l1_map_values_in_range(self):
        assert all(0 <= v < 3 for v in CLASS_IDX_TO_L1_IDX)

    def test_l2_map_values_in_range(self):
        assert all(0 <= v < 4 for v in CLASS_IDX_TO_L2_IDX)

    def test_unknown_class_raises(self):
        with pytest.raises(KeyError):
            encode("not_a_class")
