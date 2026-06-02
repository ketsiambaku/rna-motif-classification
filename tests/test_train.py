"""Tests for src/train.py — loss functions, early stopping, and training mechanics."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import EarlyStopping, FlatLoss, HierarchicalLoss, TrainConfig


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_first_value_is_always_best(self):
        es = EarlyStopping(patience=5)
        assert es.step(0.5) is True

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.step(0.50)
        es.step(0.40)   # worse — counter=1
        es.step(0.60)   # improvement — counter resets
        assert not es.should_stop

    def test_stops_after_patience_exhausted(self):
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.step(0.50)
        es.step(0.50)  # counter=1
        es.step(0.50)  # counter=2
        es.step(0.50)  # counter=3 → should_stop
        assert es.should_stop

    def test_does_not_stop_before_patience(self):
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.step(0.50)
        for _ in range(4):   # 4 non-improving steps, patience=5
            es.step(0.50)
        assert not es.should_stop

    def test_min_delta_boundary(self):
        es = EarlyStopping(patience=2, min_delta=0.10)
        es.step(0.50)          # best=0.50
        assert es.step(0.59)  is False  # 0.59 <= 0.50+0.10 → no improvement
        assert es.step(0.61)  is True   # 0.61 >  0.50+0.10 → improvement


# ---------------------------------------------------------------------------
# FlatLoss
# ---------------------------------------------------------------------------

class TestFlatLoss:
    @pytest.fixture
    def loss_fn(self):
        return FlatLoss("cpu")

    def test_returns_scalar(self, loss_fn):
        logits = torch.randn(4, 25)
        batch  = {"class_idx": torch.randint(0, 25, (4,))}
        loss   = loss_fn(logits, batch)
        assert loss.shape == torch.Size([])

    def test_loss_is_positive(self, loss_fn):
        logits = torch.randn(4, 25)
        batch  = {"class_idx": torch.randint(0, 25, (4,))}
        loss   = loss_fn(logits, batch)
        assert loss.item() > 0

    def test_loss_is_differentiable(self, loss_fn):
        logits = torch.randn(4, 25, requires_grad=True)
        batch  = {"class_idx": torch.randint(0, 25, (4,))}
        loss   = loss_fn(logits, batch)
        loss.backward()
        assert logits.grad is not None

    def test_perfect_prediction_lower_than_random(self, loss_fn):
        # Loss on perfect logits should be lower than on random logits
        targets = torch.tensor([0, 1, 2, 3])
        perfect = torch.zeros(4, 25)
        perfect[range(4), targets] = 100.0
        random  = torch.randn(4, 25)
        batch   = {"class_idx": targets}
        assert loss_fn(perfect, batch).item() < loss_fn(random, batch).item()


# ---------------------------------------------------------------------------
# HierarchicalLoss
# ---------------------------------------------------------------------------

class TestHierarchicalLoss:
    @pytest.fixture
    def loss_fn(self):
        return HierarchicalLoss("cpu", alpha=0.25, beta=0.25, gamma=0.50)

    def _make_batch(self, B=4, include_asymmetric=False):
        l3_idx = torch.randint(0, 15, (B,))
        if include_asymmetric:
            l3_idx[0] = -1  # simulate an asymmetric sample
        return {
            "l1_idx": torch.randint(0, 3, (B,)),
            "l2_idx": torch.randint(0, 4, (B,)),
            "l3_idx": l3_idx,
        }

    def _make_output(self, B=4):
        return {
            "l1": torch.randn(B, 3),
            "l2": torch.randn(B, 4),
            "l3": torch.randn(B, 15),
        }

    def test_returns_scalar(self, loss_fn):
        loss = loss_fn(self._make_output(), self._make_batch())
        assert loss.shape == torch.Size([])

    def test_loss_is_positive(self, loss_fn):
        loss = loss_fn(self._make_output(), self._make_batch())
        assert loss.item() > 0

    def test_loss_is_differentiable(self, loss_fn):
        output = {k: v.requires_grad_(True) for k, v in self._make_output().items()}
        loss   = loss_fn(output, self._make_batch())
        loss.backward()
        assert all(v.grad is not None for v in output.values())

    def test_asymmetric_samples_dont_break_l3(self, loss_fn):
        """l3_idx=-1 samples should be silently ignored by the L3 loss."""
        output = self._make_output()
        batch  = self._make_batch(include_asymmetric=True)
        loss   = loss_fn(output, batch)  # must not raise
        assert loss.item() > 0

    def test_weights_sum_to_one(self, loss_fn):
        """α + β + γ = 1.0 for the default config."""
        total = loss_fn.alpha + loss_fn.beta + loss_fn.gamma
        assert abs(total - 1.0) < 1e-6

    def test_cross_family_penalised_more_than_within_family(self, loss_fn):
        """
        Confusing bulge1 (l3=10) with bulge2 (l3=11) should produce lower
        total loss than confusing bulge1 (l3=10) with hairpin3 (l3=0),
        because L1/L2 losses are zero for within-family errors.
        """
        # True label: bulge1 (l1=bulge=2, l2=bulge=3, l3=10)
        batch = {
            "l1_idx": torch.tensor([2]),
            "l2_idx": torch.tensor([3]),
            "l3_idx": torch.tensor([10]),
        }

        # Within-family error: model predicts bulge2 (l3=11) — correct L1/L2
        l3_within = torch.zeros(1, 15); l3_within[0, 11] = 100.0
        out_within = {
            "l1": torch.tensor([[0.0, 0.0, 100.0]]),
            "l2": torch.tensor([[0.0, 0.0, 0.0, 100.0]]),
            "l3": l3_within,
        }

        # Cross-family error: model predicts hairpin3 (l3=0) — wrong L1/L2
        l3_cross = torch.zeros(1, 15); l3_cross[0, 0] = 100.0
        out_cross = {
            "l1": torch.tensor([[100.0, 0.0, 0.0]]),
            "l2": torch.tensor([[100.0, 0.0, 0.0, 0.0]]),
            "l3": l3_cross,
        }

        loss_within = loss_fn(out_within, batch).item()
        loss_cross  = loss_fn(out_cross,  batch).item()
        assert loss_within < loss_cross, (
            f"Within-family loss ({loss_within:.4f}) should be < "
            f"cross-family loss ({loss_cross:.4f})"
        )


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_default_device_is_valid(self):
        cfg = TrainConfig()
        assert cfg.device in ("cpu", "cuda", "mps")

    def test_custom_values_set(self):
        cfg = TrainConfig(model_name="test", epochs=5, batch_size=8, lr=1e-4)
        assert cfg.epochs     == 5
        assert cfg.batch_size == 8
        assert cfg.lr         == 1e-4

    def test_hierarchical_weights_default_sum_to_one(self):
        cfg = TrainConfig()
        assert abs(cfg.alpha + cfg.beta + cfg.gamma - 1.0) < 1e-6
