"""Tests for model architectures.

Covers output shapes, return types, gradient flow, multi-channel upgrade,
and a one-batch overfit sanity check for each model as it is implemented.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.flat_cnn import CNN3DBaseline
from models.resnet3d import ResNet3DBaseline
from models.swin3d import SwinTransformer3D
from src.train import FlatLoss, HierarchicalLoss


# ---------------------------------------------------------------------------
# Helpers shared across model tests
# ---------------------------------------------------------------------------

def _fake_batch(B: int = 4, in_channels: int = 1, size: int = 64) -> torch.Tensor:
    return torch.randn(B, in_channels, size, size, size)


def _fake_flat_batch(B: int = 4) -> dict:
    return {"class_idx": torch.randint(0, 25, (B,))}


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CNN3DBaseline (M1)
# ---------------------------------------------------------------------------

class TestCNN3DBaseline:

    @pytest.fixture
    def model(self):
        return CNN3DBaseline()

    def test_output_shape_default(self, model):
        x   = _fake_batch()
        out = model(x)
        assert out.shape == torch.Size([4, 25])

    def test_output_is_tensor_not_dict(self, model):
        """train.py auto-selects FlatLoss when output is a plain Tensor."""
        out = model(_fake_batch())
        assert isinstance(out, torch.Tensor)
        assert not isinstance(out, dict)

    def test_output_dtype_is_float32(self, model):
        out = model(_fake_batch())
        assert out.dtype == torch.float32

    def test_custom_num_classes(self):
        m   = CNN3DBaseline(num_classes=15)
        out = m(_fake_batch())
        assert out.shape == torch.Size([4, 15])

    def test_in_channels_4(self):
        """Model should accept 4-channel input without any other changes."""
        m   = CNN3DBaseline(in_channels=4)
        out = m(_fake_batch(in_channels=4))
        assert out.shape == torch.Size([4, 25])

    def test_single_sample(self, model):
        out = model(_fake_batch(B=1))
        assert out.shape == torch.Size([1, 25])

    def test_gradients_flow_to_all_parameters(self, model):
        out  = model(_fake_batch())
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_train_eval_modes(self, model):
        """BatchNorm behaves differently in train vs eval — both should run."""
        model.train()
        model(_fake_batch())
        model.eval()
        with torch.no_grad():
            model(_fake_batch())

    def test_has_name_attribute(self, model):
        assert hasattr(model, "name")
        assert isinstance(model.name, str)

    def test_parameter_count_reasonable(self, model):
        """Rough sanity check — M1 should be between 1M and 20M params."""
        n = _count_parameters(model)
        assert 1_000_000 < n < 20_000_000, f"Unexpected param count: {n:,}"

    def test_no_nan_in_output(self, model):
        out = model(_fake_batch())
        assert not torch.isnan(out).any()

    def test_no_nan_after_backward(self, model):
        out  = model(_fake_batch())
        loss = out.mean()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_overfit_one_batch(self):
        """
        The model should be able to memorise a tiny batch.
        Loss must decrease substantially (>50%) within 30 gradient steps.
        This catches capacity issues and broken forward/backward passes.
        """
        torch.manual_seed(0)
        model     = CNN3DBaseline(dropout=0.0)   # disable dropout for overfitting
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn   = FlatLoss("cpu")

        # Small batch: 4 samples, fixed targets
        x      = torch.randn(4, 1, 64, 64, 64)
        batch  = {"class_idx": torch.tensor([0, 1, 2, 3])}

        losses = []
        for _ in range(30):
            out  = model(x)
            loss = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        reduction = (losses[0] - losses[-1]) / losses[0]
        assert reduction > 0.50, (
            f"M1 loss did not decrease >50% in 30 steps "
            f"(initial={losses[0]:.4f}, final={losses[-1]:.4f}, "
            f"reduction={reduction:.1%})"
        )


# ---------------------------------------------------------------------------
# ResNet3DBaseline (M2)
# ---------------------------------------------------------------------------

def _fake_hier_batch(B: int = 4) -> dict:
    return {
        "l1_idx":  torch.randint(0,  3, (B,)),
        "l2_idx":  torch.randint(0,  4, (B,)),
        "l3_idx":  torch.randint(0, 15, (B,)),   # all L3-eligible for simplicity
        "class_idx": torch.randint(0, 25, (B,)),
    }


class TestResNet3DBaseline:

    @pytest.fixture
    def model(self):
        return ResNet3DBaseline()

    def test_output_is_dict(self, model):
        """train.py auto-selects HierarchicalLoss when output is a dict."""
        out = model(_fake_batch())
        assert isinstance(out, dict)

    def test_output_keys(self, model):
        out = model(_fake_batch())
        assert set(out.keys()) == {"l1", "l2", "l3"}

    def test_l1_shape(self, model):
        out = model(_fake_batch())
        assert out["l1"].shape == torch.Size([4, 3])

    def test_l2_shape(self, model):
        out = model(_fake_batch())
        assert out["l2"].shape == torch.Size([4, 4])

    def test_l3_shape(self, model):
        """L3 head outputs 15 classes (L3-eligible only, not 25)."""
        out = model(_fake_batch())
        assert out["l3"].shape == torch.Size([4, 15])

    def test_in_channels_4(self):
        m   = ResNet3DBaseline(in_channels=4)
        out = m(_fake_batch(in_channels=4))
        assert out["l3"].shape == torch.Size([4, 15])

    def test_output_dtype(self, model):
        out = model(_fake_batch())
        assert all(v.dtype == torch.float32 for v in out.values())

    def test_gradients_flow_to_all_parameters(self, model):
        out  = model(_fake_batch())
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_in_output(self, model):
        out = model(_fake_batch())
        assert not any(torch.isnan(v).any() for v in out.values())

    def test_has_name_attribute(self, model):
        assert model.name == "resnet3d"

    def test_parameter_count_reasonable(self, model):
        n = _count_parameters(model)
        assert 1_000_000 < n < 50_000_000, f"Unexpected param count: {n:,}"

    def test_train_eval_modes(self, model):
        model.train()
        model(_fake_batch())
        model.eval()
        with torch.no_grad():
            model(_fake_batch())

    def test_asymmetric_samples_handled(self, model):
        """Loss must not crash when some l3_idx values are -1."""
        loss_fn = HierarchicalLoss("cpu")
        batch   = _fake_hier_batch()
        batch["l3_idx"][0] = -1   # inject an asymmetric sample
        out  = model(_fake_batch())
        loss = loss_fn(out, batch)
        assert loss.item() > 0

    def test_overfit_one_batch(self):
        """Loss must decrease >50% in 30 gradient steps on a fixed batch."""
        torch.manual_seed(0)
        model     = ResNet3DBaseline(dropout=0.0)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn   = HierarchicalLoss("cpu")

        x     = torch.randn(4, 1, 64, 64, 64)
        batch = {
            "l1_idx":    torch.tensor([0, 1, 2, 0]),
            "l2_idx":    torch.tensor([0, 1, 2, 3]),
            "l3_idx":    torch.tensor([0, 5, 10, 14]),
            "class_idx": torch.tensor([0, 5, 10, 14]),
        }

        losses = []
        for _ in range(30):
            out  = model(x)
            loss = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        reduction = (losses[0] - losses[-1]) / losses[0]
        assert reduction > 0.50, (
            f"M2 loss did not decrease >50% in 30 steps "
            f"(initial={losses[0]:.4f}, final={losses[-1]:.4f}, "
            f"reduction={reduction:.1%})"
        )


# ---------------------------------------------------------------------------
# SwinTransformer3D (M3) — small config to keep tests fast on CPU
# ---------------------------------------------------------------------------

def _small_swin() -> SwinTransformer3D:
    """Tiny Swin config: fewer blocks/heads so tests run in < 60s on CPU."""
    return SwinTransformer3D(depths=[1, 1, 2, 1], num_heads=[2, 4, 8, 16])


class TestSwinTransformer3D:

    @pytest.fixture
    def model(self):
        return _small_swin()

    def test_output_is_dict(self, model):
        out = model(_fake_batch())
        assert isinstance(out, dict)

    def test_output_keys(self, model):
        out = model(_fake_batch())
        assert set(out.keys()) == {"l1", "l2", "l3"}

    def test_l1_shape(self, model):
        assert model(_fake_batch())["l1"].shape == torch.Size([4, 3])

    def test_l2_shape(self, model):
        assert model(_fake_batch())["l2"].shape == torch.Size([4, 4])

    def test_l3_shape(self, model):
        assert model(_fake_batch())["l3"].shape == torch.Size([4, 15])

    def test_in_channels_4(self):
        m   = SwinTransformer3D(in_channels=4, depths=[1, 1, 2, 1], num_heads=[2, 4, 8, 16])
        out = m(_fake_batch(in_channels=4))
        assert out["l3"].shape == torch.Size([4, 15])

    def test_output_dtype(self, model):
        out = model(_fake_batch())
        assert all(v.dtype == torch.float32 for v in out.values())

    def test_no_nan_in_output(self, model):
        out = model(_fake_batch())
        assert not any(torch.isnan(v).any() for v in out.values())

    def test_gradients_flow_to_all_parameters(self, model):
        out  = model(_fake_batch())
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_has_name_attribute(self, model):
        assert model.name == "swin3d"

    def test_parameter_count_reasonable(self, model):
        n = _count_parameters(model)
        assert 100_000 < n < 200_000_000, f"Unexpected param count: {n:,}"

    def test_train_eval_modes(self, model):
        model.train()
        model(_fake_batch())
        model.eval()
        with torch.no_grad():
            model(_fake_batch())

    def test_decode_constrains_l3_logits(self, model):
        """decode() must zero out impossible L3 classes given the L2 prediction."""
        model.eval()
        with torch.no_grad():
            out     = model(_fake_batch(B=1))
            decoded = model.decode(out)

        l2_pred = out["l2"].argmax(1).item()
        l3_log  = decoded["l3"][0]
        L2_TO_L3 = {0: (0, 5), 1: (5, 10), 2: None, 3: (10, 15)}
        valid     = L2_TO_L3[l2_pred]

        if valid is None:
            assert (l3_log == float("-inf")).all()
        else:
            lo, hi = valid
            assert not (l3_log[lo:hi] == float("-inf")).any()
            outside = list(range(0, lo)) + list(range(hi, 15))
            if outside:
                assert (l3_log[outside] == float("-inf")).all()

    def test_overfit_one_batch(self):
        """Loss must decrease >40% in 30 gradient steps."""
        torch.manual_seed(0)
        model     = SwinTransformer3D(depths=[1, 1, 2, 1], num_heads=[2, 4, 8, 16], dropout=0.0)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn   = HierarchicalLoss("cpu")

        x     = torch.randn(2, 1, 64, 64, 64)
        batch = {
            "l1_idx":    torch.tensor([0, 2]),
            "l2_idx":    torch.tensor([0, 3]),
            "l3_idx":    torch.tensor([2, 12]),
            "class_idx": torch.tensor([2, 12]),
        }

        losses = []
        for _ in range(30):
            out  = model(x)
            loss = loss_fn(out, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        reduction = (losses[0] - losses[-1]) / losses[0]
        assert reduction > 0.40, (
            f"M3 loss did not decrease >40% in 30 steps "
            f"(initial={losses[0]:.4f}, final={losses[-1]:.4f}, "
            f"reduction={reduction:.1%})"
        )
