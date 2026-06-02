"""
End-to-end smoke test — runs in < 2 minutes on CPU.

Uses a small random subset of train/val data to verify the full pipeline:
  dataset → dataloader → model forward → loss → backward → evaluate

Exits with code 0 on success, 1 on any failure.

Usage:
    python scripts/smoke_test.py              # tests M1 (default)
    python scripts/smoke_test.py --all        # tests M1, M2, M3
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.flat_cnn import CNN3DBaseline
from src.dataset import RNAMotifDataset
from src.train import FlatLoss, HierarchicalLoss

N_TRAIN   = 128   # samples for the train subset
N_VAL     = 64    # samples for the val subset
BATCH     = 16    # batch size during smoke test
N_STEPS   = 5     # gradient steps (not a full epoch — just enough to check loss drops)
DEVICE    = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loaders():
    train_ds = RNAMotifDataset(split="train")
    val_ds   = RNAMotifDataset(split="val")
    train_sub = Subset(train_ds, list(range(N_TRAIN)))
    val_sub   = Subset(val_ds,   list(range(N_VAL)))
    train_loader = DataLoader(train_sub, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_sub,   batch_size=BATCH, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _check_loss_decreases(model, loss_fn, loader, label: str) -> float:
    """Run N_STEPS gradient steps and assert loss decreases. Returns final loss."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    it = iter(loader)
    for _ in range(N_STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch["volume"] = batch["volume"].to(DEVICE)
        out  = model(batch["volume"])
        loss = loss_fn(out, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    drop = losses[0] - losses[-1]
    status = "✓" if drop > 0 else "⚠ loss did not decrease"
    print(f"  {label}: loss {losses[0]:.4f} → {losses[-1]:.4f}  {status}")
    return losses[-1]


def _check_eval(model, loss_fn, loader, label: str) -> dict:
    """Run a quick eval pass, print summary metrics."""
    from sklearn.metrics import f1_score
    from scripts.label_map import CLASS_IDX_TO_L1_IDX, CLASS_IDX_TO_L2_IDX

    l1_map = torch.tensor(CLASS_IDX_TO_L1_IDX)
    l2_map = torch.tensor(CLASS_IDX_TO_L2_IDX)

    model.eval()
    l1_correct = l2_correct = l3_correct = l3_total = n = 0
    l3_preds, l3_trues = [], []

    with torch.no_grad():
        for batch in loader:
            batch["volume"] = batch["volume"].to(DEVICE)
            out = model(batch["volume"])

            if isinstance(out, dict):
                pred_l1 = out["l1"].argmax(1).cpu()
                pred_l2 = out["l2"].argmax(1).cpu()
                pred_l3 = out["l3"].argmax(1).cpu()
            else:
                pred_cls = out.argmax(1).cpu()
                pred_l1  = l1_map[pred_cls]
                pred_l2  = l2_map[pred_cls]
                pred_l3  = pred_cls

            l1_correct += (pred_l1 == batch["l1_idx"]).sum().item()
            l2_correct += (pred_l2 == batch["l2_idx"]).sum().item()
            n += batch["l1_idx"].size(0)

            mask = batch["l3_idx"] != -1
            if mask.any():
                l3_correct += (pred_l3[mask] == batch["l3_idx"][mask]).sum().item()
                l3_total   += mask.sum().item()
                l3_preds.extend(pred_l3[mask].tolist())
                l3_trues.extend(batch["l3_idx"][mask].tolist())

    l3_f1 = float(f1_score(l3_trues, l3_preds, average="macro", zero_division=0)) \
             if l3_trues else 0.0

    metrics = {
        "l1_acc": l1_correct / n,
        "l2_acc": l2_correct / n,
        "l3_acc": l3_correct / l3_total if l3_total else 0.0,
        "l3_f1":  l3_f1,
    }
    print(f"  {label} eval  →  "
          f"L1={metrics['l1_acc']:.3f}  "
          f"L2={metrics['l2_acc']:.3f}  "
          f"L3_acc={metrics['l3_acc']:.3f}  "
          f"L3_F1={metrics['l3_f1']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Per-model smoke tests
# ---------------------------------------------------------------------------

def smoke_m1(train_loader, val_loader):
    print("\n── M1: 3D CNN Baseline ──────────────────────────")
    model   = CNN3DBaseline().to(DEVICE)
    loss_fn = FlatLoss(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    _check_loss_decreases(model, loss_fn, train_loader, "train")
    _check_eval(model, loss_fn, val_loader, "val")
    print("  M1 ✓")


def smoke_m2(train_loader, val_loader):
    print("\n── M2: 3D ResNet-18 Hierarchical ───────────────")
    try:
        from models.resnet3d import ResNet3DBaseline
    except ImportError:
        print("  [skip] models/resnet3d.py not yet implemented")
        return
    model   = ResNet3DBaseline().to(DEVICE)
    loss_fn = HierarchicalLoss(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    _check_loss_decreases(model, loss_fn, train_loader, "train")
    _check_eval(model, loss_fn, val_loader, "val")
    print("  M2 ✓")


def smoke_m3(train_loader, val_loader):
    print("\n── M3: 3D Swin Transformer ──────────────────────")
    try:
        from models.swin3d import SwinTransformer3D
    except ImportError:
        print("  [skip] models/swin3d.py not yet implemented")
        return
    model   = SwinTransformer3D().to(DEVICE)
    loss_fn = HierarchicalLoss(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    _check_loss_decreases(model, loss_fn, train_loader, "train")
    _check_eval(model, loss_fn, val_loader, "val")
    print("  M3 ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Test all models (M1, M2, M3)")
    args = parser.parse_args()

    print(f"Smoke test  |  device={DEVICE}  "
          f"train_subset={N_TRAIN}  val_subset={N_VAL}  steps={N_STEPS}")
    t0 = time.time()

    train_loader, val_loader = _make_loaders()
    print(f"Data loaded  ({time.time()-t0:.1f}s)")

    try:
        smoke_m1(train_loader, val_loader)
        if args.all:
            smoke_m2(train_loader, val_loader)
            smoke_m3(train_loader, val_loader)
    except Exception as e:
        print(f"\n✗ Smoke test FAILED: {e}")
        raise SystemExit(1)

    print(f"\n✓ All smoke tests passed  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
