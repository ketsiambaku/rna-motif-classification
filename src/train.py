from __future__ import annotations
"""
Generic training loop for RNA motif classification models.

Works with all three architectures:
  M1 — flat_cnn     : model(x) → Tensor[B, 25]          (FlatLoss)
  M2 — resnet3d     : model(x) → dict{l1, l2, l3}       (HierarchicalLoss)
  M3 — swin3d       : model(x) → dict{l1, l2, l3}       (HierarchicalLoss)

Usage:
    from src.train import train, TrainConfig
    from models.flat_cnn import FlatCNN

    cfg = TrainConfig(model_name="flat_cnn", epochs=50, batch_size=32)
    model = FlatCNN(in_channels=1)
    train(model, cfg)

Checkpoints are saved to:
    checkpoints/<model_name>/best.pt          ← best val macro-F1
    checkpoints/<model_name>/epoch_<N>.pt     ← every save_every epochs

Training log (one row per epoch) is written to:
    checkpoints/<model_name>/log.csv
"""
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.label_map import (
    CLASS_IDX_TO_L1_IDX,
    CLASS_IDX_TO_L2_IDX,
)
from src.dataset import RNAMotifDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters and paths for one training run.

    Attributes:
        model_name:    Identifies the run; used for checkpoint directory naming.
        epochs:        Maximum number of training epochs.
        batch_size:    Samples per batch.
        lr:            Initial learning rate (Adam).
        weight_decay:  L2 regularisation coefficient.
        alpha:         L1 loss weight (hierarchical models only).
        beta:          L2 loss weight (hierarchical models only).
        gamma:         L3 loss weight (hierarchical models only).
        patience:      Early-stopping patience (epochs without improvement).
        min_delta:     Minimum improvement in val macro-F1 to reset patience.
        save_every:    Save a periodic checkpoint every N epochs (0 = disable).
        checkpoint_dir: Root directory for all checkpoints.
        device:        Training device; auto-detected if not specified.
    """
    model_name:     str   = "model"
    epochs:         int   = 100
    batch_size:     int   = 32
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    # Hierarchical loss weights (M2 / M3 only)
    alpha:          float = 0.25   # L1 topology head
    beta:           float = 0.25   # L2 symmetry head
    gamma:          float = 0.50   # L3 fine-grained head
    # Early stopping
    patience:       int   = 15
    min_delta:      float = 1e-4
    # Checkpointing
    save_every:     int   = 5
    checkpoint_dir: Path  = field(default_factory=lambda: PROJECT_ROOT / "checkpoints")
    # Device (auto-detected)
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FlatLoss:
    """Cross-entropy loss for M1 (flat 25-class output).

    Target is class_idx (0–24). Class weights are derived from training
    set counts using inverse weighting.
    """

    def __init__(self, device: str) -> None:
        weights = RNAMotifDataset.class_weights("l3", split="train").to(device)
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def __call__(self, output: torch.Tensor, batch: dict) -> torch.Tensor:
        targets = batch["class_idx"].to(output.device)
        return self.ce(output, targets)


class HierarchicalLoss:
    """Weighted multi-head loss for M2 / M3.

        L = α · CE(L1) + β · CE(L2) + γ · CE(L3, masked)

    The L3 loss is masked so that asymmetric loop samples (l3_idx == -1)
    do not contribute — their L3 label is undefined by design.

    Args:
        device: Torch device string.
        alpha:  Weight for L1 (topology) head loss.
        beta:   Weight for L2 (symmetry) head loss.
        gamma:  Weight for L3 (fine-grained) head loss.
    """

    def __init__(
        self,
        device: str,
        alpha: float = 0.25,
        beta:  float = 0.25,
        gamma: float = 0.50,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        w1 = RNAMotifDataset.class_weights("l1").to(device)
        w2 = RNAMotifDataset.class_weights("l2").to(device)
        w3 = RNAMotifDataset.class_weights("l3").to(device)
        self.ce_l1 = nn.CrossEntropyLoss(weight=w1)
        self.ce_l2 = nn.CrossEntropyLoss(weight=w2)
        self.ce_l3 = nn.CrossEntropyLoss(weight=w3)

    def __call__(
        self,
        output: dict[str, torch.Tensor],
        batch: dict,
    ) -> torch.Tensor:
        dev = output["l1"].device
        loss_l1 = self.ce_l1(output["l1"], batch["l1_idx"].to(dev))
        loss_l2 = self.ce_l2(output["l2"], batch["l2_idx"].to(dev))
        loss_l3 = self.ce_l3(output["l3"], batch["l3_idx"].to(dev))
        return self.alpha * loss_l1 + self.beta * loss_l2 + self.gamma * loss_l3


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when validation macro-F1 stops improving.

    Args:
        patience:   Number of epochs to wait after last improvement.
        min_delta:  Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = -float("inf")
        self._counter   = 0

    @property
    def should_stop(self) -> bool:
        return self._counter >= self.patience

    def step(self, val_f1: float) -> bool:
        """Call once per epoch. Returns True if a new best was reached."""
        if val_f1 > self._best + self.min_delta:
            self._best    = val_f1
            self._counter = 0
            return True
        self._counter += 1
        return False


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def _make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_ds = RNAMotifDataset(split="train")
    val_ds   = RNAMotifDataset(split="val")

    # WeightedRandomSampler: weight each sample by inverse class count.
    # Straight inverse (not sqrt) to fully counteract the 290x imbalance —
    # ensures scarce classes (e.g. 3x5 with 34 samples) appear as often as
    # large ones in each epoch.
    from collections import Counter
    counts = Counter(r["class"] for r in train_ds.rows)
    sample_weights = torch.tensor(
        [1.0 / counts[r["class"]] for r in train_ds.rows],
        dtype=torch.float32,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=(cfg.device != "cpu"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(cfg.device != "cpu"),
        persistent_workers=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
) -> float:
    """Run one training epoch. Returns mean loss over all batches."""
    model.train()
    total_loss = 0.0

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in bar:
        batch["volume"] = batch["volume"].to(device)
        output = model(batch["volume"])
        loss   = loss_fn(output, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: str,
) -> dict[str, float]:
    """Run one validation pass. Returns a dict of metrics.

    Metrics:
        loss      : mean loss over all batches
        l1_acc    : accuracy of L1 (topology) predictions
        l2_acc    : accuracy of L2 (symmetry) predictions
        l3_acc    : accuracy over L3-eligible samples only
        l3_f1     : macro-averaged F1 over 15 L3 classes (primary metric)
    """
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    l1_correct = l2_correct = l3_correct = l3_total = n_total = 0
    l3_preds: list[int] = []
    l3_trues: list[int] = []

    # Lookup tensors for mapping flat 25-class predictions → L1/L2
    l1_map = torch.tensor(CLASS_IDX_TO_L1_IDX, dtype=torch.long)
    l2_map = torch.tensor(CLASS_IDX_TO_L2_IDX, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            batch["volume"] = batch["volume"].to(device)
            output = model(batch["volume"])
            loss   = loss_fn(output, batch)
            total_loss += loss.item()

            # --- Derive predictions ---
            if isinstance(output, dict):
                # M2 / M3: explicit heads
                pred_l1 = output["l1"].argmax(1).cpu()
                pred_l2 = output["l2"].argmax(1).cpu()
                pred_l3 = output["l3"].argmax(1).cpu()
            else:
                # M1: flat 25-class output → map back to L1/L2
                pred_cls = output.argmax(1).cpu()
                pred_l1  = l1_map[pred_cls]
                pred_l2  = l2_map[pred_cls]
                pred_l3  = pred_cls  # class_idx == l3_idx for L3-eligible samples

            true_l1 = batch["l1_idx"]
            true_l2 = batch["l2_idx"]
            true_l3 = batch["l3_idx"]
            B = true_l1.size(0)
            n_total += B

            l1_correct += (pred_l1 == true_l1).sum().item()
            l2_correct += (pred_l2 == true_l2).sum().item()

            # L3: only L3-eligible samples (l3_idx != -1)
            mask = true_l3 != -1
            if mask.any():
                l3_correct += (pred_l3[mask] == true_l3[mask]).sum().item()
                l3_total   += mask.sum().item()
                l3_preds.extend(pred_l3[mask].tolist())
                l3_trues.extend(true_l3[mask].tolist())

    l3_f1 = float(f1_score(l3_trues, l3_preds, average="macro", zero_division=0)) \
            if l3_trues else 0.0

    return {
        "loss":   total_loss / len(loader),
        "l1_acc": l1_correct / n_total,
        "l2_acc": l2_correct / n_total,
        "l3_acc": l3_correct / l3_total if l3_total else 0.0,
        "l3_f1":  l3_f1,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_f1: float,
    cfg: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_f1":     best_val_f1,
        "config":          cfg,
    }, path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    cfg: TrainConfig,
    loss_fn=None,
) -> nn.Module:
    """Train a model and return it in its best-validation-F1 state.

    Args:
        model:   Instantiated model (M1, M2, or M3).
        cfg:     TrainConfig with all hyperparameters.
        loss_fn: Optional pre-built loss function. If None, automatically
                 selects FlatLoss for single-tensor outputs (M1) and
                 HierarchicalLoss for dict outputs (M2/M3) by doing a
                 dry-run forward pass on a dummy input.
    """
    device = cfg.device
    model  = model.to(device)

    # --- Auto-detect loss function ---
    if loss_fn is None:
        with torch.no_grad():
            dummy  = torch.zeros(1, 1, 64, 64, 64, device=device)
            sample = model(dummy)
        if isinstance(sample, dict):
            loss_fn = HierarchicalLoss(device, cfg.alpha, cfg.beta, cfg.gamma)
            print(f"Loss: HierarchicalLoss  (α={cfg.alpha} β={cfg.beta} γ={cfg.gamma})")
        else:
            loss_fn = FlatLoss(device)
            print("Loss: FlatLoss (25-class weighted CE)")

    # --- Optimiser + scheduler ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Reduce LR by 0.5 when val macro-F1 plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    # --- Data ---
    train_loader, val_loader = _make_loaders(cfg)
    print(f"Device : {device}")
    print(f"Train  : {len(train_loader.dataset):,} samples  |  "
          f"Val: {len(val_loader.dataset):,} samples")
    print(f"Batches/epoch: {len(train_loader)}")

    # --- Checkpoint directory + CSV log ---
    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path  = ckpt_dir / "log.csv"
    log_fields = ["epoch", "time_s", "train_loss",
                  "val_loss", "val_l1_acc", "val_l2_acc",
                  "val_l3_acc", "val_l3_f1", "lr"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # --- Training loop ---
    stopper     = EarlyStopping(cfg.patience, cfg.min_delta)
    best_val_f1 = -float("inf")

    epoch_bar = tqdm(
        range(1, cfg.epochs + 1),
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        t0 = time.time()

        train_loss = _run_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, cfg.epochs
        )
        val_metrics = _evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_metrics["l3_f1"])

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        # --- Update progress bar ---
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_metrics['val_loss'] if 'val_loss' in val_metrics else val_metrics['loss']:.4f}",
            l3_f1=f"{val_metrics['l3_f1']:.4f}",
            lr=f"{lr_now:.1e}",
        )

        # --- Print epoch summary below the bar ---
        tqdm.write(
            f"  Ep {epoch:>3}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"l1={val_metrics['l1_acc']:.3f}  "
            f"l2={val_metrics['l2_acc']:.3f}  "
            f"l3_acc={val_metrics['l3_acc']:.3f}  "
            f"l3_F1={val_metrics['l3_f1']:.4f}  "
            f"lr={lr_now:.1e}  "
            f"[{elapsed:.0f}s]"
        )

        # --- CSV log ---
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":       epoch,
                "time_s":      f"{elapsed:.1f}",
                "train_loss":  f"{train_loss:.6f}",
                "val_loss":    f"{val_metrics['loss']:.6f}",
                "val_l1_acc":  f"{val_metrics['l1_acc']:.6f}",
                "val_l2_acc":  f"{val_metrics['l2_acc']:.6f}",
                "val_l3_acc":  f"{val_metrics['l3_acc']:.6f}",
                "val_l3_f1":   f"{val_metrics['l3_f1']:.6f}",
                "lr":          f"{lr_now:.2e}",
            })

        # --- Checkpointing ---
        is_best = stopper.step(val_metrics["l3_f1"])
        if is_best:
            best_val_f1 = val_metrics["l3_f1"]
            _save_checkpoint(
                ckpt_dir / "best.pt",
                model, optimizer, scheduler, epoch, best_val_f1, cfg,
            )
            tqdm.write(f"  ✓ New best val L3-F1={best_val_f1:.4f} — saved best.pt")

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            _save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                model, optimizer, scheduler, epoch, best_val_f1, cfg,
            )

        if stopper.should_stop:
            tqdm.write(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs)."
            )
            break

    # --- Restore best weights ---
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state"])
        tqdm.write(f"\nRestored best weights (val L3-F1={best_val_f1:.4f})")

    tqdm.write(f"Training log: {log_path}")
    return model
