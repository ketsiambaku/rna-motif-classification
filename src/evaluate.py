from __future__ import annotations
"""
Comprehensive post-training evaluation for RNA motif classification models.

Unlike the lightweight per-epoch validation in train.py, this module runs
full inference on a split and produces:
  - Accuracy and macro-F1 at L1, L2, and L3
  - Per-class sensitivity at each level
  - Confusion matrices (saved as PNG figures)
  - A JSON summary of all metrics

Usage:
    from src.evaluate import evaluate, load_model
    results = evaluate(model, split="test", device="mps")
    # results saved to checkpoints/<model_name>/eval_test.json
    #               and checkpoints/<model_name>/confusion_*.png
"""
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.label_map import (
    ALL25_CLASSES,
    CLASS_IDX_TO_L1_IDX,
    CLASS_IDX_TO_L2_IDX,
    L1_CLASSES,
    L2_CLASSES,
    L3_CLASSES,
)
from src.dataset import RNAMotifDataset


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, list]:
    """Run full inference and collect ground-truth + predictions for all levels.

    Returns a dict with keys:
        true_l1, pred_l1  — L1 topology indices  (all samples)
        true_l2, pred_l2  — L2 symmetry indices   (all samples)
        true_l3, pred_l3  — L3 fine-grained indices (all 25 classes, all samples)
        true_cls, pred_cls — flat 25-class indices  (all samples; M1 only)
    """
    l1_map = torch.tensor(CLASS_IDX_TO_L1_IDX, dtype=torch.long)
    l2_map = torch.tensor(CLASS_IDX_TO_L2_IDX, dtype=torch.long)

    out: dict[str, list] = {k: [] for k in
                            ["true_l1", "pred_l1", "prob_l1",
                             "true_l2", "pred_l2", "prob_l2",
                             "true_l3", "pred_l3", "prob_l3",
                             "true_cls", "pred_cls"]}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False, dynamic_ncols=True):
            batch["volume"] = batch["volume"].to(device)
            output = model(batch["volume"])

            import torch.nn.functional as F
            if isinstance(output, dict):
                prob_l1  = F.softmax(output["l1"], dim=1).cpu()
                prob_l2  = F.softmax(output["l2"], dim=1).cpu()
                prob_l3  = F.softmax(output["l3"], dim=1).cpu()
                pred_l1  = prob_l1.argmax(1)
                pred_l2  = prob_l2.argmax(1)
                pred_l3  = prob_l3.argmax(1)
                pred_cls = pred_l3
            else:
                prob_cls = F.softmax(output, dim=1).cpu()
                pred_cls = prob_cls.argmax(1)
                pred_l1  = l1_map[pred_cls]
                pred_l2  = l2_map[pred_cls]
                pred_l3  = pred_cls
                # Proxy L1/L2 probs: sum softmax over classes belonging to each group
                prob_l1  = torch.zeros(pred_cls.size(0), 3)
                prob_l2  = torch.zeros(pred_cls.size(0), 4)
                for ci in range(25):
                    prob_l1[:, CLASS_IDX_TO_L1_IDX[ci]] += prob_cls[:, ci]
                    prob_l2[:, CLASS_IDX_TO_L2_IDX[ci]] += prob_cls[:, ci]
                prob_l3  = prob_cls           # all 25 classes; l3_idx == class_idx

            true_l1  = batch["l1_idx"]
            true_l2  = batch["l2_idx"]
            true_l3  = batch["l3_idx"]
            true_cls = batch["class_idx"] if "class_idx" in batch else true_l3

            out["true_l1"].extend(true_l1.tolist())
            out["pred_l1"].extend(pred_l1.tolist())
            out["prob_l1"].extend(prob_l1.tolist())
            out["true_l2"].extend(true_l2.tolist())
            out["pred_l2"].extend(pred_l2.tolist())
            out["prob_l2"].extend(prob_l2.tolist())
            out["true_cls"].extend(true_cls.tolist())
            out["pred_cls"].extend(pred_cls.tolist())

            out["true_l3"].extend(true_l3.tolist())
            out["pred_l3"].extend(pred_l3.tolist())
            out["prob_l3"].extend(prob_l3.tolist())

    return out


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _level_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict:
    """Compute all classification metrics for one hierarchy level.

    Returns accuracy, macro-F1, macro-sensitivity, macro-specificity,
    macro-AUC (one-vs-rest), and per-class sensitivity.
    Matches the metric set reported in the proposal baseline table.
    """
    import numpy as np
    n = len(class_names)
    labels = list(range(n))

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report     = classification_report(
        y_true, y_pred, labels=labels,
        target_names=class_names, output_dict=True, zero_division=0,
    )

    per_class_sensitivity = {
        cls: round(report[cls]["recall"], 4)
        for cls in class_names if cls in report
    }
    per_class_f1 = {
        cls: round(report[cls]["f1-score"], 4)
        for cls in class_names if cls in report
    }

    # Macro sensitivity = mean of per-class recall
    macro_sensitivity = round(
        float(np.mean([report[c]["recall"] for c in class_names if c in report])), 4
    )

    # Macro specificity: for each class c, specificity = TN / (TN + FP)
    # Computed from the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    macro_specificity = round(float(np.mean(specificities)), 4)

    # Macro AUC (one-vs-rest) — needs probability scores, approximated here
    # using a one-hot encoding of hard predictions as a proxy.
    # True AUC requires model output probabilities — computed in evaluate()
    # where logits are available; this fallback is for the confusion-only path.
    macro_auc = None  # filled in by evaluate() when logits are available

    return {
        "accuracy":              round(acc, 4),
        "macro_f1":              round(macro_f1, 4),
        "macro_sensitivity":     macro_sensitivity,
        "macro_specificity":     macro_specificity,
        "macro_auc":             macro_auc,
        "per_class_sensitivity": per_class_sensitivity,
        "per_class_f1":          per_class_f1,
    }


# ---------------------------------------------------------------------------
# Confusion matrix figure
# ---------------------------------------------------------------------------

def _plot_confusion(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Normalise rows to [0,1] for readability across imbalanced classes
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    n = len(class_names)
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Annotate cells with raw counts
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curve figure
# ---------------------------------------------------------------------------

def _plot_roc(
    y_true: list[int],
    y_prob: list[list[float]],
    class_names: list[str],
    title: str,
    save_path: Path,
) -> None:
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    n = len(class_names)
    y_true_arr  = np.array(y_true)
    y_prob_arr  = np.array(y_prob)
    y_bin       = label_binarize(y_true_arr, classes=list(range(n)))

    # Group colours for L3 (25 classes); single colour for L1/L2
    if n == 25:
        # hairpin=blue, symmetric=green, asymmetric=purple, bulge=orange
        group_colors = (
            ["#2980B9"] * 5 +   # hairpin3-7
            ["#1E8449"] * 5 +   # 1x1-5x5
            ["#F39C12"] * 5 +   # bulge1-5
            ["#7D3C98"] * 10    # 1x2-4x5
        )
    else:
        palette = plt.cm.tab10.colors
        group_colors = [palette[i % 10] for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Per-class curves (thin, semi-transparent)
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
        auc_i = float(roc_auc_score(y_bin[:, i], y_prob_arr[:, i]))
        ax.plot(fpr, tpr, lw=0.9, alpha=0.45, color=group_colors[i],
                label=f"{cls} ({auc_i:.2f})")

    # Macro average curve
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_bin[:, i], y_prob_arr[:, i])[0]
         for i in range(n) if y_bin[:, i].sum() > 0]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    count = 0
    for i in range(n):
        if y_bin[:, i].sum() == 0:
            continue
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob_arr[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
        count += 1
    mean_tpr /= count
    macro_auc = float(roc_auc_score(y_bin, y_prob_arr,
                                    multi_class="ovr", average="macro",
                                    labels=list(range(n))))
    ax.plot(all_fpr, mean_tpr, color="black", lw=2.5,
            label=f"Macro avg (AUC = {macro_auc:.4f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)

    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=title)
    ax.grid(alpha=0.25)

    # Legend: compact for L3 (too many classes to list individually)
    if n <= 4:
        ax.legend(fontsize=8, loc="lower right")
    elif n <= 15:
        ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    else:
        # For 25 classes show group patches + macro avg only
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color="#2980B9", label="Hairpin (5 classes)"),
            mpatches.Patch(color="#1E8449", label="Symmetric (5 classes)"),
            mpatches.Patch(color="#F39C12", label="Bulge (5 classes)"),
            mpatches.Patch(color="#7D3C98", label="Asymmetric (10 classes)"),
            plt.Line2D([0], [0], color="black", lw=2.5,
                       label=f"Macro avg (AUC = {macro_auc:.4f})"),
        ]
        ax.legend(handles=handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    split: str = "test",
    device: str = "cpu",
    batch_size: int = 64,
    output_dir: Optional[Path] = None,
) -> dict:
    """Run comprehensive evaluation on a data split.

    Args:
        model:      Trained model (M1, M2, or M3).
        split:      "train", "val", or "test".
        device:     Torch device string.
        batch_size: Inference batch size (can be larger than training).
        output_dir: Directory for saving JSON + figures. Defaults to
                    checkpoints/<model_name>/ if model has a name attribute,
                    otherwise current working directory.

    Returns:
        dict with keys "l1", "l2", "l3", each containing accuracy,
        macro_f1, and per_class_sensitivity.
    """
    dataset = RNAMotifDataset(split=split)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device != "cpu"),
    )

    import numpy as np
    preds = _collect_predictions(model.to(device), loader, device)

    results = {
        "split": split,
        "n_samples": len(dataset),
        "l1": _level_metrics(preds["true_l1"], preds["pred_l1"], L1_CLASSES),
        "l2": _level_metrics(preds["true_l2"], preds["pred_l2"], L2_CLASSES),
        "l3": _level_metrics(preds["true_l3"], preds["pred_l3"], ALL25_CLASSES),
    }

    # Fill in macro AUC (one-vs-rest) using collected softmax probabilities
    for level, key_true, key_prob, n_cls in [
        ("l1", "true_l1", "prob_l1", 3),
        ("l2", "true_l2", "prob_l2", 4),
        ("l3", "true_l3", "prob_l3", 25),
    ]:
        try:
            auc = roc_auc_score(
                preds[key_true],
                np.array(preds[key_prob]),
                multi_class="ovr",
                average="macro",
            )
            results[level]["macro_auc"] = round(float(auc), 4)
        except Exception:
            results[level]["macro_auc"] = None

    # --- Save outputs ---
    if output_dir is None:
        name = getattr(model, "name", "model")
        output_dir = PROJECT_ROOT / "checkpoints" / name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"eval_{split}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _plot_confusion(preds["true_l1"], preds["pred_l1"], L1_CLASSES,
                    f"L1 Confusion ({split})",
                    output_dir / f"confusion_l1_{split}.png")
    _plot_confusion(preds["true_l2"], preds["pred_l2"], L2_CLASSES,
                    f"L2 Confusion ({split})",
                    output_dir / f"confusion_l2_{split}.png")
    _plot_confusion(preds["true_l3"], preds["pred_l3"], ALL25_CLASSES,
                    f"L3 Confusion ({split})",
                    output_dir / f"confusion_l3_{split}.png")

    _plot_roc(preds["true_l1"], preds["prob_l1"], L1_CLASSES,
              f"ROC Curves — L1 Topology ({split})",
              output_dir / f"roc_l1_{split}.png")
    _plot_roc(preds["true_l2"], preds["prob_l2"], L2_CLASSES,
              f"ROC Curves — L2 Symmetry ({split})",
              output_dir / f"roc_l2_{split}.png")
    _plot_roc(preds["true_l3"], preds["prob_l3"], ALL25_CLASSES,
              f"ROC Curves — L3 Fine-grained ({split})",
              output_dir / f"roc_l3_{split}.png")

    # --- Print summary ---
    print(f"\n{'='*55}")
    print(f"  Evaluation — {split.upper()}  ({results['n_samples']:,} samples)")
    print(f"{'='*55}")
    for level, label in [("l1", "L1 topology"), ("l2", "L2 symmetry"), ("l3", "L3 fine-grained")]:
        m = results[level]
        print(f"  {label:<20}  acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    print(f"{'='*55}")
    print(f"  Saved: {json_path}")

    return results


# ---------------------------------------------------------------------------
# Checkpoint loader convenience
# ---------------------------------------------------------------------------

def load_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: str = "cpu",
) -> nn.Module:
    """Load model weights from a checkpoint saved by train.py.

    Args:
        model:           Instantiated model with the same architecture used
                         during training (weights will be overwritten).
        checkpoint_path: Path to a .pt file saved by train.py.
        device:          Device to load the model onto.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    f1    = ckpt.get("best_val_f1", "?")
    print(f"Loaded checkpoint from epoch {epoch}  (best val L3-F1={f1})")
    return model


