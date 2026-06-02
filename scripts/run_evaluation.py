"""
Run full evaluation on a saved checkpoint and print all metrics.

Produces:
  - Accuracy, macro-F1, macro-sensitivity, macro-specificity, macro-AUC at L1 / L2 / L3
  - Per-class sensitivity for every class at every level
  - Confusion matrix figures (PNG) at L1, L2, L3
  - eval_<split>.json summary saved alongside the checkpoint

Usage:
    python scripts/run_evaluation.py --model cnn_baseline
    python scripts/run_evaluation.py --model resnet3d   --split val
    python scripts/run_evaluation.py --model swin3d     --checkpoint checkpoints/swin3d/epoch_0050.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate, load_model
from src.train import TrainConfig


MODEL_MAP = {
    "cnn_baseline": "models.flat_cnn.CNN3DBaseline",
    "resnet3d":     "models.resnet3d.ResNet3DBaseline",
    "swin3d":       "models.swin3d.SwinTransformer3D",
}


def _load_model_class(model_name: str):
    module_path, class_name = MODEL_MAP[model_name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _print_results(results: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  Evaluation results — {results['split'].upper()}"
          f"  ({results['n_samples']:,} samples)")
    print(f"{'='*65}")

    headers = ["Level", "Accuracy", "Macro-F1", "Sensitivity", "Specificity", "AUC"]
    print(f"  {'Level':<10} {'Accuracy':>9} {'Macro-F1':>9} "
          f"{'Sensitivity':>12} {'Specificity':>12} {'AUC':>8}")
    print(f"  {'-'*63}")

    for level, label in [("l1", "L1 topology"), ("l2", "L2 symmetry"), ("l3", "L3 fine-grained")]:
        m = results[level]
        auc = f"{m['macro_auc']:.4f}" if m.get("macro_auc") is not None else "  N/A"
        print(f"  {label:<18} {m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
              f"{m['macro_sensitivity']:>12.4f} {m['macro_specificity']:>12.4f} {auc:>8}")

    print(f"\n  {'─'*63}")
    print(f"  Per-class sensitivity at L3:")
    for cls, sens in results["l3"]["per_class_sensitivity"].items():
        bar = "█" * int(sens * 20)
        print(f"    {cls:<14}  {sens:.4f}  {bar}")

    print(f"\n  Per-class sensitivity at L2:")
    for cls, sens in results["l2"]["per_class_sensitivity"].items():
        bar = "█" * int(sens * 20)
        print(f"    {cls:<14}  {sens:.4f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt file. Defaults to checkpoints/<model>/best.pt")
    parser.add_argument("--device",     default=None,
                        help="Device override (cuda / mps / cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                PROJECT_ROOT / "checkpoints" / args.model / "best.pt"

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")

    print(f"Model      : {args.model}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Split      : {args.split}")
    print(f"Device     : {device}")

    # ── Build model and load weights ──────────────────────────────────────────
    ModelClass = _load_model_class(args.model)
    model = ModelClass()
    model = load_model(model, ckpt_path, device=device)

    # ── Run evaluation ────────────────────────────────────────────────────────
    output_dir = ckpt_path.parent
    results = evaluate(model, split=args.split, device=device, output_dir=output_dir)

    _print_results(results)

    print(f"\n  Saved to : {output_dir / f'eval_{args.split}.json'}")
    print(f"  Figures  : {output_dir}/confusion_{{l1,l2,l3}}_{args.split}.png\n")


if __name__ == "__main__":
    main()
