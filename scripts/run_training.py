"""
Entry point for single-model training — called by deploy.sh on each GPU server.

Usage:
    python scripts/run_training.py --model cnn_baseline --run-id v2
    python scripts/run_training.py --model resnet3d     --run-id v2
    python scripts/run_training.py --model swin3d       --run-id v2

--run-id appends a suffix to the checkpoint directory so runs never overwrite
each other:
    checkpoints/cnn_baseline_v2/log.csv
    checkpoints/cnn_baseline_v2/best.pt
    checkpoints/cnn_baseline_v2/train.log

Omitting --run-id falls back to checkpoints/<model>/ (plain, no suffix).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one RNA motif classification model.")
    parser.add_argument(
        "--model", required=True,
        choices=["cnn_baseline", "resnet3d", "swin3d"],
        help="Which model to train.",
    )
    parser.add_argument("--epochs",     type=int,   default=None, help="Override epoch count.")
    parser.add_argument("--batch_size", type=int,   default=None, help="Override batch size.")
    parser.add_argument("--lr",         type=float, default=None, help="Override learning rate.")
    parser.add_argument("--patience",   type=int,   default=None, help="Override early-stopping patience.")
    parser.add_argument("--run-id",     type=str,   default=None,
                        help="Version suffix appended to checkpoint dir, e.g. 'v2' → checkpoints/<model>_v2/.")
    args = parser.parse_args()

    # ── Model + default config ────────────────────────────────────────────
    if args.model == "cnn_baseline":
        from models.flat_cnn import CNN3DBaseline
        model = CNN3DBaseline()
        cfg   = TrainConfig(model_name="cnn_baseline", epochs=100, batch_size=32)

    elif args.model == "resnet3d":
        from models.resnet3d import ResNet3DBaseline
        model = ResNet3DBaseline()
        cfg   = TrainConfig(model_name="resnet3d", epochs=100, batch_size=32)

    elif args.model == "swin3d":
        from models.swin3d import SwinTransformer3D
        model = SwinTransformer3D()
        # Transformers need much lower LR than CNNs — 1e-3 causes immediate collapse.
        # Lower patience too: 20 epochs to allow slower warmup before stopping.
        cfg   = TrainConfig(model_name="swin3d", epochs=100, batch_size=16,
                            lr=1e-4, patience=20)

    # ── CLI overrides ─────────────────────────────────────────────────────
    if args.epochs     is not None: cfg.epochs     = args.epochs
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.lr         is not None: cfg.lr         = args.lr
    if args.patience   is not None: cfg.patience   = args.patience
    if args.run_id     is not None:
        cfg.model_name = f"{cfg.model_name}_{args.run_id}"

    # ── Print banner ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Model    : {args.model}")
    print(f"  Run ID   : {args.run_id or '(none)'}")
    print(f"  Ckpt dir : checkpoints/{cfg.model_name}/")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Batch    : {cfg.batch_size}")
    print(f"  LR       : {cfg.lr}")
    print(f"  Patience : {cfg.patience}")
    print(f"  Device   : {cfg.device}")
    print("=" * 60)

    # ── Train ─────────────────────────────────────────────────────────────
    train(model, cfg)

    # ── Evaluate on test set (runs automatically after training) ──────────
    print("\n" + "=" * 60)
    print("  Running test-set evaluation...")
    print("=" * 60)
    from src.evaluate import evaluate
    evaluate(
        model,
        split="test",
        device=cfg.device,
        batch_size=cfg.batch_size * 2,
    )


if __name__ == "__main__":
    main()
