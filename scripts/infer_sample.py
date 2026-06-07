"""
Single-sample inference with M3 V3 (SwinTransformer3D).

Loads one MRC density volume, runs it through the trained model, and
prints the predicted class at L1, L2, and L3 with confidence scores.
Optionally accepts a paired PDB file path for display purposes only
(PDB is not used as model input — density-only inference).

Usage:
    python scripts/infer_sample.py \
        --mrc /path/to/sample.mrc \
        --pdb /path/to/sample.pdb \
        --checkpoint checkpoints/swin3d_v3/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.label_map import L1_CLASSES, L2_CLASSES, ALL25_CLASSES
from src.dataset import _normalize_95p


def load_volume(mrc_path: str, target_size=(64, 64, 64)) -> torch.Tensor:
    import mrcfile
    import numpy as np
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data.astype("float32")
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape}")
    data = _normalize_95p(data)
    volume = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    if tuple(volume.shape[2:]) != target_size:
        volume = F.interpolate(volume, size=target_size,
                               mode="trilinear", align_corners=False)
    return volume  # [1, 1, 64, 64, 64]


def top_k(logits: torch.Tensor, names: list, k: int = 3):
    probs = F.softmax(logits[0], dim=0)
    topk  = probs.topk(min(k, len(names)))
    return [(names[i], float(topk.values[j])) for j, i in enumerate(topk.indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrc",        required=True,  help="Path to .mrc density file")
    parser.add_argument("--pdb",        default=None,   help="Path to paired .pdb file (display only)")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    else:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps"  if torch.backends.mps.is_available()
                  else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    from models.swin3d import SwinTransformer3D
    model = SwinTransformer3D().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint  : {args.checkpoint}  (epoch {ckpt['epoch']}, "
          f"best val L3-F1={ckpt['best_val_f1']:.4f})")

    # ── Load volume ───────────────────────────────────────────────────────
    volume = load_volume(args.mrc, target_size=(64, 64, 64)).to(device)
    print(f"MRC file           : {args.mrc}")
    if args.pdb:
        print(f"PDB file           : {args.pdb}  (not used as model input)")
    print(f"Volume shape       : {tuple(volume.shape[1:])}")
    print(f"Device             : {device}")
    print()

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model(volume)

    # ── Hard-constraint decode: use L2 prediction to constrain L3 ─────────
    output_decoded = model.decode(output)

    # ── Results ───────────────────────────────────────────────────────────
    pred_l1 = output["l1"].argmax(1).item()
    pred_l2 = output["l2"].argmax(1).item()
    pred_l3 = output_decoded["l3"].argmax(1).item()

    conf_l1 = float(F.softmax(output["l1"][0], dim=0).max())
    conf_l2 = float(F.softmax(output["l2"][0], dim=0).max())
    conf_l3 = float(F.softmax(output_decoded["l3"][0], dim=0).max())

    print("=" * 52)
    print("  Prediction")
    print("=" * 52)
    print(f"  L1  topology    : {L1_CLASSES[pred_l1]:<18}  ({conf_l1*100:.1f}% confidence)")
    print(f"  L2  symmetry    : {L2_CLASSES[pred_l2]:<18}  ({conf_l2*100:.1f}% confidence)")
    print(f"  L3  fine-grained: {ALL25_CLASSES[pred_l3]:<18}  ({conf_l3*100:.1f}% confidence)")
    print("=" * 52)

    print("\n  Top-3 at L3 (after hierarchical constraint):")
    for name, prob in top_k(output_decoded["l3"], ALL25_CLASSES, k=3):
        bar = "█" * int(prob * 30)
        print(f"    {name:<14}  {prob*100:5.1f}%  {bar}")

    print("\n  Full L2 distribution:")
    probs_l2 = F.softmax(output["l2"][0], dim=0)
    for i, name in enumerate(L2_CLASSES):
        bar = "█" * int(float(probs_l2[i]) * 30)
        print(f"    {name:<14}  {float(probs_l2[i])*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
