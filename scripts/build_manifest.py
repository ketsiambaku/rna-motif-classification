"""
Generate data/manifest.csv — one row per MRC file with L1/L2/L3 labels.

Columns:
  filepath   — path relative to project root  (e.g. data/hairpin3/foo.mrc)
  class      — folder name                    (e.g. hairpin3)
  l1         — topology label                 (hairpin | internal_loop | bulge)
  l2         — symmetry label                 (hairpin | symmetric | asymmetric | bulge)
  l3         — fine-grained label or ""       (empty for asymmetric loops)
  l1_idx     — integer index for l1
  l2_idx     — integer index for l2
  l3_idx     — integer index for l3, or -1
"""
import csv
from pathlib import Path

from label_map import LABEL_MAP, encode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "manifest.csv"

FIELDNAMES = ["filepath", "class", "l1", "l2", "l3", "l1_idx", "l2_idx", "l3_idx", "class_idx"]


def main() -> None:
    rows = []
    skipped = []

    for class_dir in sorted(DATA_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        if cls not in LABEL_MAP:
            skipped.append(cls)
            continue

        l1, l2, l3 = LABEL_MAP[cls]
        l1_idx, l2_idx, l3_idx, class_idx = encode(cls)

        for mrc in sorted(class_dir.glob("*.mrc")):
            rows.append({
                "filepath":  str(mrc.relative_to(PROJECT_ROOT)),
                "class":     cls,
                "l1":        l1,
                "l2":        l2,
                "l3":        l3 or "",
                "l1_idx":    l1_idx,
                "l2_idx":    l2_idx,
                "l3_idx":    l3_idx,
                "class_idx": class_idx,
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows):,} rows to {OUT_CSV}")
    if skipped:
        print(f"Skipped folders (not in LABEL_MAP): {skipped}")

    # Summary
    from collections import Counter
    l1_counts = Counter(r["l1"] for r in rows)
    l2_counts = Counter(r["l2"] for r in rows)
    print("\nL1 distribution:")
    for label in ["hairpin", "internal_loop", "bulge"]:
        print(f"  {label:20s}  {l1_counts[label]:6,}")
    print("\nL2 distribution:")
    for label in ["hairpin", "symmetric", "asymmetric", "bulge"]:
        print(f"  {label:20s}  {l2_counts[label]:6,}")
    l3_rows = [r for r in rows if r["l3"]]
    print(f"\nL3-eligible rows: {len(l3_rows):,}  (asymmetric excluded: {len(rows)-len(l3_rows):,})")


if __name__ == "__main__":
    main()
