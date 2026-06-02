"""
Add a train/val/test split column to data/manifest.csv.

Split strategy:
  - Group files by source map ID (EMD-XXXXX or PDB accession)
  - For each class independently, shuffle map IDs and assign 70/15/15
  - All patches from the same map stay in the same split (no covariate leakage)
  - Reproducible via a fixed random seed

Writes the split label into a new 'split' column in manifest.csv in-place.
"""
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = remainder (~0.15)
SEED = 42


def source_id(filepath: str) -> str:
    """Extract map-level identifier from a file path."""
    name = Path(filepath).name
    m = re.match(r"(EMD-\d+)_", name)
    if m:
        return m.group(1)
    # simple format: PDBID_NUM.mrc
    m = re.match(r"([A-Z0-9]{4})_\d+\.mrc", name)
    if m:
        return m.group(1)
    return name  # fallback: treat whole filename as its own group


def split_ids(ids: list[str], seed: int) -> dict[str, str]:
    """Shuffle ids and assign train/val/test labels. Returns {id: split}."""
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * TRAIN_FRAC)
    n_val   = round(n * VAL_FRAC)
    assignment = {}
    for i, mid in enumerate(shuffled):
        if i < n_train:
            assignment[mid] = "train"
        elif i < n_train + n_val:
            assignment[mid] = "val"
        else:
            assignment[mid] = "test"
    return assignment


def main() -> None:
    rows = []
    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    # Group map IDs by class
    class_to_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        class_to_ids[row["class"]].add(source_id(row["filepath"]))

    # Assign train/val/test to each map ID, per class
    id_to_split: dict[str, dict[str, str]] = {}  # class -> {map_id: split}
    for cls, ids in class_to_ids.items():
        id_to_split[cls] = split_ids(sorted(ids), seed=SEED)

    # Annotate rows
    for row in rows:
        cls = row["class"]
        mid = source_id(row["filepath"])
        row["split"] = id_to_split[cls][mid]

    # Write back
    fieldnames = list(rows[0].keys())
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {MANIFEST}\n")
    _print_summary(rows)


def _print_summary(rows: list[dict]) -> None:
    from collections import Counter

    # Per-class split counts
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["class"]][row["split"]] += 1

    # Also aggregate by L2
    l2_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        l2_counts[row["l2"]][row["split"]] += 1

    header = f"{'class':<12} {'total':>6}  {'train':>6} {'val':>5} {'test':>5}"
    print(header)
    print("-" * len(header))
    total = Counter()
    for cls in sorted(counts):
        c = counts[cls]
        n = sum(c.values())
        print(f"{cls:<12} {n:>6}  {c['train']:>6} {c['val']:>5} {c['test']:>5}")
        total += c
    print("-" * len(header))
    n = sum(total.values())
    print(f"{'TOTAL':<12} {n:>6}  {total['train']:>6} {total['val']:>5} {total['test']:>5}")

    print("\nBy L2 group:")
    for l2 in ["hairpin", "symmetric", "asymmetric", "bulge"]:
        c = l2_counts[l2]
        n = sum(c.values())
        print(f"  {l2:<14} {n:>6}  train {c['train']:>6}  val {c['val']:>5}  test {c['test']:>5}")


if __name__ == "__main__":
    main()
