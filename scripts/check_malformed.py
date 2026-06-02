"""
Scan data/manifest.csv for malformed MRC files and optionally remove them.

A file is malformed if:
  - It cannot be opened by mrcfile
  - Its data array is not 3-dimensional
  - Its data array is all zeros or has shape with any dimension == 0

Usage:
    python scripts/check_malformed.py             # report only
    python scripts/check_malformed.py --fix        # remove bad rows from manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mrcfile
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "data" / "manifest.csv"


def check_file(filepath: str) -> tuple[str, str | None]:
    """
    Returns (filepath, None) if the file is valid 3D.
    Returns (filepath, reason) if malformed.
    """
    path = PROJECT_ROOT / filepath
    try:
        with mrcfile.open(path, permissive=True) as mrc:
            data = mrc.data
            if data is None:
                return filepath, "data is None"
            if data.ndim != 3:
                return filepath, f"ndim={data.ndim} shape={data.shape}"
            if 0 in data.shape:
                return filepath, f"zero-length dimension shape={data.shape}"
    except Exception as e:
        return filepath, f"open error: {e}"
    return filepath, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true",
                        help="Remove malformed rows from manifest.csv in place")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for scanning (default 8)")
    args = parser.parse_args()

    with open(MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Scanning {len(rows):,} files with {args.workers} workers...")

    bad: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_file, row["filepath"]): row
                   for row in rows}
        for future in tqdm(as_completed(futures), total=len(rows), unit="file"):
            filepath, reason = future.result()
            if reason is not None:
                bad.append((filepath, reason))

    print(f"\n{'='*55}")
    print(f"  Total files scanned : {len(rows):,}")
    print(f"  Malformed files     : {len(bad):,}")
    print(f"  Clean files         : {len(rows) - len(bad):,}")
    print(f"{'='*55}")

    if bad:
        print("\nMalformed files:")
        for filepath, reason in sorted(bad):
            print(f"  {reason:30s}  {filepath}")

    if args.fix and bad:
        bad_set = {fp for fp, _ in bad}
        clean_rows = [r for r in rows if r["filepath"] not in bad_set]
        fieldnames = list(rows[0].keys())
        with open(MANIFEST, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)
        print(f"\n✓ Removed {len(bad):,} rows from manifest.csv ({len(clean_rows):,} remain)")
    elif bad and not args.fix:
        print(f"\nRun with --fix to remove these {len(bad):,} rows from manifest.csv")


if __name__ == "__main__":
    main()
