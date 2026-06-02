"""
Build rna-motif-classification/data/ from multiple MRC source directories.

Deduplicates by filename. When both plain and _normalized versions of the same
file exist, the plain (non-normalized) version is kept. Typos in class names
(haripin3/5/6) are corrected to hairpin3/5/6.
"""
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"

# (source_dir, class_extraction_method)
# "dir"      -> class = immediate parent directory of the file within source_dir
# "filename" -> class = field[2] of EMD-{ID}_{PDB}_{CLASS}_{NUM}[_normalized].mrc
SOURCES = [
    (PROJECT_ROOT / "dataset", "dir"),
    (PROJECT_ROOT / "dataset2", "dir"),
    (DOWNLOADS / "TrainingAndValidation", "filename"),
    (DOWNLOADS / "densityMapsAndPDBsBatch3" / "allMaps", "filename"),
    (DOWNLOADS / "densityMapsAndPDBsBatch1" / "allMaps", "filename"),
    (DOWNLOADS / "testSET", "filename"),
]

DEST = PROJECT_ROOT / "data"

TYPO_MAP = {
    "haripin3": "hairpin3",
    "haripin5": "hairpin5",
    "haripin6": "hairpin6",
}


def class_from_dir(mrc_path: Path, source_root: Path) -> str:
    return mrc_path.relative_to(source_root).parts[0]


def class_from_filename(name: str) -> str:
    stem = name
    if stem.endswith("_normalized.mrc"):
        stem = stem[: -len("_normalized.mrc")]
    elif stem.endswith(".mrc"):
        stem = stem[: -len(".mrc")]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("EMD-"):
        return parts[2]
    return "unknown"


def dedup_key(name: str) -> str:
    """Basename with _normalized stripped, used as deduplication key."""
    if name.endswith("_normalized.mrc"):
        return name[: -len("_normalized.mrc")] + ".mrc"
    return name


def main() -> None:
    # dedup_key -> (src_path, class_name, is_normalized)
    registry: dict[str, tuple[Path, str, bool]] = {}

    for source_dir, mode in SOURCES:
        if not source_dir.exists():
            print(f"  [skip] {source_dir} not found")
            continue
        for mrc_path in sorted(source_dir.rglob("*.mrc")):
            if mode == "dir":
                cls = class_from_dir(mrc_path, source_dir)
            else:
                cls = class_from_filename(mrc_path.name)

            cls = TYPO_MAP.get(cls, cls)
            is_norm = mrc_path.name.endswith("_normalized.mrc")
            key = dedup_key(mrc_path.name)

            if key not in registry:
                registry[key] = (mrc_path, cls, is_norm)
            elif is_norm is False and registry[key][2] is True:
                # Replace normalized entry with the plain version
                registry[key] = (mrc_path, cls, is_norm)

    stats: dict[str, int] = defaultdict(int)
    for key, (src_path, cls, _) in registry.items():
        dest_dir = DEST / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_dir / src_path.name)
        stats[cls] += 1

    total = sum(stats.values())
    print(f"\nCopied {total} unique MRC files into {len(stats)} classes:\n")
    for cls in sorted(stats):
        print(f"  {cls:20s}  {stats[cls]:6d} files")
    print(f"\n  {'TOTAL':20s}  {total:6d} files")
    print(f"\nOutput: {DEST}")


if __name__ == "__main__":
    main()
