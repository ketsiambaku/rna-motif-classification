from __future__ import annotations
"""
Label hierarchy for RNA motif classification.

Three-level CoSSMos hierarchy:
  L1  — topology:  hairpin | internal_loop | bulge
  L2  — symmetry:  hairpin | symmetric | asymmetric | bulge
  L3  — fine-grained (15 classes): hairpins + symmetric loops + bulges only.
         Asymmetric loops stop at L2; L3 is None for those classes.
"""

# class_folder -> (l1, l2, l3_or_None)
LABEL_MAP: dict[str, tuple[str, str, str | None]] = {
    # --- Hairpins ---
    "hairpin3": ("hairpin",       "hairpin",    "hairpin3"),
    "hairpin4": ("hairpin",       "hairpin",    "hairpin4"),
    "hairpin5": ("hairpin",       "hairpin",    "hairpin5"),
    "hairpin6": ("hairpin",       "hairpin",    "hairpin6"),
    "hairpin7": ("hairpin",       "hairpin",    "hairpin7"),
    # --- Symmetric internal loops ---
    "1x1":      ("internal_loop", "symmetric",  "1x1"),
    "2x2":      ("internal_loop", "symmetric",  "2x2"),
    "3x3":      ("internal_loop", "symmetric",  "3x3"),
    "4x4":      ("internal_loop", "symmetric",  "4x4"),
    "5x5":      ("internal_loop", "symmetric",  "5x5"),
    # --- Asymmetric internal loops (L3 withheld — future work) ---
    "1x2":      ("internal_loop", "asymmetric", None),
    "1x3":      ("internal_loop", "asymmetric", None),
    "1x4":      ("internal_loop", "asymmetric", None),
    "1x5":      ("internal_loop", "asymmetric", None),
    "2x3":      ("internal_loop", "asymmetric", None),
    "2x4":      ("internal_loop", "asymmetric", None),
    "2x5":      ("internal_loop", "asymmetric", None),
    "3x4":      ("internal_loop", "asymmetric", None),
    "3x5":      ("internal_loop", "asymmetric", None),
    "4x5":      ("internal_loop", "asymmetric", None),
    # --- Bulges ---
    "bulge1":   ("bulge",         "bulge",      "bulge1"),
    "bulge2":   ("bulge",         "bulge",      "bulge2"),
    "bulge3":   ("bulge",         "bulge",      "bulge3"),
    "bulge4":   ("bulge",         "bulge",      "bulge4"),
    "bulge5":   ("bulge",         "bulge",      "bulge5"),
}

# Ordered label sets for consistent integer encoding
L1_CLASSES = ["hairpin", "internal_loop", "bulge"]
L2_CLASSES = ["hairpin", "symmetric", "asymmetric", "bulge"]
L3_CLASSES = [
    "hairpin3", "hairpin4", "hairpin5", "hairpin6", "hairpin7",
    "1x1", "2x2", "3x3", "4x4", "5x5",
    "bulge1", "bulge2", "bulge3", "bulge4", "bulge5",
]

L1_TO_IDX = {c: i for i, c in enumerate(L1_CLASSES)}
L2_TO_IDX = {c: i for i, c in enumerate(L2_CLASSES)}
L3_TO_IDX = {c: i for i, c in enumerate(L3_CLASSES)}

# ALL25_CLASSES — full 25-class index used by the flat baseline (M1).
# L3_CLASSES occupy indices 0–14 so that for L3-eligible samples:
#   class_idx == l3_idx  (invariant relied on during evaluation).
# Asymmetric loop classes occupy indices 15–24.
ASYMMETRIC_CLASSES = [
    "1x2", "1x3", "1x4", "1x5",
    "2x3", "2x4", "2x5",
    "3x4", "3x5", "4x5",
]
ALL25_CLASSES = L3_CLASSES + ASYMMETRIC_CLASSES
CLASS_TO_IDX  = {c: i for i, c in enumerate(ALL25_CLASSES)}

# Lookup tables: 25-class index → L1/L2 index (used to evaluate M1 at L1/L2).
CLASS_IDX_TO_L1_IDX = [
    L1_TO_IDX[LABEL_MAP[cls][0]] for cls in ALL25_CLASSES
]
CLASS_IDX_TO_L2_IDX = [
    L2_TO_IDX[LABEL_MAP[cls][1]] for cls in ALL25_CLASSES
]


def encode(class_folder: str) -> tuple[int, int, int, int]:
    """Return (l1_idx, l2_idx, l3_idx, class_idx) for a class folder name.

    l3_idx is -1 for asymmetric loops (no L3 label defined).
    class_idx covers all 25 classes (0–24); for L3-eligible classes
    class_idx == l3_idx.
    Raises KeyError for unknown class folders.
    """
    l1, l2, l3 = LABEL_MAP[class_folder]
    return (
        L1_TO_IDX[l1],
        L2_TO_IDX[l2],
        L3_TO_IDX[l3] if l3 is not None else -1,
        CLASS_TO_IDX[class_folder],
    )
