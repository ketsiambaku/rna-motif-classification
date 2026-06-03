from __future__ import annotations
"""
Label hierarchy for RNA motif classification.

Three-level CoSSMos hierarchy:
  L1  — topology:  hairpin | internal_loop | bulge
  L2  — symmetry:  hairpin | symmetric | asymmetric | bulge
  L3  — fine-grained (25 classes): all 25 motif types, including asymmetric.
         Asymmetric loops are classified at L3 by their specific type (1x2, etc.).
"""

# class_folder -> (l1, l2, l3)
LABEL_MAP: dict[str, tuple[str, str, str]] = {
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
    # --- Asymmetric internal loops — classified at L3 by specific type ---
    "1x2":      ("internal_loop", "asymmetric", "1x2"),
    "1x3":      ("internal_loop", "asymmetric", "1x3"),
    "1x4":      ("internal_loop", "asymmetric", "1x4"),
    "1x5":      ("internal_loop", "asymmetric", "1x5"),
    "2x3":      ("internal_loop", "asymmetric", "2x3"),
    "2x4":      ("internal_loop", "asymmetric", "2x4"),
    "2x5":      ("internal_loop", "asymmetric", "2x5"),
    "3x4":      ("internal_loop", "asymmetric", "3x4"),
    "3x5":      ("internal_loop", "asymmetric", "3x5"),
    "4x5":      ("internal_loop", "asymmetric", "4x5"),
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

# L3 covers all 25 motif types.
# Indices 0-14: L3-eligible (hairpins, symmetric loops, bulges)
# Indices 15-24: asymmetric loops
# Invariant: for all classes, l3_idx == class_idx.
L3_CLASSES = [
    # 0-4: hairpins
    "hairpin3", "hairpin4", "hairpin5", "hairpin6", "hairpin7",
    # 5-9: symmetric internal loops
    "1x1", "2x2", "3x3", "4x4", "5x5",
    # 10-14: bulges
    "bulge1", "bulge2", "bulge3", "bulge4", "bulge5",
    # 15-24: asymmetric internal loops
    "1x2", "1x3", "1x4", "1x5",
    "2x3", "2x4", "2x5",
    "3x4", "3x5", "4x5",
]

ASYMMETRIC_CLASSES = [
    "1x2", "1x3", "1x4", "1x5",
    "2x3", "2x4", "2x5",
    "3x4", "3x5", "4x5",
]

# ALL25_CLASSES == L3_CLASSES (kept for backwards compatibility with evaluate.py imports)
ALL25_CLASSES = L3_CLASSES

L1_TO_IDX = {c: i for i, c in enumerate(L1_CLASSES)}
L2_TO_IDX = {c: i for i, c in enumerate(L2_CLASSES)}
L3_TO_IDX = {c: i for i, c in enumerate(L3_CLASSES)}
CLASS_TO_IDX = L3_TO_IDX  # identical — l3_idx == class_idx for all 25 classes

# Lookup tables: 25-class index → L1/L2 index (used to evaluate M1 at L1/L2).
CLASS_IDX_TO_L1_IDX = [
    L1_TO_IDX[LABEL_MAP[cls][0]] for cls in ALL25_CLASSES
]
CLASS_IDX_TO_L2_IDX = [
    L2_TO_IDX[LABEL_MAP[cls][1]] for cls in ALL25_CLASSES
]

# L3 index → valid L3 index range within the L2 group (used by swin3d.decode())
# hairpin  (L2=0): indices  0- 4
# symmetric(L2=1): indices  5- 9
# bulge    (L2=3): indices 10-14
# asymmetric(L2=2): indices 15-24
L2_IDX_TO_L3_RANGE: dict[int, tuple[int, int]] = {
    0: (0,  5),
    1: (5,  10),
    2: (15, 25),
    3: (10, 15),
}


def encode(class_folder: str) -> tuple[int, int, int, int]:
    """Return (l1_idx, l2_idx, l3_idx, class_idx) for a class folder name.

    l3_idx == class_idx for all 25 classes (asymmetric loops included).
    Raises KeyError for unknown class folders.
    """
    l1, l2, l3 = LABEL_MAP[class_folder]
    l3_idx = L3_TO_IDX[l3]
    return (
        L1_TO_IDX[l1],
        L2_TO_IDX[l2],
        l3_idx,
        l3_idx,   # class_idx == l3_idx
    )
