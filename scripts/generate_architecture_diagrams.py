"""
Generate architecture diagrams for M1, M2, M3 and save to documents/.

Usage:
    python scripts/generate_architecture_diagrams.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "documents"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared drawing primitives
# ---------------------------------------------------------------------------

COLORS = {
    "input":   "#AED6F1",   # light blue
    "conv":    "#A9DFBF",   # light green
    "pool":    "#F9E79F",   # light yellow
    "res":     "#A9DFBF",   # light green
    "attn":    "#D7BDE2",   # light purple
    "merge":   "#F5CBA7",   # light orange
    "head":    "#F1948A",   # light red
    "output":  "#D5DBDB",   # light grey
    "embed":   "#85C1E9",   # medium blue
}


def box(ax, x, y, w, h, label, sublabel="", color="#CCCCCC", fontsize=9):
    """Draw a rounded rectangle with a label."""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#555555",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.05 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.13, sublabel,
                ha="center", va="center", fontsize=fontsize - 1.5,
                color="#444444", zorder=4)


def arrow(ax, x, y_top, y_bot, label=""):
    """Draw a downward arrow with an optional shape label."""
    ax.annotate("", xy=(x, y_bot + 0.02), xytext=(x, y_top - 0.02),
                arrowprops=dict(arrowstyle="-|>", color="#333333",
                                lw=1.2, mutation_scale=12), zorder=2)
    if label:
        ax.text(x + 0.07, (y_top + y_bot) / 2, label,
                ha="left", va="center", fontsize=7.5, color="#555555")


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def legend(ax, items):
    handles = [mpatches.Patch(facecolor=c, edgecolor="#555", label=l)
               for l, c in items]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              framealpha=0.9, edgecolor="#AAAAAA")


# ---------------------------------------------------------------------------
# M1 — 3D CNN Baseline
# ---------------------------------------------------------------------------

def draw_m1():
    fig, ax = plt.subplots(figsize=(4.5, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 10.5)
    ax.axis("off")
    ax.set_title("M1 — 3D CNN Baseline", fontsize=13, fontweight="bold", pad=10)

    cx = 0.5
    bw, bh, gap = 0.72, 0.38, 0.70

    layers = [
        (9.9,  "Input volume",        "[1 × 64 × 64 × 64]",   "input"),
        (9.1,  "Conv3d + BN + ReLU",  "1 → 32 ch",            "conv"),
        (8.4,  "Conv3d + BN + ReLU",  "32 → 32 ch",           "conv"),
        (7.7,  "MaxPool3d(2)",         "→ [32 × 32³]",         "pool"),
        (6.9,  "Conv3d + BN + ReLU",  "32 → 64 ch",           "conv"),
        (6.2,  "Conv3d + BN + ReLU",  "64 → 64 ch",           "conv"),
        (5.5,  "MaxPool3d(2)",         "→ [64 × 16³]",         "pool"),
        (4.7,  "Conv3d + BN + ReLU",  "64 → 128 ch",          "conv"),
        (4.0,  "Conv3d + BN + ReLU",  "128 → 256 ch",         "conv"),
        (3.3,  "MaxPool3d(2)",         "→ [256 × 8³]",         "pool"),
        (2.5,  "AdaptiveAvgPool3d(1)","→ [256]",              "pool"),
        (1.75, "Dropout(0.3)",         "",                      "pool"),
        (1.05, "Linear(256 → 25)",    "flat classifier",       "head"),
        (0.3,  "Output logits",        "[25 classes]",         "output"),
    ]

    for i, (y, lbl, sub, ckey) in enumerate(layers):
        box(ax, cx, y, bw, bh, lbl, sub, COLORS[ckey])
        if i < len(layers) - 1:
            arrow(ax, cx, y - bh / 2, layers[i + 1][0] + bh / 2)

    # Stage brackets
    for y_top, y_bot, label in [
        (9.3, 7.45, "Stage 1"),
        (7.1, 5.25, "Stage 2"),
        (4.9, 3.1,  "Stage 3"),
    ]:
        ax.annotate("", xy=(0.94, y_bot), xytext=(0.94, y_top),
                    arrowprops=dict(arrowstyle="-", color="#888", lw=1))
        ax.text(0.97, (y_top + y_bot) / 2, label,
                ha="left", va="center", fontsize=7.5, color="#666",
                rotation=90)

    legend(ax, [
        ("Conv + BN + ReLU", COLORS["conv"]),
        ("Pooling",           COLORS["pool"]),
        ("Classification head", COLORS["head"]),
    ])
    save(fig, "architecture_m1_cnn.png")


# ---------------------------------------------------------------------------
# M2 — 3D ResNet-18
# ---------------------------------------------------------------------------

def draw_m2():
    fig, ax = plt.subplots(figsize=(6.0, 14))
    ax.set_xlim(0, 1.15); ax.set_ylim(0, 14.5)
    ax.axis("off")
    ax.set_title("M2 — 3D ResNet-18\nHierarchical Multi-Head + Constrained Decode",
                 fontsize=13, fontweight="bold", pad=10)

    cx = 0.50
    bw, bh = 0.76, 0.40

    backbone = [
        (13.8, "Input volume",         "[1 × 64 × 64 × 64]",          "input"),
        (12.9, "Stem",                 "Conv+BN+ReLU+MaxPool → [32 × 32³]", "conv"),
        (11.9, "ResBlock × 2",         "32 → 32 ch  [32³]",            "res"),
        (10.9, "ResBlock × 2",         "32 → 64 ch  [16³]",            "res"),
        (9.9,  "ResBlock × 2",         "64 → 128 ch  [8³]",            "res"),
        (8.9,  "ResBlock × 2",         "128 → 256 ch  [4³]",           "res"),
        (7.9,  "AdaptiveAvgPool3d(1)", "→ [256]",                      "pool"),
        (7.0,  "Dropout(0.3)",         "",                              "pool"),
    ]

    for i, (y, lbl, sub, ckey) in enumerate(backbone):
        box(ax, cx, y, bw, bh, lbl, sub, COLORS[ckey])
        if i < len(backbone) - 1:
            arrow(ax, cx, y - bh / 2, backbone[i + 1][0] + bh / 2)

    # Three independent heads side by side
    heads = [
        (0.15, "L1 Head\nLinear(256→3)",  "hairpin\ninternal_loop\nbulge",      "head"),
        (0.50, "L2 Head\nLinear(256→4)",  "hairpin / symmetric\nasymmetric / bulge", "head"),
        (0.85, "L3 Head\nLinear(256→25)", "all 25\nmotif classes",               "head"),
    ]

    for hx, _, _, _ in heads:
        ax.annotate("", xy=(hx, 5.85), xytext=(cx, 7.0 - bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#333",
                                   lw=1.1, mutation_scale=10), zorder=2)

    for hx, lbl, sub, ckey in heads:
        box(ax, hx, 5.3, 0.27, 0.72, lbl, sub, COLORS[ckey], fontsize=7.5)

    # Raw logit outputs
    for hx, out_lbl in [(0.15, "3 logits"), (0.50, "4 logits"), (0.85, "25 logits (raw)")]:
        ax.annotate("", xy=(hx, 4.55), xytext=(hx, 4.95),
                    arrowprops=dict(arrowstyle="-|>", color="#333",
                                   lw=1.0, mutation_scale=9), zorder=2)
        box(ax, hx, 4.3, 0.27, 0.38, out_lbl, "", COLORS["output"], fontsize=7.0)

    # decode() box — constrained inference
    DECODE_COLOR = "#FADBD8"
    DECODE_EDGE  = "#C0392B"

    # Arrow from L2 logits down into decode box
    ax.annotate("", xy=(0.50, 3.52), xytext=(0.50, 4.12),
                arrowprops=dict(arrowstyle="-|>", color=DECODE_EDGE,
                                lw=1.3, mutation_scale=11), zorder=2)
    # Arrow from L3 raw logits into decode box
    ax.annotate("", xy=(0.77, 3.52), xytext=(0.85, 4.12),
                arrowprops=dict(arrowstyle="-|>", color=DECODE_EDGE,
                                lw=1.3, mutation_scale=11,
                                connectionstyle="arc3,rad=0.15"), zorder=2)

    decode_rect = FancyBboxPatch((0.25, 2.95), 0.60, 0.55,
                                  boxstyle="round,pad=0.04",
                                  linewidth=2.0, edgecolor=DECODE_EDGE,
                                  facecolor=DECODE_COLOR, zorder=3)
    ax.add_patch(decode_rect)
    ax.text(0.55, 3.30, "decode()  — Hard Constraint", ha="center", va="center",
            fontsize=9, fontweight="bold", color=DECODE_EDGE, zorder=4)
    ax.text(0.55, 3.08, "argmax(L2) → zero out impossible L3 logits",
            ha="center", va="center", fontsize=7.5, color="#555", zorder=4)

    # Constraint examples
    ax.text(0.55, 2.72,
            "L2=hairpin → keep L3[0:5]   L2=symmetric → keep L3[5:10]\n"
            "L2=bulge   → keep L3[10:15]  L2=asymmetric → keep L3[15:25]",
            ha="center", va="center", fontsize=7.0, color="#666",
            style="italic", zorder=4)

    # Final constrained output
    ax.annotate("", xy=(0.55, 2.05), xytext=(0.55, 2.65),
                arrowprops=dict(arrowstyle="-|>", color="#333",
                                lw=1.2, mutation_scale=10), zorder=2)
    box(ax, 0.15, 1.6, 0.27, 0.50, "L1 pred", "3 classes", COLORS["output"], fontsize=7.5)
    box(ax, 0.50, 1.6, 0.27, 0.50, "L2 pred", "4 classes", COLORS["output"], fontsize=7.5)
    box(ax, 0.85, 1.6, 0.27, 0.50, "L3 pred", "constrained\n25 classes", COLORS["output"], fontsize=7.0)

    for hx in [0.15, 0.85]:
        ax.annotate("", xy=(hx, 1.85), xytext=(hx, 4.12),
                    arrowprops=dict(arrowstyle="-|>", color="#AAA",
                                   lw=1.0, mutation_scale=8,
                                   connectionstyle="arc3,rad=0.0"), zorder=1)

    # Inference only label
    ax.text(0.88, 3.22, "inference\nonly", ha="left", va="center",
            fontsize=7.5, color=DECODE_EDGE, style="italic")

    # Layer labels
    for y, label in [(11.9, "Layer 1"), (10.9, "Layer 2"),
                      (9.9,  "Layer 3"), (8.9,  "Layer 4")]:
        ax.text(0.91, y, label, ha="left", va="center", fontsize=7.5, color="#666")

    # Training note
    ax.text(0.55, 0.55,
            "Training: L = 0.25·CE(L1) + 0.25·CE(L2) + 0.50·CE(L3)  — heads independent",
            ha="center", va="center", fontsize=7.5, color="#555", style="italic")

    legend(ax, [
        ("Residual Block",       COLORS["res"]),
        ("Pooling / Dropout",    COLORS["pool"]),
        ("Classification head",  COLORS["head"]),
        ("Constrained decode()", DECODE_COLOR),
    ])
    save(fig, "architecture_m2_resnet.png")


# ---------------------------------------------------------------------------
# M3 — Swin Transformer 3D
# ---------------------------------------------------------------------------

def draw_m3():
    fig, ax = plt.subplots(figsize=(6.5, 14))
    ax.set_xlim(-0.1, 1.3); ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("M3 — 3D Swin Transformer\nConditional Hierarchical Heads",
                 fontsize=13, fontweight="bold", pad=10)

    cx = 0.55
    bw, bh = 0.80, 0.40

    backbone = [
        (13.4, "Input volume",        "[1 × 64 × 64 × 64]",              "input"),
        (12.5, "Patch Embedding",     "Conv3d(1→48, k=4, s=4)  →  [48 × 16³]", "embed"),
        (11.5, "Stage 1  (×2 blocks)","W-MSA + SW-MSA  →  [48 × 16³]",  "attn"),
        (10.5, "Patch Merging",       "48 → 96  →  [96 × 8³]",           "merge"),
        (9.5,  "Stage 2  (×2 blocks)","W-MSA + SW-MSA  →  [96 × 8³]",   "attn"),
        (8.5,  "Patch Merging",       "96 → 192  →  [192 × 4³]",         "merge"),
        (7.5,  "Stage 3  (×6 blocks)","W-MSA + SW-MSA  →  [192 × 4³]",  "attn"),
        (6.5,  "Patch Merging",       "192 → 384  →  [384 × 2³]",        "merge"),
        (5.5,  "Stage 4  (×2 blocks)","W-MSA + SW-MSA  →  [384 × 2³]",  "attn"),
        (4.5,  "LayerNorm + GlobalAvgPool", "→  [384]",                   "pool"),
    ]

    for i, (y, lbl, sub, ckey) in enumerate(backbone):
        box(ax, cx, y, bw, bh, lbl, sub, COLORS[ckey])
        if i < len(backbone) - 1:
            arrow(ax, cx, y - bh / 2, backbone[i + 1][0] + bh / 2)

    # Conditional heads
    y_l1 = 3.3
    box(ax, cx, y_l1, bw, bh, "L1 Head  —  Linear(384 → 3)",
        "topology:  hairpin  /  internal_loop  /  bulge", COLORS["head"])
    arrow(ax, cx, 4.5 - bh / 2, y_l1 + bh / 2)
    ax.text(cx + 0.44, y_l1 + 0.05, "3 logits", ha="left", va="center",
            fontsize=8, color="#444", fontstyle="italic")

    y_l2 = 2.1
    box(ax, cx, y_l2, bw, bh, "L2 Head  —  Linear(387 → 4)",
        "symmetry:  hairpin / symmetric / asymmetric / bulge", COLORS["head"])
    arrow(ax, cx, y_l1 - bh / 2, y_l2 + bh / 2)
    ax.text(cx - 0.44, (y_l1 + y_l2) / 2, "cat[f, softmax(L1)]",
            ha="right", va="center", fontsize=7.5, color="#666", style="italic")
    ax.text(cx + 0.44, y_l2 + 0.05, "4 logits", ha="left", va="center",
            fontsize=8, color="#444", fontstyle="italic")

    y_l3 = 0.9
    box(ax, cx, y_l3, bw, bh, "L3 Head  —  Linear(388 → 25)",
        "fine-grained:  all 25 motif classes", COLORS["head"])
    arrow(ax, cx, y_l2 - bh / 2, y_l3 + bh / 2)
    ax.text(cx - 0.44, (y_l2 + y_l3) / 2, "cat[f, softmax(L2)]",
            ha="right", va="center", fontsize=7.5, color="#666", style="italic")
    ax.text(cx + 0.44, y_l3 + 0.05, "25 logits", ha="left", va="center",
            fontsize=8, color="#444", fontstyle="italic")

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=COLORS["embed"], edgecolor="#555", label="Patch Embedding"),
            mpatches.Patch(facecolor=COLORS["attn"],  edgecolor="#555", label="Swin Attention Block"),
            mpatches.Patch(facecolor=COLORS["merge"], edgecolor="#555", label="Patch Merging"),
            mpatches.Patch(facecolor=COLORS["head"],  edgecolor="#555", label="Classification Head"),
        ],
        loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#AAAAAA",
    )
    save(fig, "architecture_m3_swin.png")


# ---------------------------------------------------------------------------
# Class hierarchy org-chart
# ---------------------------------------------------------------------------

def draw_class_hierarchy():
    # ── Layout constants ────────────────────────────────────────────────
    # x-slots: each L3 class gets 1 unit; groups separated by 1.5 unit gap
    # hairpin(0-4)  gap  symmetric(6-10)  gap  asymmetric(12-21)  gap  bulge(23-27)
    L3_X = {
        "hairpin3": 0, "hairpin4": 1, "hairpin5": 2, "hairpin6": 3, "hairpin7": 4,
        "1x1": 6,  "2x2": 7,  "3x3": 8,  "4x4": 9,  "5x5": 10,
        "1x2": 12, "1x3": 13, "1x4": 14, "1x5": 15,
        "2x3": 16, "2x4": 17, "2x5": 18,
        "3x4": 19, "3x5": 20, "4x5": 21,
        "bulge1": 23, "bulge2": 24, "bulge3": 25, "bulge4": 26, "bulge5": 27,
    }
    L2_GROUPS = {
        "hairpin":    (["hairpin3","hairpin4","hairpin5","hairpin6","hairpin7"], "#AED6F1", "#2980B9"),
        "symmetric":  (["1x1","2x2","3x3","4x4","5x5"],                        "#A9DFBF", "#1E8449"),
        "asymmetric": (["1x2","1x3","1x4","1x5","2x3","2x4","2x5","3x4","3x5","4x5"], "#D7BDE2", "#7D3C98"),
        "bulge":      (["bulge1","bulge2","bulge3","bulge4","bulge5"],           "#F9E79F", "#B7950B"),
    }
    L1_GROUPS = {
        "hairpin":       (["hairpin"],               "#AED6F1", "#2471A3"),
        "internal_loop": (["symmetric","asymmetric"],"#A9DFBF", "#1A5276"),
        "bulge":         (["bulge"],                 "#F9E79F", "#9A7D0A"),
    }

    y_l1, y_l2, y_l3 = 6.2, 4.1, 2.0
    node_w, node_h   = 0.85, 0.55
    l3_w,  l3_h      = 0.85, 0.50

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.set_xlim(-1, 28.5)
    ax.set_ylim(0.8, 7.5)
    ax.axis("off")
    ax.set_title("RNA Motif Classification Hierarchy", fontsize=16,
                 fontweight="bold", pad=14)

    def node(x, y, w, h, label, fc, ec, fs=8.5, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.04",
                              linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def connect(x1, y1, x2, y2, color="#888888"):
        ax.plot([x1, x2], [y1 - 0.01, y2 + 0.01],
                color=color, lw=1.1, zorder=1, solid_capstyle="round")

    # ── L3 nodes ────────────────────────────────────────────────────────
    for l2_name, (classes, fc, ec) in L2_GROUPS.items():
        for cls in classes:
            x = L3_X[cls]
            node(x, y_l3, l3_w, l3_h, cls, fc, ec, fs=7.8)

    # ── L2 nodes + connections to L3 ────────────────────────────────────
    L2_X = {}
    for l2_name, (classes, fc, ec) in L2_GROUPS.items():
        xs = [L3_X[c] for c in classes]
        cx = (min(xs) + max(xs)) / 2
        L2_X[l2_name] = cx
        node(cx, y_l2, node_w * (1 + 0.3 * (len(l2_name) > 8)),
             node_h, l2_name, fc, ec, fs=9, bold=True)
        for cls in classes:
            connect(L3_X[cls], y_l3 + l3_h/2, cx, y_l2 - node_h/2, ec)

    # ── L1 nodes + connections to L2 ────────────────────────────────────
    for l1_name, (l2_children, fc, ec) in L1_GROUPS.items():
        xs = [L2_X[c] for c in l2_children]
        cx = (min(xs) + max(xs)) / 2
        node(cx, y_l1, node_w * (1 + 0.35 * (len(l1_name) > 6)),
             node_h, l1_name, fc, ec, fs=10, bold=True)
        for l2_name in l2_children:
            _, _, ec2 = L2_GROUPS[l2_name]
            connect(L2_X[l2_name], y_l2 + node_h/2, cx, y_l1 - node_h/2, ec2)

    # ── Level labels on left ─────────────────────────────────────────────
    for y, lbl, detail in [
        (y_l1, "L1", "Topology\n(3 classes)"),
        (y_l2, "L2", "Symmetry\n(4 classes)"),
        (y_l3, "L3", "Fine-grained\n(25 classes)"),
    ]:
        ax.text(-0.6, y, lbl, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#333")
        ax.text(-0.6, y - 0.38, detail, ha="center", va="center",
                fontsize=7.5, color="#666")
        ax.axhline(y=y + node_h/2 + 0.1, xmin=0.032, xmax=0.99,
                   color="#DDDDDD", lw=0.8, zorder=0)

    # ── Legend ───────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#2471A3", label="Hairpin"),
        mpatches.Patch(facecolor="#A9DFBF", edgecolor="#1E8449", label="Symmetric loop"),
        mpatches.Patch(facecolor="#D7BDE2", edgecolor="#7D3C98", label="Asymmetric loop"),
        mpatches.Patch(facecolor="#F9E79F", edgecolor="#B7950B", label="Bulge"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.95, edgecolor="#AAAAAA", ncol=4,
              bbox_to_anchor=(1.0, 0.0))

    save(fig, "class_hierarchy.png")


# ---------------------------------------------------------------------------
# Hierarchy variant: L1 + L2 only
# ---------------------------------------------------------------------------

def draw_hierarchy_l1_l2():
    L2_GROUPS = {
        "hairpin":    (["hairpin"],               "#AED6F1", "#2980B9"),
        "symmetric":  (["symmetric"],             "#A9DFBF", "#1E8449"),
        "asymmetric": (["asymmetric"],            "#D7BDE2", "#7D3C98"),
        "bulge":      (["bulge"],                 "#F9E79F", "#B7950B"),
    }
    L1_TO_L2 = {
        "hairpin":       ["hairpin"],
        "internal_loop": ["symmetric", "asymmetric"],
        "bulge":         ["bulge"],
    }
    L1_FC = {"hairpin": "#AED6F1", "internal_loop": "#A9DFBF", "bulge": "#F9E79F"}
    L1_EC = {"hairpin": "#2471A3", "internal_loop": "#1A5276", "bulge": "#9A7D0A"}

    L2_X = {"hairpin": 2, "symmetric": 6, "asymmetric": 11, "bulge": 15}
    L1_X = {
        "hairpin":       2,
        "internal_loop": (L2_X["symmetric"] + L2_X["asymmetric"]) / 2,
        "bulge":         15,
    }

    y_l1, y_l2 = 4.0, 2.0
    node_w, node_h = 1.6, 0.65

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 17); ax.set_ylim(0.8, 5.5)
    ax.axis("off")
    ax.set_title("RNA Motif Hierarchy — L1 & L2", fontsize=14,
                 fontweight="bold", pad=12)

    def node(x, y, label, fc, ec, fs=10, bold=False):
        rect = FancyBboxPatch((x - node_w/2, y - node_h/2), node_w, node_h,
                              boxstyle="round,pad=0.05", linewidth=1.5,
                              edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def connect(x1, y1, x2, y2, color):
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.3, zorder=1)

    for l2, (_, fc, ec) in L2_GROUPS.items():
        node(L2_X[l2], y_l2, l2, fc, ec, fs=10, bold=True)

    for l1, l2_children in L1_TO_L2.items():
        node(L1_X[l1], y_l1, l1, L1_FC[l1], L1_EC[l1], fs=11, bold=True)
        for l2 in l2_children:
            _, _, ec = L2_GROUPS[l2]
            connect(L2_X[l2], y_l2 + node_h/2, L1_X[l1], y_l1 - node_h/2, ec)

    for y, lbl, detail in [(y_l1, "L1", "Topology  (3 classes)"),
                            (y_l2, "L2", "Symmetry  (4 classes)")]:
        ax.text(-0.2, y, lbl, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#333")
        ax.text(-0.2, y - 0.38, detail, ha="center", va="center",
                fontsize=8, color="#666")

    handles = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#2471A3", label="Hairpin"),
        mpatches.Patch(facecolor="#A9DFBF", edgecolor="#1E8449", label="Symmetric loop"),
        mpatches.Patch(facecolor="#D7BDE2", edgecolor="#7D3C98", label="Asymmetric loop"),
        mpatches.Patch(facecolor="#F9E79F", edgecolor="#B7950B", label="Bulge"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.95, edgecolor="#AAAAAA", ncol=4)

    save(fig, "class_hierarchy_l1_l2.png")


# ---------------------------------------------------------------------------
# Hierarchy variant: L2 + L3 only
# ---------------------------------------------------------------------------

def draw_hierarchy_l2_l3():
    L2_GROUPS = {
        "hairpin":    (["hairpin3","hairpin4","hairpin5","hairpin6","hairpin7"], "#AED6F1", "#2980B9"),
        "symmetric":  (["1x1","2x2","3x3","4x4","5x5"],                        "#A9DFBF", "#1E8449"),
        "asymmetric": (["1x2","1x3","1x4","1x5","2x3","2x4","2x5","3x4","3x5","4x5"], "#D7BDE2", "#7D3C98"),
        "bulge":      (["bulge1","bulge2","bulge3","bulge4","bulge5"],           "#F9E79F", "#B7950B"),
    }
    L3_X = {
        "hairpin3": 0, "hairpin4": 1, "hairpin5": 2, "hairpin6": 3, "hairpin7": 4,
        "1x1": 6,  "2x2": 7,  "3x3": 8,  "4x4": 9,  "5x5": 10,
        "1x2": 12, "1x3": 13, "1x4": 14, "1x5": 15,
        "2x3": 16, "2x4": 17, "2x5": 18,
        "3x4": 19, "3x5": 20, "4x5": 21,
        "bulge1": 23, "bulge2": 24, "bulge3": 25, "bulge4": 26, "bulge5": 27,
    }

    y_l2, y_l3 = 4.0, 2.0
    node_w, node_h = 0.85, 0.55
    l3_w,  l3_h   = 0.85, 0.50

    fig, ax = plt.subplots(figsize=(24, 6))
    ax.set_xlim(-1, 28.5); ax.set_ylim(0.8, 5.5)
    ax.axis("off")
    ax.set_title("RNA Motif Hierarchy — L2 & L3", fontsize=14,
                 fontweight="bold", pad=12)

    def node(x, y, w, h, label, fc, ec, fs=8.5, bold=False):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.04", linewidth=1.4,
                              edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def connect(x1, y1, x2, y2, color):
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.1, zorder=1)

    for l2_name, (classes, fc, ec) in L2_GROUPS.items():
        for cls in classes:
            node(L3_X[cls], y_l3, l3_w, l3_h, cls, fc, ec, fs=7.8)

    L2_X = {}
    for l2_name, (classes, fc, ec) in L2_GROUPS.items():
        xs = [L3_X[c] for c in classes]
        cx = (min(xs) + max(xs)) / 2
        L2_X[l2_name] = cx
        node(cx, y_l2, node_w * (1 + 0.3 * (len(l2_name) > 8)),
             node_h, l2_name, fc, ec, fs=10, bold=True)
        for cls in classes:
            connect(L3_X[cls], y_l3 + l3_h/2, cx, y_l2 - node_h/2, ec)

    for y, lbl, detail in [(y_l2, "L2", "Symmetry  (4 classes)"),
                            (y_l3, "L3", "Fine-grained  (25 classes)")]:
        ax.text(-0.6, y, lbl, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#333")
        ax.text(-0.6, y - 0.38, detail, ha="center", va="center",
                fontsize=7.5, color="#666")

    handles = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#2980B9", label="Hairpin"),
        mpatches.Patch(facecolor="#A9DFBF", edgecolor="#1E8449", label="Symmetric loop"),
        mpatches.Patch(facecolor="#D7BDE2", edgecolor="#7D3C98", label="Asymmetric loop"),
        mpatches.Patch(facecolor="#F9E79F", edgecolor="#B7950B", label="Bulge"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.95, edgecolor="#AAAAAA", ncol=4)

    save(fig, "class_hierarchy_l2_l3.png")


# ---------------------------------------------------------------------------
# Hierarchy variant: L3 only
# ---------------------------------------------------------------------------

def draw_hierarchy_l3_only():
    groups = {
        "hairpin":    (["hairpin3","hairpin4","hairpin5","hairpin6","hairpin7"], "#AED6F1", "#2980B9"),
        "symmetric":  (["1x1","2x2","3x3","4x4","5x5"],                        "#A9DFBF", "#1E8449"),
        "asymmetric": (["1x2","1x3","1x4","1x5","2x3","2x4","2x5","3x4","3x5","4x5"], "#D7BDE2", "#7D3C98"),
        "bulge":      (["bulge1","bulge2","bulge3","bulge4","bulge5"],           "#F9E79F", "#B7950B"),
    }

    L3_X = {
        "hairpin3": 0, "hairpin4": 1, "hairpin5": 2, "hairpin6": 3, "hairpin7": 4,
        "1x1": 6,  "2x2": 7,  "3x3": 8,  "4x4": 9,  "5x5": 10,
        "1x2": 12, "1x3": 13, "1x4": 14, "1x5": 15,
        "2x3": 16, "2x4": 17, "2x5": 18,
        "3x4": 19, "3x5": 20, "4x5": 21,
        "bulge1": 23, "bulge2": 24, "bulge3": 25, "bulge4": 26, "bulge5": 27,
    }

    node_w, node_h = 0.85, 0.55

    fig, ax = plt.subplots(figsize=(24, 4))
    ax.set_xlim(-1, 28.5); ax.set_ylim(0.5, 4.0)
    ax.axis("off")
    ax.set_title("RNA Motif Hierarchy — L3 (25 Fine-grained Classes)",
                 fontsize=14, fontweight="bold", pad=12)

    y_node   = 1.8
    y_label  = 3.2

    for g_name, (classes, fc, ec) in groups.items():
        xs = [L3_X[c] for c in classes]
        cx = (min(xs) + max(xs)) / 2

        # Group bracket / background
        pad = 0.3
        rect = FancyBboxPatch(
            (min(xs) - node_w/2 - pad, y_node - node_h/2 - pad),
            max(xs) - min(xs) + node_w + 2*pad,
            node_h + 2*pad,
            boxstyle="round,pad=0.05",
            linewidth=1.5, edgecolor=ec,
            facecolor=fc, alpha=0.15, zorder=1,
        )
        ax.add_patch(rect)

        # Group name above
        ax.text(cx, y_label, g_name, ha="center", va="center",
                fontsize=11, fontweight="bold", color=ec)
        ax.plot([cx], [y_label - 0.22], marker="v", color=ec,
                markersize=6, zorder=3)

        # Individual class nodes
        for cls in classes:
            x = L3_X[cls]
            node_rect = FancyBboxPatch(
                (x - node_w/2, y_node - node_h/2), node_w, node_h,
                boxstyle="round,pad=0.04", linewidth=1.3,
                edgecolor=ec, facecolor=fc, zorder=3,
            )
            ax.add_patch(node_rect)
            ax.text(x, y_node, cls, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", zorder=4)

    # Level label
    ax.text(-0.6, y_node, "L3", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#333")
    ax.text(-0.6, y_node - 0.35, "Fine-grained\n(25 classes)",
            ha="center", va="center", fontsize=7.5, color="#666")

    handles = [
        mpatches.Patch(facecolor="#AED6F1", edgecolor="#2980B9", label="Hairpin (5)"),
        mpatches.Patch(facecolor="#A9DFBF", edgecolor="#1E8449", label="Symmetric loop (5)"),
        mpatches.Patch(facecolor="#D7BDE2", edgecolor="#7D3C98", label="Asymmetric loop (10)"),
        mpatches.Patch(facecolor="#F9E79F", edgecolor="#B7950B", label="Bulge (5)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.95, edgecolor="#AAAAAA", ncol=4)

    save(fig, "class_hierarchy_l3.png")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating architecture diagrams...")
    draw_m1()
    draw_m2()
    draw_m3()
    draw_class_hierarchy()
    draw_hierarchy_l1_l2()
    draw_hierarchy_l2_l3()
    draw_hierarchy_l3_only()
    print("Done.")
