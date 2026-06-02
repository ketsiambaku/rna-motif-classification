# RNA Motif Classification — Implementation Roadmap

## 1. Project Goal

Demonstrate that hierarchical supervision over the CoSSMos 25-class taxonomy improves fine-grained RNA motif classification from cryo-EM density volumes, compared to a flat cross-entropy baseline.

Three models are trained and compared:

| # | Model | Supervision | Architecture |
|---|-------|-------------|--------------|
| M1 | 3D CNN Baseline | 25-class flat CE | 3D CNN encoder |
| M2 | Hierarchical ResNet | Multi-head hierarchical loss | 3D ResNet-18 |
| M3 | Conditional Swin | Conditional hierarchical inference | 3D Swin Transformer |

---

## 2. Problem Formulation (Final)

### Label Hierarchy

| Level | Classes | Scope |
|-------|---------|-------|
| L1 — topology | hairpin / internal_loop / bulge | All 25 classes |
| L2 — symmetry | hairpin / symmetric / asymmetric / bulge | All 25 classes |
| L3 — fine-grained | hairpin3–7, 1×1–5×5, bulge1–5 | **15 classes only** |

**Asymmetric internal loops (1×2 … 4×5) stop at L2.** Fine-grained asymmetric loop discrimination is deferred to future work: density patches are unoriented, so which strand is shorter is not recoverable from the volume alone. This is framed as a finding, not a limitation.

### Evaluation Claims
- L1 and L2 accuracy reported for all data
- L3 macro-F1 reported for the **15 L3-eligible classes only**
- Asymmetric loop confusion at L2 analysed separately to characterise the difficulty

---

## 3. Status

### Done ✅
- `data/` — 81,835 unique MRC files, 25 classes + unknown, organised by class folder
- `data/manifest.csv` — per-file labels (L1/L2/L3 string + integer), train/val/test split column
- `scripts/label_map.py` — importable hierarchy encoding; `encode("hairpin3")` → `(0, 0, 0)`
- `scripts/build_data_folder.py` — data consolidation from 6 sources; dedup by filename
- `scripts/build_manifest.py` — generates manifest from data/ folder
- `scripts/build_splits.py` — EMD-ID-grouped stratified 70/15/15 splits

### Done ✅ (Milestone 2, partial)
- `src/dataset.py` — MRC loading, 95th-percentile normalisation, `class_weights()` utility

### Not Yet Started ❌
- `src/train.py` — generic training loop (single-head + multi-head)
- `src/evaluate.py` — inference + metrics
- `models/flat_cnn.py` — M1
- `models/resnet3d.py` — M2
- `models/swin3d.py` — M3

### Dropped / Deferred
- Data augmentation — replaced by `WeightedRandomSampler` + weighted CE loss to handle class imbalance
- Asymmetric loop L3 — deferred to future work (see Section 9)

---

## 4. Model Architectures

All models receive a **1×64×64×64** normalised density volume as input and output class logits. No PDB features are used — density only for now, for consistency across all three models. All architectures accept an `in_channels` constructor argument (default 1) so the input can be expanded to 4 channels (density + backbone + ribose + base masks) without architectural changes when PDB files become available.

---

### M1 — 3D CNN Baseline

Replicates the CSS581 baseline architecture retrained for 25 classes.

```
Input: 1 × 64 × 64 × 64

Stage 1:  Conv3d(1,  32, 3, pad=1) → BN → ReLU
          Conv3d(32, 32, 3, pad=1) → BN → ReLU → MaxPool3d(2)   → 32×32×32

Stage 2:  Conv3d(32,  64, 3, pad=1) → BN → ReLU
          Conv3d(64,  64, 3, pad=1) → BN → ReLU → MaxPool3d(2)  → 16×16×16

Stage 3:  Conv3d(64,  128, 3, pad=1) → BN → ReLU
          Conv3d(128, 256, 3, pad=1) → BN → ReLU → MaxPool3d(2) → 8×8×8

Bottleneck: AdaptiveAvgPool3d(1) → Flatten → 256-dim embedding

Head: Linear(256, 25)
```

**Loss:** `CrossEntropyLoss` with class weights `w_c = 1 / sqrt(count_c)`, normalised to sum to 1.

**Purpose:** establishes the performance floor. Expected ~38% flat accuracy (consistent with proposal).

---

### M2 — 3D ResNet-18 with Hierarchical Multi-Head Loss

Shared backbone, three independent classification heads trained jointly.

```
Input: 1 × 64 × 64 × 64

Stem:    Conv3d(1, 32, 3, pad=1) → BN → ReLU → MaxPool3d(2)    → 32×32×32

Layer 1: ResBlock(32,  32)  × 2                                  → 32×32×32
Layer 2: ResBlock(32,  64,  stride=2) × 2                        → 16×16×16
Layer 3: ResBlock(64,  128, stride=2) × 2                        → 8×8×8
Layer 4: ResBlock(128, 256, stride=2) × 2                        → 4×4×4

Bottleneck: AdaptiveAvgPool3d(1) → Flatten → 256-dim embedding

Head L1: Linear(256, 3)    # hairpin / internal_loop / bulge
Head L2: Linear(256, 4)    # hairpin / symmetric / asymmetric / bulge
Head L3: Linear(256, 15)   # 15 L3-eligible classes only; asymmetric loops stop at L2
```

**ResBlock** (standard pre-activation):
```
x → Conv3d(C_in, C_out, 3, stride, pad=1) → BN → ReLU
  → Conv3d(C_out, C_out, 3, pad=1) → BN
  + skip (1×1 conv if dimensions change) → ReLU
```

**Loss:**
```
L = α · CE(ŷ_L1, y_L1)
  + β · CE(ŷ_L2, y_L2)
  + γ · CE(ŷ_L3[mask], y_L3[mask])

mask = (l3_idx != -1)   # excludes asymmetric loop samples from L3 loss
```

Default starting point: `α=0.25, β=0.25, γ=0.5`.
Ablation: grid over `{(1,0,0), (0,1,0), (0,0,1), (0.25,0.25,0.5), (0.1,0.1,0.8)}`.

**Purpose:** isolates the contribution of hierarchical loss (same backbone as M1 conceptually, stronger architecture).

---

### M3 — 3D Swin Transformer with Conditional Hierarchical Inference

Same hierarchical loss as M2, but predictions at each level **condition the next**:
L1 soft probabilities are concatenated with the backbone features before L2; L2 soft probabilities gate L3.

```
Input: 1 × 64 × 64 × 64

Patch Embed: 4×4×4 patches → 16×16×16 tokens, dim=48

Stage 1: SwinBlock × 2  (window=4, dim=48)                   → 16×16×16
PatchMerge → dim=96

Stage 2: SwinBlock × 2  (window=4, dim=96)                   → 8×8×8
PatchMerge → dim=192

Stage 3: SwinBlock × 6  (window=4, dim=192)                  → 4×4×4
PatchMerge → dim=384

Stage 4: SwinBlock × 2  (window=4, dim=384)                  → 2×2×2

GlobalAvgPool → 384-dim backbone embedding  (f)
```

**Conditional heads (soft gating, fully differentiable):**
```
L1 logits = Linear(384, 3)
p_L1       = softmax(L1 logits)                              # 3-dim

L2 input   = concat(f, p_L1)                                 # 387-dim
L2 logits  = Linear(387, 4)
p_L2       = softmax(L2 logits)                              # 4-dim

L3 input   = concat(f, p_L2)                                 # 388-dim
L3 logits  = Linear(388, 15)
```

During inference the conditional structure also enables **hard constraint decoding**: after predicting L2, zero out logits for classes that are impossible given the L2 prediction (e.g., if L2 = symmetric, mask all hairpin/bulge/asymmetric L3 logits). This is applied at inference only.

**Loss:** same `α/β/γ` weighted sum as M2, with identical masking for asymmetric samples.

**Memory note:** gradient checkpointing on stages 3–4 if GPU memory is a constraint.

---

## 5. Training Infrastructure Needed

All three models share a common training stack. To be built in `src/`:

| File | Purpose |
|------|---------|
| `src/dataset.py` | ✅ PyTorch Dataset — reads manifest.csv, loads MRC via `mrcfile`, 95th-pct normalisation, `class_weights()` |
| `src/train.py` | Training loop — single-head (M1) and multi-head (M2/M3) loss; `WeightedRandomSampler`; CSV logging |
| `src/evaluate.py` | Inference + metrics: per-class sensitivity, macro-F1 at L1/L2/L3, confusion matrices |
| `models/flat_cnn.py` | M1 architecture |
| `models/resnet3d.py` | M2 architecture |
| `models/swin3d.py` | M3 architecture |

### Dataset class contract
```python
sample = dataset[i]
# sample["volume"]   : FloatTensor [1, 64, 64, 64], normalised to [0, 1]
# sample["l1_idx"]   : int  (0–2)
# sample["l2_idx"]   : int  (0–3)
# sample["l3_idx"]   : int  (0–14) or -1 for asymmetric loops
# sample["filepath"] : str  (for debugging)
```

### Normalisation
**95th-percentile normalisation** (from `label.py`, the SOTA pipeline):
```python
p95 = np.percentile(data[data != 0], 95)   # non-zero voxels only
data = data / p95
data = np.clip(data, 0, 1)
```
Applied on-the-fly in the Dataset `__getitem__` — no pre-processed copies needed.

Why not min-max: cryo-EM maps have noise spikes; min-max maps those spikes to 1.0 and compresses all real signal into a narrow range. The 95th-percentile normalisation is robust to outliers and preserves relative density contrast.

### Multi-channel input (future work)
`label.py` generates three binary spatial masks from paired PDB files (backbone / ribose / nucleobase), enabling **4-channel input** `[density, backbone, ribose, base]`. This is how the Murugadass 2026 dataset was designed to be used.

PDB files are not currently available. When they are, the Dataset class `in_channels` parameter can be increased from 1 → 4 with no other changes to the model architectures (all three models use `in_channels` as a constructor argument).

### Class imbalance strategy (replaces augmentation)
Data augmentation was considered and dropped — no transformation is safe for all classes without risking label corruption. Class imbalance is handled by two complementary mechanisms instead:

1. **`WeightedRandomSampler`** in the DataLoader — oversamples rare classes at batch construction time so each batch sees a balanced class distribution
2. **Weighted `CrossEntropyLoss`** via `RNAMotifDataset.class_weights(level)` — inverse-sqrt weights so rare classes contribute proportionally more to the gradient

---

## 6. Evaluation Protocol

Shared across all three models — computed on the locked test split.

| Metric | Scope |
|--------|-------|
| Accuracy, macro-F1, macro-sensitivity | L1 (3 classes) |
| Accuracy, macro-F1, macro-sensitivity | L2 (4 classes) |
| Per-class sensitivity, macro-F1 | L3 (15 classes only) |
| Confusion matrix | L1, L2, L3 (separate figures) |
| L2 confusion — asymmetric loops | How often symmetric/asymmetric are confused |

**Ablation (M2 only):** vary `(α, β, γ)` to isolate the contribution of each level's supervision signal.

---

## 7. Milestones

### Milestone 1 — Data Infrastructure ✅ DONE
- Data consolidation, manifest, label map, stratified splits

### Milestone 2 — Training Infrastructure 🔄 IN PROGRESS
- ✅ `src/dataset.py` — MRC loading, 95th-percentile normalisation, class weights
- ❌ `src/train.py` — generic loop (single-head + multi-head), WeightedRandomSampler, CSV logging
- ❌ `src/evaluate.py` — per-class metrics, confusion matrices
- ❌ Sanity check: overfit one batch on each model

### Milestone 3 — Baseline (M1)
- Implement `models/flat_cnn.py`
- Train to convergence on 25 classes
- Generate L1/L2/L3 evaluation report and confusion matrices
- **Gate:** reproduce ~38% flat accuracy reported in proposal

### Milestone 4 — Hierarchical ResNet (M2)
- Implement `models/resnet3d.py`
- Train with default `α=0.25, β=0.25, γ=0.5`
- Run ablation over loss weights
- Compare to M1: does macro-F1 improve at L3? Are within-family confusions reduced?

### Milestone 5 — Conditional Swin (M3)
- Implement `models/swin3d.py` (or adapt `timm`/`monai` if a 3D Swin is available)
- Train with same loss as M2 best config
- Evaluate conditional inference decoding at test time
- Compare to M2

### Milestone 6 — Analysis & Writing
- Unified comparison table across M1/M2/M3
- Confusion matrix figures at each hierarchy level
- Ablation table (M2 loss weights)
- Asymmetric loop L2 confusion analysis (the "future work" finding)
- Update proposal → full paper

---

## 9. Future Work (Out of Scope for This Paper)

| Enhancement | Description |
|-------------|-------------|
| **Asymmetric loop L3** | Fine-grained size discrimination within asymmetric loops. Blocked by data scarcity and orientation ambiguity — patches are unoriented so which strand is shorter is unrecoverable from density alone. Requires either orientation-aware patch extraction or more data. |
| **4-channel input** | Add backbone / ribose / nucleobase label maps as input channels 2–4 (generated by `label.py` from PDB files). All three model architectures already accept `in_channels` as a constructor argument. |
| **Taxonomic distance loss at L3** | The current L3 head uses flat cross-entropy, so `bulge1`→`bulge2` and `bulge1`→`bulge5` are penalised equally within L3. A distance-weighted loss (e.g. label smoothing with taxonomy-derived targets, or a hierarchical softmax using the CoSSMos tree) would penalise size-adjacent errors less than cross-family errors at L3, extending the hierarchical supervision principle all the way down to the leaf level. |
| **Orientation-aware extraction** | Re-extracting patches in a canonical helix-aligned frame would make asymmetric loop subclasses distinguishable and enable L3 classification for the full 25-class taxonomy. |

---

## 8. Directory Layout (Target)

```
rna-motif-classification/
├── data/
│   ├── manifest.csv          # labels + split for every MRC file
│   ├── 1x1/ … hairpin7/      # class folders (25 + unknown)
│   └── data_class_histogram.png
├── documents/
│   ├── proposal.tex
│   └── roadmap.md            # this file
├── models/
│   ├── flat_cnn.py           # M1
│   ├── resnet3d.py           # M2
│   └── swin3d.py             # M3
├── scripts/
│   ├── label_map.py
│   ├── build_data_folder.py
│   ├── build_manifest.py
│   └── build_splits.py
└── src/
    ├── dataset.py
    ├── augmentation.py
    ├── train.py
    └── evaluate.py
```
