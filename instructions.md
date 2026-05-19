# Instructions — running the placement network end-to-end

This document covers everything needed to install, verify, run inference, and (optionally) retrain the placement network from scratch.

> If you just want a one-paragraph overview of *what* this is, read [README.md](README.md). If you want to know *how* it works, read [METHODOLOGY.md](METHODOLOGY.md).

---

## 1. Prerequisites

| Required for | Tool | Notes |
|---|---|---|
| Always | Python 3.10 | 3.11 also works; 3.12 untested. |
| Always | git | To clone the repo. |
| GPU inference / training | CUDA 11.8 or 12.1 + NVIDIA driver | Falls back to CPU if absent. |
| Generating the PDF report | `pdflatex` (MiKTeX or TeX Live) | Optional. |
| Modal training / precompute | A free [Modal](https://modal.com) account | Optional; only needed to reproduce the corpus from scratch. |

---

## 2. Installation

### Option A — pip (recommended)

```bash
git clone https://github.com/augustdua/TurmpfNesting.git
cd TurmpfNesting

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# CPU-only PyTorch:
pip install -r requirements.txt

# GPU PyTorch (CUDA 12.1) — install the right torch wheel BEFORE the rest:
pip install torch==2.1.* --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option B — conda

```bash
git clone https://github.com/augustdua/TurmpfNesting.git
cd TurmpfNesting

conda env create -f environment.yml
conda activate placement-net
```

Either path gives you the same environment. Use whichever you prefer.

---

## 3. Verify the install

Geometry primitives (random polygon generation, IFP, Shapely reward) — no data files needed, no model needed:

```bash
python -m tests.test_geometry
```

You should see a series of asserted geometry checks pass.

---

## 4. Run inference with the pre-trained model

### 4.1 Zero-data demo (works out of the box)

`scripts/demo.py` generates a random convex free-space and a smaller random convex part on the fly, runs the model, and prints the placement. No external pkl required.

```bash
python -m scripts.demo                              # GPU if available, else CPU
python -m scripts.demo --device cpu --seed 7        # deterministic CPU run
python -m scripts.demo --area-ratio 0.5 --seed 42   # harder problem (bigger part)
```

Expected output:

```
=== Result ===
  theta  =  XXX.X deg
  (x, y) = (+0.XXX, +0.XXX)  (in [-1, 1])
  reward = XX.XXX
```

### 4.2 Use the model in your own code

```python
from shapely.wkt import loads as wkt_loads
from src.inference.placement import PlacementModel

pm = PlacementModel(
    ckpt="checkpoints/perthet_combined/final.pt",
    device="cuda",          # or "cpu"
)

fs   = wkt_loads("POLYGON ((...))")     # your free-space polygon
part = wkt_loads("POLYGON ((...))")     # your part polygon

theta, x, y, reward = pm.place(
    fs, part,
    refine_pixels=10,       # set 0 to skip Shapely refinement
    refine_thetas=2,        # set 0 to skip rotation refinement
)
```

Both polygons must lie in `[-1, 1] × [-1, 1]` (the model's normalized world). If yours are in arbitrary units, center and scale them first (see `scripts/demo.py::normalize_to_unit` for a one-liner).

### 4.3 Smoke test against the held-out validation set

`scripts/smoke_refine.py` evaluates the model on `N` random pairs from the convex validation split. **This requires the data file `data/bc_snapshot_raster128.pkl` (482 MB), which is NOT in the repo** — only the trained checkpoint is committed. If you have a copy, drop it into `data/` and run:

```bash
python -m scripts.smoke_refine                                    # 5 random val pairs
python -m scripts.smoke_refine --n 25 --device cpu                # bigger sample
python -m scripts.smoke_refine --refine-pixels 0 --refine-thetas 0  # model-only
```

If you don't have the pkl, use `scripts/demo.py` instead — it exercises the same code path on synthetic pairs.

---

## 5. Generate the PDF report

The 4 MB LaTeX report (12 convex + 6 concave example pages, full methodology, summary) is already committed at `visualizations/report/placement_pipeline_report.pdf`. To rebuild it from scratch:

```bash
python -m scripts.generate_placement_report                       # default (12 + 6)
python -m scripts.generate_placement_report --n-convex 4 --n-concave 2  # quicker
python -m scripts.generate_placement_report --no-compile          # emit .tex only
```

**Requirements for this step:**
- `pdflatex` on PATH (install MiKTeX on Windows or TeX Live elsewhere; verify with `pdflatex --version`).
- `data/bc_snapshot_raster128.pkl` (482 MB) — convex source.
- `data/bo_train_pool_10k.pkl` (709 MB) — concave source.
- `data/reward_heatmaps_exp_k10_inside.npy_chunks/pair_NNNNN.npy` — pre-computed brute-force reward heatmaps for the convex pairs (used as ground-truth `r*`).

Concave brute-force is computed on the fly during the report run (~7–40 s per pair, serial Shapely + IFP).

---

## 6. Reproduce the corpus and retrain (Modal)

Skip this whole section if you only want to *use* the trained model.

### 6.1 One-time Modal setup

```bash
pip install modal      # already in requirements.txt
modal token new        # opens a browser, links this machine to your Modal account
```

### 6.2 Provide the seed data

Two pkl files seed the entire pipeline. Neither is in the repo (they're 0.5–0.7 GB each and regenerating them needs scripts not yet published):

- `data/bc_snapshot_raster128.pkl` — 12 000 convex–convex pairs with WKT polygons.
- `data/bo_train_pool_10k.pkl` — 10 000 (concave-fs, convex-part) pairs with WKT polygons.

Upload them once to your Modal volume:

```bash
modal volume create nestingrl-data            # if not already created
modal volume put nestingrl-data data/bc_snapshot_raster128.pkl /bc_snapshot_raster128.pkl
modal volume put nestingrl-data data/bo_train_pool_10k.pkl /bo_train_pool_10k.pkl
```

Check what's on the volume:

```bash
modal run modal_check_volume.py
```

### 6.3 Exhaustive reward-heatmap precompute (concave half)

```bash
# 200 chunks × 50 pairs × 36 thetas × IFP-pixel Shapely scoring.
# ~9 min wall on Modal (200 containers × 8 cores). ~$4 spend at $0.0473/core/hr.
modal run modal_concave_precompute.py --chunk-size 50 --n-pairs 10000

# Smoke test with only 200 pairs first:
modal run modal_concave_precompute.py --chunk-size 50 --n-pairs 200
```

Outputs land in the Modal volume at `/concave_reward_chunks/pair_NNNNN.npy` and `/concave_rot_part_masks/pair_NNNNN.npy`. The script is **resumable** — re-running skips pairs that already have both output files.

### 6.4 Combine convex + concave into a single training pkl

```bash
modal run modal_build_combined.py
```

This stitches the convex chunks + concave chunks + their rotated-part masks into `combined.pkl` on the volume.

### 6.5 Train

```bash
modal run modal_train_perthet.py
```

~30 min on A100-80GB at batch 256, cosine LR 3e-4 → 0, 8 000 steps, soft cross-entropy on the normalized reward heatmap with D2 augmentation. Val recovery reaches **0.73** on the 2 200-pair val split (1 200 convex + 1 000 concave).

The checkpoint is written to the Modal volume; download it with:

```bash
modal volume get nestingrl-data /checkpoints/perthet_combined/final.pt \
                                ./checkpoints/perthet_combined/final.pt
```

---

## 7. File reference

| What | Where |
|---|---|
| Trained checkpoint | `checkpoints/perthet_combined/final.pt` |
| Model architecture | `src/models/neural_bo_policy.py` — `_SmallUNet` (3.13 M params) |
| Inference wrapper | `src/inference/placement.py` — `PlacementModel` |
| Reward function | `src/geometry/rewards.py` — `compute_reward_exp` |
| IFP (handles concave fs) | `src/geometry/ifp.py` — `compute_ifp_exact` |
| Random polygon utilities | `src/geometry/polygons.py` |
| Rasterizer (Shapely → 128×128 mask) | `scripts/rasterize_ifp_union.py` |
| Geometry tests | `tests/test_geometry.py` |
| Zero-data inference demo | `scripts/demo.py` |
| Validation smoke test | `scripts/smoke_refine.py` |
| Local training (assumes precomputed pkl) | `scripts/train_perthet.py` |
| Modal training | `modal_train_perthet.py` |
| Modal concave heatmap precompute | `modal_concave_precompute.py` |
| Modal combined-pkl build | `modal_build_combined.py` |
| Report generator (LaTeX → PDF) | `scripts/generate_placement_report.py` |

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: src` | You're not at the repo root. Run all `python -m ...` commands from the directory containing `src/`. |
| `RuntimeError: CUDA out of memory` on Windows when loading the checkpoint | Always `map_location='cpu'` before moving to GPU. `PlacementModel.__init__` already does this. |
| `pdflatex: command not found` | Install MiKTeX (Windows) or TeX Live (Linux/macOS), then reopen your shell. |
| Demo prints a low or zero reward | The random pair may be near-degenerate. Try a different `--seed`. |
| Modal job stalls in "pkl loading" | Confirm the volume actually has the two seed pkls (`modal volume ls nestingrl-data`). |
| `import pyclipper` fails on macOS | `pip install pyclipper` may need build tools; `brew install gcc` then retry. |
| Smoke test errors with FileNotFoundError on `data/bc_snapshot_raster128.pkl` | You don't have the validation pkl. Use `scripts/demo.py` instead, or obtain the pkl separately. |
