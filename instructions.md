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

`scripts/smoke_refine.py` evaluates the model on `N` random pairs from the convex validation split. It needs `data/bc_snapshot_raster128.pkl` (482 MB), which is not in the repo. Generate it (and the concave pkl you'll need for §5) with:

```bash
python -m scripts.generate_seed_pkls
```

That's the one command for all seed data — see §5 for what it produces. Then:

```bash
python -m scripts.smoke_refine                                    # 5 random val pairs
python -m scripts.smoke_refine --n 25 --device cpu                # bigger sample
python -m scripts.smoke_refine --refine-pixels 0 --refine-thetas 0  # model-only
```

If you only want a quick sanity check and don't care about the val pairs, use `scripts/demo.py` — it generates a random pair on the fly and exercises the same code path with no data files at all.

---

## 5. Generate the PDF report

The 4 MB LaTeX report (12 convex + 6 concave example pages, full methodology, summary) is already committed at `visualizations/report/placement_pipeline_report.pdf`. To rebuild it from scratch:

```bash
# 1. Regenerate the two seed pkls (~1.2 GB combined, a few minutes).
#    Only needs the geometry primitives in src/ — no model, no Modal.
python -m scripts.generate_seed_pkls

# 2. Render the report.
python -m scripts.generate_placement_report                       # default (12 + 6)
python -m scripts.generate_placement_report --n-convex 4 --n-concave 2  # quicker
python -m scripts.generate_placement_report --no-compile          # emit .tex only
```

**Requirements for this step:**
- `pdflatex` on PATH (install MiKTeX on Windows or TeX Live elsewhere; verify with `pdflatex --version`). Use `--no-compile` to skip this and produce only the `.tex` + figures.
- `data/bc_snapshot_raster128.pkl` (482 MB) — convex source. Regenerate with `python -m scripts.generate_seed_pkls`.
- `data/bo_train_pool_10k.pkl` (709 MB) — concave source. Same command regenerates this.
- `data/reward_heatmaps_exp_k10_inside.npy_chunks/pair_NNNNN.npy` and `data/reward_heatmaps_concave_exp_k10_inside.npy_chunks/pair_NNNNN.npy` — **optional cache of precomputed BF heatmaps**. The report falls back to on-the-fly Shapely + IFP per picked pair (~7–40 s each) when chunks are missing. Precompute them once with `scripts/precompute_bf.py` (see §5.1 below) if you plan to render many pages or rerun the report.

> **Note on `--n-convex` / `--n-concave`:** these are sample counts, not pool sizes. Defaults are `12` and `6`. The numbers `12000` / `10000` you'll see in the source are just the *upper bounds* of the validation pool the picks are drawn from. Pass them as sample counts to render the full corpus — see §5.2.

### 5.1 Precompute the BF heatmap cache (optional, for many-page renders)

`scripts/precompute_bf.py` fills the convex and concave BF chunk directories in parallel using `ProcessPoolExecutor`. Each chunk is a `(36, 128, 128) float32` `.npy` (~590 KB). The script is resumable — existing chunk files are skipped on re-run.

```bash
# Full corpus, all CPU cores. ~5 h convex + ~4 h concave on 16 cores
# (~25 s mean per pair, embarrassingly parallel).
python -m scripts.precompute_bf --kind convex
python -m scripts.precompute_bf --kind concave

# Subset, custom worker count:
python -m scripts.precompute_bf --kind convex --workers 8 --start 0 --end 500

# Recompute existing chunks:
python -m scripts.precompute_bf --kind convex --force
```

Output sizes at full corpus: ~7 GB (12 000 convex chunks) and ~5.8 GB (10 000 concave chunks). Once these directories exist, every subsequent report run uses the cached `r*` heatmaps and the per-pair cost drops to a fraction of a second (just matplotlib + Shapely refinement).

### 5.2 Render the full corpus (12 000 convex + 10 000 concave)

This is the path to a 22 000-page report. **Order matters: precompute the BF cache first, otherwise each picked pair recomputes BF on the fly and the run takes days.**

```bash
# 1. Seed pkls (~few minutes).
python -m scripts.generate_seed_pkls

# 2. BF cache (~10 h on 16 cores; resumable, run overnight).
python -m scripts.precompute_bf --kind convex
python -m scripts.precompute_bf --kind concave

# 3. Render. --no-compile emits .tex + PNGs only; pdflatex would
#    almost certainly OOM on a 22 000-page document, and the .tex +
#    PNGs are what you actually consume programmatically anyway. Plan
#    for tens of GB of PNGs in visualizations/report/figs/.
python -m scripts.generate_placement_report \
    --n-convex 12000 --n-concave 10000 --no-compile
```

The val pool caps `--n-convex` at `CONVEX_VAL_END = 12000` and `--n-concave` at `CONCAVE_VAL_END = 10000`; values above those are silently clamped. Cost breakdown for the full 22 000-page render assuming the BF cache is in place: BF lookup is essentially free (a `np.load` per pair), matplotlib renders dominate at ~2–5 s per page on a typical desktop → ~12–30 hours of single-threaded figure rendering. Run it in `tmux`/`screen` (Linux/macOS) or a long-lived PowerShell window (Windows).

---

## 6. Reproduce the corpus and retrain (Modal)

Skip this whole section if you only want to *use* the trained model.

### 6.1 One-time Modal setup

```bash
pip install modal      # already in requirements.txt
modal token new        # opens a browser, links this machine to your Modal account
```

### 6.2 Provide the seed data

Two pkl files seed the entire pipeline. Neither is in the repo (they're 0.5–0.7 GB each):

- `data/bc_snapshot_raster128.pkl` — 12 000 convex–convex pairs with WKT polygons.
- `data/bo_train_pool_10k.pkl` — 10 000 (concave-fs, convex-part) pairs with WKT polygons.

Regenerate them locally (a few minutes, no Modal needed for this step):

```bash
python -m scripts.generate_seed_pkls          # writes both pkls to data/
```

Then upload to your Modal volume:

```bash
modal volume create nestingrl-data            # if not already created
modal volume put nestingrl-data data/bc_snapshot_raster128.pkl /bc_snapshot_raster128.pkl
modal volume put nestingrl-data data/bo_train_pool_10k.pkl /bo_train_pool_10k.pkl
```

> The regenerated pkls are not byte-identical to the originals used during the internship (they use a different RNG seed sequence), but the distribution and format match. The downstream precompute, train, and report scripts all run against them with no code changes. Validation indices in the report script reference positions in the file, so example pair IDs in the resulting PDF will differ.

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
| Local BF heatmap precompute (parallel) | `scripts/precompute_bf.py` |
| BF heatmap module (used by both report + precompute) | `src/geometry/brute_force.py` |
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
| `FileNotFoundError: data/bc_snapshot_raster128.pkl` (smoke test or report) | The seed pkl isn't on disk. Regenerate it locally with `python -m scripts.generate_seed_pkls` (or pass `--convex-source`/`--source` to point at an existing copy). For a quick model sanity check with no data files at all, use `scripts/demo.py`. |
| `FileNotFoundError: data/bo_train_pool_10k.pkl` (report) | Same fix — `python -m scripts.generate_seed_pkls` regenerates both seed pkls. |
| `FileNotFoundError: data/reward_heatmaps_exp_k10_inside.npy_chunks/pair_NNNNN.npy` | Stale message from an older build. The report script now falls back to computing convex BF on the fly when chunks are missing. Pull the latest `main` if you still hit this. To cache the chunks (much faster for large renders) run `python -m scripts.precompute_bf --kind convex`. |
| Many-page render is taking forever | You're hitting the on-the-fly BF path for every pair. Precompute the cache first: `python -m scripts.precompute_bf --kind convex && python -m scripts.precompute_bf --kind concave`. Resumable — safe to Ctrl-C and re-run. |
| `pdflatex` OOMs or stalls compiling the full 22 000-page report | LaTeX wasn't built for documents this large. Always pass `--no-compile` for renders past a few hundred pages — you'll still get the per-pair PNGs in `visualizations/report/figs/` and the `.tex` source. If you actually need a bound PDF, run the script multiple times with different `--seed` values into separate output dirs (`--out-dir`), then compile each subset's `.tex` independently. |
| Precompute job dies with `OSError: [Errno 28] No space left on device` | Each chunk is ~590 KB; the full convex + concave cache is ~13 GB. Free up disk or point `--out-dir` at a different volume. |
