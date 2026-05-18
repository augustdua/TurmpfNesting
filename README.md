# TurmpfNesting — Placement Network

A learned placement function that, given a free-space polygon `F` and a part polygon `P`, returns a placement `(θ, x, y)` that maximizes a Shapely-defined reward. Trained on 22 000 (convex–convex + concave-fs / convex-part) pairs. Reaches **0.73 mean val recovery** with a single 3.13M-parameter U-Net and one forward pass per rotation, pushed to near-1.0 with a small Shapely refinement step.

See [METHODOLOGY.md](METHODOLOGY.md) for the full derivation.

## Architecture (one-paragraph)

Per-(pair, θ) U-Net with 2-channel input (`fs_mask`, `rotated_part_mask`), both `128×128`. Output is a `128×128` per-pixel reward logit map for that rotation. At inference: 36 forward passes (one per `θ ∈ {0, 10, …, 350}°`), argmax over the flat `(36, 128, 128)` volume → predicted `(θ̂, x̂, ŷ)`. Optional bounded Shapely refinement (`±10` px × `±2` θ-bins = 2205 Shapely calls, ≈165 ms) closes the gap to the brute-force optimum.

## Quickstart

```bash
conda env create -f environment.yml
conda activate placement-net

# Run geometry tests
python -m tests.test_geometry

# Smoke test: run the trained model + Shapely refinement on a few random
# convex val pairs and print the reward before/after refinement.
python -m scripts.smoke_refine

# Generate the multi-page LaTeX visualization report (needs pdflatex).
python -m scripts.generate_placement_report
```

## What's here

| Concern | File |
|---|---|
| Model | `src/models/neural_bo_policy.py` (`_SmallUNet`) |
| IFP (handles concave free-space) | `src/geometry/ifp.py` (`compute_ifp_exact`) |
| Reward function | `src/geometry/rewards.py` (`compute_reward_exp`) |
| Inference (forward + refine) | `src/inference/placement.py` (`PlacementModel`) |
| Training | `scripts/train_perthet.py` |
| Modal training entrypoint | `modal_train_perthet.py` |
| Concave heatmap precompute | `modal_concave_precompute.py` |
| Combined-corpus build | `modal_build_combined.py` |

## Pre-trained checkpoint

`checkpoints/perthet_combined/final.pt` — 3.13M-parameter U-Net trained on the combined 22 k pair corpus (10 800 convex + 9 000 concave-fs train, 1 200 + 1 000 val).

## Visualization

- `visualizations/report/placement_pipeline_report.pdf` — multi-page LaTeX report: full methodology, then 12 convex + 6 concave example pages (predicted reward map vs brute-force ground-truth reward map vs refined placement, with per-pair Shapely-call counts and timings) plus an aggregate summary.
- `visualizations/sample/` *(not yet populated — see below)* — a 50-pair sample of training-data visualizations (input shapes, GT heatmap, predicted heatmap).

## Training data

The full 22 k-pair corpus is not committed (the heatmap pkls are 14+ GB). The Modal entrypoints regenerate them end-to-end:

1. `modal_concave_precompute.py` — exhaustive Shapely reward heatmaps for the 10 k concave-fs / convex-part pairs.
2. `modal_build_combined.py` — combines convex + concave into the unified training pkl.
3. `modal_train_perthet.py` — trains the U-Net (≈30 min on A100-80GB).

## License

TBD.
