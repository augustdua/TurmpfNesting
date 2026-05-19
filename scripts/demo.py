"""
Self-contained inference demo — no external data files needed.

Generates a random convex free-space polygon and a smaller convex part on the
fly, normalizes both into the model's [-1, 1] coordinate space, runs the
trained checkpoint, and prints the predicted (theta, x, y, reward).

Usage:
    python -m scripts.demo
    python -m scripts.demo --device cpu --seed 7 --area-ratio 0.25
"""
import argparse
import numpy as np
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate

from src.geometry.polygons import random_convex_polygon
from src.inference.placement import PlacementModel


def normalize_to_unit(poly, target_radius=0.9):
    """Center poly at origin and scale so its farthest vertex sits at target_radius."""
    cx, cy = poly.centroid.coords[0]
    poly = shp_translate(poly, -cx, -cy)
    xs, ys = poly.exterior.xy
    r_max = float(max(np.hypot(np.asarray(xs), np.asarray(ys))))
    if r_max > 0:
        s = target_radius / r_max
        poly = shp_scale(poly, xfact=s, yfact=s, origin=(0.0, 0.0))
    return poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/perthet_combined/final.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--area-ratio", type=float, default=0.30,
                    help="Target part_area / fs_area.")
    ap.add_argument("--refine-pixels", type=int, default=10)
    ap.add_argument("--refine-thetas", type=int, default=2)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("Loading model ...", flush=True)
    pm = PlacementModel(ckpt=args.ckpt, device=args.device)
    print(f"  device = {pm.device}", flush=True)

    print("Generating random fs + part ...", flush=True)
    fs = normalize_to_unit(random_convex_polygon(n_vertices=8), target_radius=0.9)
    part_raw = normalize_to_unit(random_convex_polygon(n_vertices=6),
                                 target_radius=0.5)
    s = (args.area_ratio * fs.area / part_raw.area) ** 0.5
    part = shp_scale(part_raw, xfact=s, yfact=s, origin=(0.0, 0.0))
    print(f"  fs.area = {fs.area:.3f},  part.area = {part.area:.3f},  "
          f"ratio = {part.area / fs.area:.3f}", flush=True)

    print("Running placement (model + Shapely refinement) ...", flush=True)
    theta, x, y, reward = pm.place(
        fs, part,
        refine_pixels=args.refine_pixels,
        refine_thetas=args.refine_thetas,
    )

    print("\n=== Result ===")
    print(f"  theta  = {theta:6.1f} deg")
    print(f"  (x, y) = ({x:+.3f}, {y:+.3f})  (in [-1, 1])")
    print(f"  reward = {reward:.3f}")


if __name__ == "__main__":
    main()
