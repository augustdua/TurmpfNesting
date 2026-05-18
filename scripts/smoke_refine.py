"""
Quick smoke test: run perthet model with and without Shapely refinement
on a handful of convex val pairs. Prints rewards side-by-side.

Usage:
    python -m scripts.smoke_refine                       # 5 random val pairs
    python -m scripts.smoke_refine --n 10 --refine-pixels 10 --refine-thetas 2
"""
import argparse
import pickle
import time
import numpy as np
from shapely.wkt import loads as wkt_loads

from src.inference.placement import PlacementModel

CONVEX_VAL_START = 10800
CONVEX_VAL_END = 12000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/perthet_combined/final.pt")
    ap.add_argument("--source", default="data/bc_snapshot_raster128.pkl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--refine-pixels", type=int, default=10)
    ap.add_argument("--refine-thetas", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading model ...", flush=True)
    pm = PlacementModel(ckpt=args.ckpt, device=args.device)
    print(f"  device={pm.device}", flush=True)

    print(f"Loading {args.source} ...", flush=True)
    with open(args.source, "rb") as f:
        records = pickle.load(f)
    print(f"  {len(records)} pairs total", flush=True)

    rng = np.random.default_rng(args.seed)
    val_idxs = list(range(CONVEX_VAL_START, min(CONVEX_VAL_END, len(records))))
    picks = rng.choice(val_idxs, size=min(args.n, len(val_idxs)), replace=False)

    print(f"\n{'idx':>6} | {'no_refine':>10} | {'refined':>10} | "
          f"{'delta':>10} | {'t_noref(s)':>10} | {'t_ref(s)':>10}", flush=True)
    print("-" * 80, flush=True)

    deltas = []
    for idx in picks:
        rec = records[int(idx)]
        fs = wkt_loads(rec["fs_poly_wkt"])
        part = wkt_loads(rec["part_poly_wkt"])

        t = time.time()
        _, _, _, r_noref = pm.place(fs, part, refine_pixels=0, refine_thetas=0)
        t_noref = time.time() - t

        t = time.time()
        _, _, _, r_ref = pm.place(fs, part,
                                  refine_pixels=args.refine_pixels,
                                  refine_thetas=args.refine_thetas)
        t_ref = time.time() - t

        delta = r_ref - r_noref
        deltas.append(delta)
        print(f"{int(idx):>6} | {r_noref:>10.4f} | {r_ref:>10.4f} | "
              f"{delta:>+10.4f} | {t_noref:>10.3f} | {t_ref:>10.3f}",
              flush=True)

    deltas = np.array(deltas)
    print("-" * 80, flush=True)
    print(f"\nmean delta (refined - no_refine) = {deltas.mean():+.4f}", flush=True)
    print(f"pairs where refine helped: {int((deltas > 0).sum())}/{len(deltas)}",
          flush=True)


if __name__ == "__main__":
    main()
