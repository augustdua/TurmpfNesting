"""Parallel precompute of brute-force reward-heatmap chunks.

For each (fs, part) pair in a seed pkl, writes a (36, 128, 128) float32 .npy
chunk that `scripts/generate_placement_report.py` reads back as the
ground-truth `r*` overlay. The report falls back to on-the-fly computation
when chunks are missing, so this step is only needed when you want to render
many pages (e.g. the full 12 000 convex + 10 000 concave corpus).

Output paths (must match the report's `BF_CHUNKS_DIR{,_CONCAVE}`):
    --kind convex  -> data/reward_heatmaps_exp_k10_inside.npy_chunks/pair_NNNNN.npy
    --kind concave -> data/reward_heatmaps_concave_exp_k10_inside.npy_chunks/pair_NNNNN.npy

Resumable: pairs whose chunk already exists are skipped. Pass --force to redo.

Cost estimate (Shapely + IFP, single thread): ~7-40 s per pair, ~25 s mean.
On 16 cores -> ~5 h for the full 12 000 convex set, ~4 h for 10 000 concave.

Usage:
    # All convex pairs, all cores:
    python -m scripts.precompute_bf --kind convex

    # All concave pairs, 8 workers:
    python -m scripts.precompute_bf --kind concave --workers 8

    # Just a slice (debug):
    python -m scripts.precompute_bf --kind convex --start 0 --end 50

    # Custom source / output:
    python -m scripts.precompute_bf --kind convex \\
        --source data/my_convex.pkl --out-dir data/my_bf_chunks
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from shapely.wkt import loads as wkt_loads

from src.geometry.brute_force import compute_bf_array

KIND_DEFAULTS = {
    "convex": dict(
        source="data/bc_snapshot_raster128.pkl",
        out_dir="data/reward_heatmaps_exp_k10_inside.npy_chunks",
    ),
    "concave": dict(
        source="data/bo_train_pool_10k.pkl",
        out_dir="data/reward_heatmaps_concave_exp_k10_inside.npy_chunks",
    ),
}


def _worker(args):
    """Compute one pair's BF heatmap and save to disk. Returns (idx, status, secs)."""
    idx, fs_wkt, part_wkt, out_path, k = args
    if os.path.exists(out_path):
        return idx, "skipped", 0.0
    t0 = time.time()
    try:
        fs = wkt_loads(fs_wkt)
        part = wkt_loads(part_wkt)
        bf = compute_bf_array(fs, part, k=k)
        # Atomic write: write to tmp via file object (so np.save doesn't
        # auto-append .npy to the tmp name), then rename. Prevents partial
        # files if the run is interrupted.
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, bf)
        os.replace(tmp, out_path)
        return idx, "done", time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return idx, f"error: {exc!r}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(KIND_DEFAULTS.keys()))
    ap.add_argument("--source", default=None,
                    help="Override the input pkl path.")
    ap.add_argument("--out-dir", default=None,
                    help="Override the output chunks directory.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None,
                    help="Exclusive. Defaults to len(records).")
    ap.add_argument("--k", type=float, default=10.0,
                    help="Reward steepness; must match the report's --k.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the chunk file already exists.")
    args = ap.parse_args()

    defaults = KIND_DEFAULTS[args.kind]
    source = args.source or defaults["source"]
    out_dir = args.out_dir or defaults["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {source} ...", flush=True)
    with open(source, "rb") as f:
        records = pickle.load(f)
    print(f"  {len(records)} pairs", flush=True)

    end = args.end if args.end is not None else len(records)
    indices = list(range(args.start, min(end, len(records))))

    tasks = []
    pre_skipped = 0
    for idx in indices:
        out_path = os.path.join(out_dir, f"pair_{idx:05d}.npy")
        if (not args.force) and os.path.exists(out_path):
            pre_skipped += 1
            continue
        rec = records[idx]
        tasks.append((idx, rec["fs_poly_wkt"], rec["part_poly_wkt"],
                      out_path, args.k))

    print(f"Workers: {args.workers}", flush=True)
    print(f"Slice: [{args.start}, {min(end, len(records))})  ({len(indices)} pairs)",
          flush=True)
    print(f"Already on disk: {pre_skipped}.  To compute: {len(tasks)}.",
          flush=True)
    if not tasks:
        print("Nothing to do.", flush=True)
        return

    t_run = time.time()
    n_done = 0
    n_err = 0
    sum_secs = 0.0
    log_every = max(1, len(tasks) // 50)  # ~50 progress lines

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            idx, status, secs = fut.result()
            n_done += 1
            sum_secs += secs
            if status.startswith("error"):
                n_err += 1
                print(f"  pair {idx}: {status}", flush=True)
            if n_done % log_every == 0 or n_done == len(tasks):
                elapsed = time.time() - t_run
                rate = n_done / max(elapsed, 1e-9)
                remaining = (len(tasks) - n_done) / max(rate, 1e-9)
                mean_secs = sum_secs / max(n_done, 1)
                print(
                    f"  {n_done:5d}/{len(tasks)}  "
                    f"wall {elapsed/60:6.1f} min  "
                    f"mean {mean_secs:5.1f} s/pair  "
                    f"eta {remaining/60:6.1f} min  "
                    f"errors {n_err}",
                    flush=True,
                )

    print(
        f"Done. {n_done - n_err} written, {n_err} errors, "
        f"{pre_skipped} pre-existing.  Out dir: {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
