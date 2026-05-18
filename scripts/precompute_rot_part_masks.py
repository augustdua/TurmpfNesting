"""
Precompute rotated part masks for every (pair, theta) so the supervised
training doesn't have to call shp_rotate + rasterize_polygon per batch.

Output: <out-dir>/pair_NNNNN.npy of shape (36, 128, 128) uint8.

Usage:
  python -m scripts.precompute_rot_part_masks \
      --in-data data/bc_snapshot_raster128_ifp_theta36.pkl \
      --out-dir data/rot_part_masks_theta36 \
      --workers 8
"""
import argparse
import os
import pickle
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.wkt import loads as wkt_loads
from shapely.affinity import translate as shp_translate, rotate as shp_rotate

from scripts.rasterize_ifp_union import rasterize_polygon


def _precompute_one(args):
    pair_idx, part_wkt, thetas, res, out_path = args
    if os.path.exists(out_path):
        return pair_idx, 'skipped'
    part = wkt_loads(part_wkt)
    cx, cy = part.centroid.coords[0]
    part_centered = shp_translate(part, -cx, -cy)
    masks = np.zeros((len(thetas), res, res), dtype=np.uint8)
    for t_idx, theta in enumerate(thetas):
        rotated = shp_rotate(part_centered, float(theta), origin=(0.0, 0.0),
                             use_radians=False)
        masks[t_idx] = rasterize_polygon(rotated, res)
    np.save(out_path, masks)
    return pair_idx, 'done'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-data', default='data/bc_snapshot_raster128_ifp_theta36.pkl')
    ap.add_argument('--out-dir', default='data/rot_part_masks_theta36')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.in_data} ...", flush=True)
    with open(args.in_data, 'rb') as f:
        records = pickle.load(f)
    n = len(records)
    print(f"  {n} pairs", flush=True)

    res = records[0]['fs_mask'].shape[0]
    jobs = []
    for i, r in enumerate(records):
        out_path = os.path.join(args.out_dir, f'pair_{i:05d}.npy')
        jobs.append((i, r['part_poly_wkt'], list(r['ifp_thetas']), res, out_path))

    print(f"Precomputing {n} pair files to {args.out_dir} (workers={args.workers}) ...",
          flush=True)
    t0 = time.time()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    n_done = 0; n_skipped = 0
    for pair_idx, status in pool.map(_precompute_one, jobs, chunksize=8):
        if status == 'skipped':
            n_skipped += 1
        else:
            n_done += 1
        if (n_done + n_skipped) % 500 == 0:
            print(f"  {n_done + n_skipped}/{n}  done={n_done} skipped={n_skipped}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    pool.shutdown(wait=True)
    print(f"Done. {n_done} written, {n_skipped} skipped (already existed). "
          f"{time.time() - t0:.0f}s total.", flush=True)


if __name__ == '__main__':
    main()
