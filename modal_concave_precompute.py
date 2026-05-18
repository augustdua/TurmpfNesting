"""
Modal CPU job: compute exhaustive reward heatmaps + rot_part masks for the
10k concave-fs / convex-part pairs in bo_train_pool_10k.pkl.

Outputs (saved to volume):
  /vol/concave_reward_chunks/pair_NNNNN.npy    (36, 128, 128) float32
  /vol/concave_rot_part_masks/pair_NNNNN.npy   (36, 128, 128) uint8

Run:
  modal run modal_concave_precompute.py --chunk-size 50 --n-pairs 10000
  modal run modal_concave_precompute.py --chunk-size 50 --n-pairs 200  # smoke
"""
import os
import sys
import modal

app = modal.App("nestingrl-concave-precompute")

_HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=2.0,<3",
        "scipy",
        "shapely>=2.0",
        "pyclipper",
        "pillow",
    )
    .add_local_dir(os.path.join(_HERE, "src"), remote_path="/root/project/src")
    .add_local_dir(os.path.join(_HERE, "scripts"), remote_path="/root/project/scripts")
)

volume = modal.Volume.from_name("nestingrl-data", create_if_missing=True)

RES = 128
N_THETA = 36


def _process_one_pair(args):
    """Worker: compute IFP + rewards + rot_part for one pair. Resumable."""
    import os
    import numpy as np
    from shapely.wkt import loads as wkt_loads
    from shapely.affinity import translate as shp_translate
    from shapely.affinity import rotate as shp_rotate

    idx, rec, k, reward_dir, rotpart_dir = args

    reward_path = os.path.join(reward_dir, f"pair_{idx:05d}.npy")
    rotpart_path = os.path.join(rotpart_dir, f"pair_{idx:05d}.npy")
    if os.path.exists(reward_path) and os.path.exists(rotpart_path):
        return idx, 0, "skip"

    sys.path.insert(0, "/root/project")
    from src.geometry.ifp import compute_ifp_exact
    from src.geometry.rewards import compute_reward_exp
    from scripts.rasterize_ifp_union import rasterize_polygon

    fs = wkt_loads(rec["fs_poly_wkt"])
    part = wkt_loads(rec["part_poly_wkt"])
    cx, cy = part.centroid.coords[0]
    part_centered = shp_translate(part, -cx, -cy)

    thetas = np.linspace(0.0, 360.0, N_THETA, endpoint=False, dtype=np.float32)
    rewards = np.zeros((N_THETA, RES, RES), dtype=np.float32)
    rot_parts = np.zeros((N_THETA, RES, RES), dtype=np.uint8)
    n_queries = 0

    for t_idx, theta in enumerate(thetas):
        # Rotated part mask
        rotated = shp_rotate(part_centered, float(theta),
                             origin=(0.0, 0.0), use_radians=False)
        rot_parts[t_idx] = rasterize_polygon(rotated, RES)

        # IFP (works for concave fs + convex part)
        try:
            ifp = compute_ifp_exact(fs, part, float(theta))
        except Exception:
            continue
        if ifp.is_empty or ifp.area < 1e-8:
            continue
        ifp_mask = rasterize_polygon(ifp, RES)
        ifp_bool = ifp_mask > 0
        if not ifp_bool.any():
            continue

        # Reward at each IFP pixel
        for row, col in np.argwhere(ifp_bool):
            x = col / (RES - 1) * 2 - 1
            y = 1 - row / (RES - 1) * 2
            try:
                r = compute_reward_exp(fs, part, x, y, float(theta), k=k)
            except Exception:
                r = 0.0
            rewards[t_idx, row, col] = r
            n_queries += 1

    np.save(reward_path, rewards)
    np.save(rotpart_path, rot_parts)
    return idx, n_queries, "done"


@app.function(
    image=image,
    cpu=8.0,
    memory=16384,
    timeout=900,
    volumes={"/vol": volume},
)
def process_chunk(start_idx: int, end_idx: int, k: float = 10.0):
    """Load pkl, slice [start, end), parallel-process via ProcessPool(8)."""
    import os
    import sys
    import pickle
    import time
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, "/root/project")

    pkl_path = "/vol/bo_train_pool_10k.pkl"
    reward_dir = "/vol/concave_reward_chunks"
    rotpart_dir = "/vol/concave_rot_part_masks"
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(rotpart_dir, exist_ok=True)

    print(f"[{start_idx}-{end_idx}] loading pkl ...", flush=True)
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        all_pairs = pickle.load(f)
    print(f"[{start_idx}-{end_idx}] loaded {len(all_pairs)} pairs "
          f"in {time.time()-t0:.1f}s", flush=True)

    args_list = [
        (i, all_pairs[i], k, reward_dir, rotpart_dir)
        for i in range(start_idx, min(end_idx, len(all_pairs)))
    ]
    t1 = time.time()
    n_done = 0
    n_skip = 0
    n_queries_total = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for idx, nq, status in pool.map(_process_one_pair, args_list):
            n_queries_total += nq
            if status == "skip":
                n_skip += 1
            else:
                n_done += 1
    elapsed = time.time() - t1
    print(f"[{start_idx}-{end_idx}] done {n_done} new, {n_skip} skip, "
          f"{n_queries_total} queries in {elapsed:.1f}s "
          f"({len(args_list)/max(elapsed,1):.1f} pairs/s)", flush=True)

    volume.commit()
    return {
        "start": start_idx, "end": end_idx,
        "n_done": n_done, "n_skip": n_skip,
        "n_queries": n_queries_total,
        "elapsed": elapsed,
    }


@app.function(image=image, volumes={"/vol": volume}, timeout=60)
def list_outputs():
    import os
    reward_dir = "/vol/concave_reward_chunks"
    rotpart_dir = "/vol/concave_rot_part_masks"
    n_r = len(os.listdir(reward_dir)) if os.path.isdir(reward_dir) else 0
    n_p = len(os.listdir(rotpart_dir)) if os.path.isdir(rotpart_dir) else 0
    print(f"concave_reward_chunks: {n_r} files", flush=True)
    print(f"concave_rot_part_masks: {n_p} files", flush=True)
    return n_r, n_p


@app.local_entrypoint()
def main(chunk_size: int = 50, n_pairs: int = 10000, k: float = 10.0):
    chunks = [(i, min(i + chunk_size, n_pairs))
              for i in range(0, n_pairs, chunk_size)]
    print(f"Spawning {len(chunks)} chunks "
          f"({n_pairs} pairs, ~{chunk_size} pairs/chunk, k={k})")
    args = [(s, e, k) for (s, e) in chunks]
    results = []
    for r in process_chunk.starmap(args):
        results.append(r)
        print(f"  chunk [{r['start']}-{r['end']}] done: "
              f"{r['n_done']} new, {r['n_skip']} skip, {r['elapsed']:.1f}s")
    total_done = sum(r["n_done"] for r in results)
    total_skip = sum(r["n_skip"] for r in results)
    total_q = sum(r["n_queries"] for r in results)
    print(f"\n=== ALL DONE.  {total_done} new, {total_skip} skipped, "
          f"{total_q:,} total Shapely queries ===")
    print("Listing outputs ...")
    n_r, n_p = list_outputs.remote()
    print(f"  reward chunks: {n_r}\n  rot_part chunks: {n_p}")
