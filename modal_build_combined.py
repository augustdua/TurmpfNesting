"""
Build combined training pkl (convex 12k + concave 10k = 22k pairs) and
renumber concave rot_part masks into the shared dir.

All steps run on the Modal volume — no local upload needed.

Inputs (on /vol):
  hier_training_data_soft.pkl          12k convex pairs w/ heatmap_fp16
  bo_train_pool_10k.pkl                10k concave-fs pairs (fs/part wkt + masks)
  concave_reward_chunks/pair_NNNNN.npy 10k reward chunks (float32, (36,128,128))
  concave_rot_part_masks/pair_NNNNN.npy
  rot_part_masks_theta36/pair_NNNNN.npy  12k convex rot_part masks (already there)

Outputs (on /vol):
  hier_training_data_combined.pkl                            22k pairs combined
  rot_part_masks_theta36/pair_{12000..21999}.npy              concave masks renumbered

Run:
  modal run modal_build_combined.py
"""
import os
import modal

app = modal.App("nestingrl-build-combined")

_HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy>=2.0,<3")
)

volume = modal.Volume.from_name("nestingrl-data", create_if_missing=True)


@app.function(
    image=image,
    cpu=8.0,
    memory=98304,
    timeout=7200,
    volumes={"/vol": volume},
)
def build_combined(
    concave_train_count: int = 9000,
    concave_val_count: int = 1000,
    convex_offset: int = 12000,
):
    """
    1. Copy concave_rot_part_masks/pair_NNNNN.npy -> rot_part_masks_theta36/pair_(NNNNN+12000).npy.
    2. Load convex soft pkl + 10k concave pairs.
    3. For each concave pair, load reward chunk, build a record matching the
       convex pkl schema (fs_mask, part_mask, heatmap_fp16, orig_idx).
    4. Concave train [0..9000), val [9000..10000).
    5. Combined train = convex_train + concave_train (19800 records).
       Combined val   = convex_val  + concave_val  (2200 records).
    6. Save to /vol/hier_training_data_combined.pkl.
    """
    import os
    import pickle
    import shutil
    import time
    import numpy as np

    print(f"[combined-build] start", flush=True)
    t0 = time.time()

    # --- Step 1: SKIP file copy (slow on Modal volume).
    # Concave rot_part files stay in /vol/concave_rot_part_masks/. We store
    # the explicit path per record below so the trainer reads in place.
    convex_rot_dir = "/vol/rot_part_masks_theta36"
    concave_rot_dir = "/vol/concave_rot_part_masks"
    n_convex_rot = len(os.listdir(convex_rot_dir))
    n_concave_rot = len(os.listdir(concave_rot_dir))
    print(f"  rot_part dirs OK: convex={n_convex_rot}, concave={n_concave_rot}",
          flush=True)

    # --- Step 2: load convex soft pkl ---
    print(f"  loading convex soft pkl ...", flush=True)
    t = time.time()
    with open("/vol/hier_training_data_soft.pkl", "rb") as f:
        convex = pickle.load(f)
    print(f"    convex: train={len(convex['train'])}  val={len(convex['val'])}  "
          f"({time.time()-t:.0f}s)", flush=True)

    # Add explicit rot_part_path to each convex record (in-place).
    for rec in convex["train"] + convex["val"]:
        rec["rot_part_path"] = os.path.join(
            convex_rot_dir, f"pair_{rec['orig_idx']:05d}.npy")

    # --- Step 3: load concave bo pkl (masks + wkts) ---
    print(f"  loading bo_train_pool_10k.pkl ...", flush=True)
    t = time.time()
    with open("/vol/bo_train_pool_10k.pkl", "rb") as f:
        concave_raw = pickle.load(f)
    print(f"    concave raw: {len(concave_raw)} pairs ({time.time()-t:.0f}s)",
          flush=True)

    # --- Step 4: build concave records (parallel chunk loads) ---
    chunks_dir = "/vol/concave_reward_chunks"
    n_target = min(concave_train_count + concave_val_count, len(concave_raw))
    print(f"  building {n_target} concave records "
          f"(threadpool=64 for chunk loads) ...", flush=True)
    t = time.time()

    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    progress_lock = Lock()
    progress = [0]

    def _build_one(i):
        chunk_path = os.path.join(chunks_dir, f"pair_{i:05d}.npy")
        if not os.path.exists(chunk_path):
            return None
        heatmap = np.load(chunk_path).astype(np.float32)
        argmax_flat = int(heatmap.reshape(-1).argmax())
        argmax_reward = float(heatmap.reshape(-1)[argmax_flat])
        if argmax_reward <= 0:
            return None
        rec = concave_raw[i]
        entry = {
            "fs_mask": rec["fs_mask"].astype(np.uint8),
            "part_mask": rec["part_mask"].astype(np.uint8),
            "argmax_flat": argmax_flat,
            "argmax_reward": argmax_reward,
            "heatmap_fp16": heatmap.astype(np.float16),
            "orig_idx": convex_offset + i,
            "rot_part_path": os.path.join(concave_rot_dir,
                                          f"pair_{i:05d}.npy"),
        }
        with progress_lock:
            progress[0] += 1
            if progress[0] % 1000 == 0:
                elapsed = time.time() - t
                print(f"    built {progress[0]}/{n_target} "
                      f"({elapsed:.0f}s, {progress[0]/elapsed:.1f}/s)",
                      flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=64) as pool:
        results = list(pool.map(_build_one, range(n_target)))
    concave_records = [r for r in results if r is not None]
    n_invalid = n_target - len(concave_records)
    print(f"  built {len(concave_records)} concave records "
          f"({n_invalid} skipped, {time.time()-t:.0f}s)", flush=True)

    # --- Step 5: split & combine ---
    n_train = min(concave_train_count, len(concave_records))
    concave_train = concave_records[:n_train]
    concave_val = concave_records[n_train:n_train + concave_val_count]
    combined = {
        "train": convex["train"] + concave_train,
        "val": convex["val"] + concave_val,
        "meta": {
            **convex.get("meta", {}),
            "n_train_convex": len(convex["train"]),
            "n_train_concave": len(concave_train),
            "n_val_convex": len(convex["val"]),
            "n_val_concave": len(concave_val),
            "convex_offset": convex_offset,
        },
    }
    print(f"  combined: train={len(combined['train'])}  "
          f"val={len(combined['val'])}", flush=True)

    # --- Step 6: save ---
    out = "/vol/hier_training_data_combined.pkl"
    tmp = out + ".tmp"
    print(f"  saving {out} ...", flush=True)
    t = time.time()
    with open(tmp, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out)
    sz_gb = os.path.getsize(out) / 1e9
    print(f"  saved.  {sz_gb:.2f} GB  ({time.time()-t:.0f}s)", flush=True)

    volume.commit()
    elapsed_total = time.time() - t0
    print(f"[combined-build] DONE in {elapsed_total:.0f}s", flush=True)

    return {
        "n_train": len(combined["train"]),
        "n_val": len(combined["val"]),
        "size_gb": sz_gb,
        "elapsed": elapsed_total,
    }


@app.local_entrypoint()
def main():
    r = build_combined.remote()
    print(f"\nResult: {r}")
