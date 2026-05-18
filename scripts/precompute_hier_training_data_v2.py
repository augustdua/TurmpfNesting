"""
v2: like precompute_hier_training_data.py, but stores the FULL reward heatmap
(36, 128, 128) as float16 for every record (train + val). Enables soft-label
training (KL / soft CE over the normalized heatmap).

Size budget per pair: 36 * 128 * 128 * 2 = 1.18 MB.
12000 pairs ~= 14.2 GB. Fits in RAM, fits in float16 dynamic range
(exp(k=10 * reward) in [0, 22026] << fp16 max 65504).

Output: data/hier_training_data_soft.pkl  (default)
"""
import argparse
import os
import pickle
import time
import numpy as np

RES = 128
N_THETA = 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifp-pkl", default="data/bc_snapshot_raster128_ifp_theta36.pkl")
    ap.add_argument("--chunks-dir", default="data/reward_heatmaps_exp_k10_inside.npy_chunks")
    ap.add_argument("--out", default="data/hier_training_data_soft.pkl")
    ap.add_argument("--train-end", type=int, default=10800)
    ap.add_argument("--val-end", type=int, default=12000)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="If set, cap total pairs (after train/val cap). "
                         "Use for local smoke tests.")
    args = ap.parse_args()

    print(f"Loading {args.ifp_pkl} ...", flush=True)
    t = time.time()
    with open(args.ifp_pkl, "rb") as f:
        records = pickle.load(f)
    print(f"  {len(records)} pairs in {time.time()-t:.1f}s", flush=True)

    n_total = min(args.val_end, len(records))
    train_idxs = list(range(0, min(args.train_end, n_total)))
    val_idxs = list(range(args.train_end, n_total))

    if args.max_pairs is not None:
        # Keep the train/val ratio roughly 9:1 for a quick smoke test.
        frac = args.max_pairs / n_total
        cap_train = max(1, int(len(train_idxs) * frac))
        cap_val = max(1, int(len(val_idxs) * frac))
        train_idxs = train_idxs[:cap_train]
        val_idxs = val_idxs[:cap_val]
        print(f"  --max-pairs={args.max_pairs}: train={cap_train}, val={cap_val}",
              flush=True)

    print(f"Split: train={len(train_idxs)}, val={len(val_idxs)}", flush=True)

    train_records = []
    val_records = []
    n_invalid = 0
    n_total_processed = 0
    t0 = time.time()

    for split_name, idxs, dst in [("train", train_idxs, train_records),
                                  ("val", val_idxs, val_records)]:
        for k, i in enumerate(idxs):
            rec = records[i]
            fs_mask = rec["fs_mask"].astype(np.uint8)
            part_mask = rec["part_mask"].astype(np.uint8)

            chunk_path = os.path.join(args.chunks_dir, f"pair_{i:05d}.npy")
            heatmap = np.load(chunk_path).astype(np.float32)   # (36, 128, 128)
            argmax_flat = int(heatmap.reshape(-1).argmax())
            argmax_reward = float(heatmap.reshape(-1)[argmax_flat])
            if argmax_reward <= 0:
                n_invalid += 1
                continue

            entry = {
                "fs_mask": fs_mask,
                "part_mask": part_mask,
                "argmax_flat": argmax_flat,
                "argmax_reward": argmax_reward,
                "heatmap_fp16": heatmap.astype(np.float16),
                "orig_idx": i,
            }
            dst.append(entry)
            n_total_processed += 1

            if (k + 1) % 1000 == 0 or k + 1 == len(idxs):
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed
                gb = n_total_processed * (1.18e-3 + 2 * 128 * 128 * 1e-9)
                print(f"  [{split_name}] {k+1}/{len(idxs)}  "
                      f"({rate:.1f} pairs/s, ~{gb:.1f} GB so far)",
                      flush=True)

    print(f"Filtered {n_invalid} pairs with argmax_reward <= 0", flush=True)

    out = {
        "train": train_records,
        "val": val_records,
        "meta": {
            "n_theta": N_THETA,
            "res": RES,
            "tile": 16,
            "n_train": len(train_records),
            "n_val": len(val_records),
            "heatmap_dtype": "float16",
            "heatmap_temperature_k": 10,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    print(f"Writing {args.out} ...", flush=True)
    t = time.time()
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, args.out)
    size_gb = os.path.getsize(args.out) / 1e9
    print(f"  done.  size={size_gb:.2f} GB  write_time={time.time()-t:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
