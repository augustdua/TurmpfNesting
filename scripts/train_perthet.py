"""
Per-(pair, θ) placement trainer — NO tile head, NO cell head, no hierarchy.

Sample one (pair, θ) per step. Inputs: (fs_mask, rotated_part_mask).
Output: (128, 128) reward logits for that θ.
Loss: soft CE against the normalized per-θ heatmap, OR hard CE on the argmax pixel.

At inference: 36 forward passes (one per θ), stack, argmax → (θ*, r*, c*).
This is the 2-channel cousin of train_supervised_placement.py (no IFP, no
frozen heatmap UNet).

Data:
  --data: hier_training_data_soft.pkl  (provides fs_mask + heatmap_fp16 per pair)
  --rot-part-dir: dir of per-pair (36, 128, 128) rotated part masks
"""
import argparse
import os
import pickle
import sys
import time
import traceback
import numpy as np

_T0 = time.time()
_LOG_FH = None


def log(msg):
    line = f"[{time.time()-_T0:7.2f}s] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
            os.fsync(_LOG_FH.fileno())
        except Exception:
            pass


log("entered train_perthet.py")
log(f"python {sys.version.split()[0]}  cwd={os.getcwd()}")

log("importing torch ...")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    log(f"cuda device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

from torch.utils.tensorboard import SummaryWriter

from src.models.neural_bo_policy import _SmallUNet
log("imports done")

RES = 128
N_THETA = 36


# -------------------- model --------------------

class PerThetaPlacementUNet(nn.Module):
    """2-channel (fs, rot_part) -> 1-channel (per-pixel reward logits)."""

    def __init__(self, base=32):
        super().__init__()
        self.unet = _SmallUNet(in_ch=2, base=base, out_ch=1)

    def forward(self, fs, rp):
        x = torch.stack([fs, rp], dim=1)            # (B, 2, 128, 128)
        return self.unet(x).squeeze(1)              # (B, 128, 128)


# -------------------- dataset --------------------

class PerThetaDataset(IterableDataset):
    """Yields (fs_mask, rot_part_mask, target) for one random (pair, θ).

    target = normalized (128, 128) float32 heatmap (soft) OR int64 argmax flat (hard).
    Skips pairs/θ combos with no positive reward.
    augment: 50% H-flip & 50% V-flip applied to all of (fs, rp, target).
    """

    def __init__(self, records, rot_part_dir, seed_base=0,
                 augment=False, hard_target=False, preloaded_rot=None):
        super().__init__()
        self.records = [r for r in records if r["argmax_reward"] > 0]
        self.rot_part_dir = rot_part_dir
        self.seed_base = seed_base
        self.augment = augment
        self.hard_target = hard_target
        self.preloaded_rot = preloaded_rot   # dict[pair_idx] -> (36,128,128) np.uint8
        self._rot_cache = {}
        log(f"  PerThetaDataset: {len(self.records)} usable pairs "
            f"(filtered from {len(records)})  augment={augment}  "
            f"hard_target={hard_target}  preloaded={preloaded_rot is not None}")

    def _rot_for_rec(self, rec):
        """Returns (36, 128, 128) uint8 rot_part array for this record.
        Uses preloaded dict if available; else loads from rec['rot_part_path']
        or falls back to legacy rot_part_dir + orig_idx convention."""
        pair_idx = rec["orig_idx"]
        if self.preloaded_rot is not None and pair_idx in self.preloaded_rot:
            return self.preloaded_rot[pair_idx]
        if pair_idx not in self._rot_cache:
            p = rec.get("rot_part_path") or os.path.join(
                self.rot_part_dir, f"pair_{pair_idx:05d}.npy")
            self._rot_cache[pair_idx] = np.load(p, mmap_mode="r")
        return self._rot_cache[pair_idx]

    def _make_one(self, rng):
        # Sample pair + valid θ
        for _attempt in range(8):
            rec = self.records[int(rng.integers(0, len(self.records)))]
            heatmap = rec["heatmap_fp16"].astype(np.float32)   # (36, 128, 128)
            theta_sums = heatmap.sum(axis=(1, 2))               # (36,)
            valid = np.where(theta_sums > 0)[0]
            if len(valid) > 0:
                break
        else:
            return None
        theta = int(rng.choice(valid))

        rot_part_all = self._rot_for_rec(rec)
        rot_part = np.array(rot_part_all[theta], dtype=np.float32)   # (128, 128)
        fs_mask = rec["fs_mask"].astype(np.float32)
        target_hm = heatmap[theta]                                    # (128, 128)

        if self.augment:
            if rng.random() < 0.5:
                fs_mask = fs_mask[:, ::-1]
                rot_part = rot_part[:, ::-1]
                target_hm = target_hm[:, ::-1]
            if rng.random() < 0.5:
                fs_mask = fs_mask[::-1, :]
                rot_part = rot_part[::-1, :]
                target_hm = target_hm[::-1, :]
            fs_mask = np.ascontiguousarray(fs_mask)
            rot_part = np.ascontiguousarray(rot_part)
            target_hm = np.ascontiguousarray(target_hm)

        if self.hard_target:
            argmax_flat = int(target_hm.reshape(-1).argmax())
            target = np.int64(argmax_flat)
        else:
            s = target_hm.sum()
            if s <= 0:
                return None
            target = (target_hm / s).astype(np.float32)

        return {
            "fs_mask": fs_mask,
            "rot_part_mask": rot_part,
            "target": target,
        }

    def __iter__(self):
        winfo = torch.utils.data.get_worker_info()
        wid = winfo.id if winfo is not None else 0
        rng = np.random.default_rng(self.seed_base + wid * 100003 + os.getpid())
        while True:
            item = self._make_one(rng)
            if item is not None:
                yield item


def make_collate(hard_target):
    def collate(batch):
        fs = np.stack([b["fs_mask"] for b in batch])
        rp = np.stack([b["rot_part_mask"] for b in batch])
        out = {
            "fs_mask": torch.from_numpy(fs),
            "rot_part_mask": torch.from_numpy(rp),
        }
        if hard_target:
            out["target"] = torch.from_numpy(np.array([b["target"] for b in batch],
                                                     dtype=np.int64))
        else:
            out["target"] = torch.from_numpy(np.stack([b["target"] for b in batch]))
        return out
    return collate


# -------------------- loss --------------------

def soft_ce(logits_flat, target_dist_flat):
    """logits, target: (B, K). target sums to 1."""
    log_p = F.log_softmax(logits_flat, dim=-1)
    return -(target_dist_flat * log_p).sum(dim=-1).mean()


# -------------------- evaluation --------------------

@torch.no_grad()
def evaluate(model, val_records, rot_part_dir, device, n_max=None,
             preloaded_rot=None):
    """For each val pair: 36 forward passes, argmax over (θ, r, c), recovery."""
    model.eval()
    recoveries, regrets, top1 = [], [], []
    n = len(val_records) if n_max is None else min(n_max, len(val_records))
    for k in range(n):
        rec = val_records[k]
        heatmap = rec["heatmap_fp16"].astype(np.float32)         # (36, 128, 128)
        gt_flat = int(heatmap.reshape(-1).argmax())
        gt_t, rest = gt_flat // (RES * RES), gt_flat % (RES * RES)
        gt_r, gt_c = rest // RES, rest % RES
        gt_max = float(heatmap[gt_t, gt_r, gt_c])
        if gt_max <= 0:
            continue
        pair_idx = rec["orig_idx"]
        if preloaded_rot is not None and pair_idx in preloaded_rot:
            rot_part_all = preloaded_rot[pair_idx]
        else:
            rot_path = rec.get("rot_part_path") or os.path.join(
                rot_part_dir, f"pair_{pair_idx:05d}.npy")
            rot_part_all = np.load(rot_path)                       # (36, 128, 128) uint8
        fs = torch.from_numpy(rec["fs_mask"].astype(np.float32)).to(device)
        rp = torch.from_numpy(rot_part_all.astype(np.float32)).to(device)
        fs_b = fs.unsqueeze(0).expand(N_THETA, -1, -1).contiguous()  # (36, 128, 128)
        logits = model(fs_b, rp).cpu().numpy()                       # (36, 128, 128)
        pred_flat = int(logits.reshape(-1).argmax())
        pt, prest = pred_flat // (RES * RES), pred_flat % (RES * RES)
        pr, pc = prest // RES, prest % RES
        pick = float(heatmap[pt, pr, pc])
        recoveries.append(pick / max(gt_max, 1e-6))
        regrets.append(gt_max - pick)
        top1.append(1.0 if pred_flat == gt_flat else 0.0)
    model.train()
    return {
        "val/argmax_recovery": float(np.mean(recoveries)) if recoveries else 0.0,
        "val/argmax_regret": float(np.mean(regrets)) if regrets else 0.0,
        "val/top1": float(np.mean(top1)) if top1 else 0.0,
    }


# -------------------- main --------------------

def main():
    log("entered main()")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/hier_training_data_soft.pkl")
    ap.add_argument("--rot-part-dir", type=str, default="data/rot_part_masks_theta36")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-max", type=int, default=200)
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints/perthet")
    ap.add_argument("--log-dir", type=str, default="logs/perthet")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--hard-target", action="store_true",
                    help="One-hot pixel CE instead of soft CE on normalized heatmap.")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out-log", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--preload-rot", action="store_true",
                    help="Preload all rot_part masks into RAM at start (~7 GB).")
    ap.add_argument("--preload-workers", type=int, default=64,
                    help="ThreadPool size for the preload step.")
    args = ap.parse_args()

    global _LOG_FH
    if args.out_log:
        os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
        _LOG_FH = open(args.out_log, "a", buffering=1)
        log(f"=== new run ===  out-log={args.out_log}")
    log(f"args: {vars(args)}")

    device = torch.device(args.device)
    log(f"using device: {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        log(f"  GPU mem free/total: {free/1e9:.2f} / {total/1e9:.2f} GB")

    for d in (args.ckpt_dir, args.log_dir):
        os.makedirs(d, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    log(f"loading data: {args.data}")
    t = time.time()
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    log(f"  loaded in {time.time()-t:.1f}s  "
        f"train={len(data['train'])}  val={len(data['val'])}")

    if not os.path.isdir(args.rot_part_dir):
        raise RuntimeError(f"missing rot_part dir: {args.rot_part_dir}")
    n_rot = len(os.listdir(args.rot_part_dir))
    log(f"rot_part dir: {args.rot_part_dir}  ({n_rot} files)")

    # Preload ALL rot_part masks into RAM up front. Modal volume per-file open
    # latency is the killer otherwise. 12000 * (36, 128, 128) uint8 = ~7 GB.
    preloaded_rot = None
    if args.preload_rot:
        from concurrent.futures import ThreadPoolExecutor
        log(f"preloading rot_part masks (threadpool={args.preload_workers}) ...")
        t = time.time()
        # Build (pair_idx, path) list — supports per-record rot_part_path
        # for the combined convex+concave dataset.
        items = {}
        for r in data["train"] + data["val"]:
            pid = r["orig_idx"]
            if pid in items:
                continue
            path = r.get("rot_part_path") or os.path.join(
                args.rot_part_dir, f"pair_{pid:05d}.npy")
            items[pid] = path
        items_list = sorted(items.items())
        preloaded_rot = {}
        from threading import Lock
        lock = Lock()
        counter = [0]

        def _load_one(item):
            pidx, path = item
            arr = np.load(path)
            with lock:
                preloaded_rot[pidx] = arr
                counter[0] += 1
                if counter[0] % 2000 == 0:
                    log(f"  preloaded {counter[0]}/{len(items_list)} "
                        f"({time.time()-t:.0f}s)")

        with ThreadPoolExecutor(max_workers=args.preload_workers) as pool:
            list(pool.map(_load_one, items_list))
        needed = items_list  # used in log message below
        gb = sum(a.nbytes for a in preloaded_rot.values()) / 1e9
        log(f"  preloaded {len(preloaded_rot)} pairs ({gb:.1f} GB) "
            f"in {time.time()-t:.0f}s")

    log("building model ...")
    model = PerThetaPlacementUNet(base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  {n_params/1e6:.3f}M params")

    if args.resume:
        log(f"resuming from {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ck.get("model", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"  loaded.  missing={len(missing)}  unexpected={len(unexpected)}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    log(f"  AMP enabled: {scaler.is_enabled()}")

    ds = PerThetaDataset(data["train"], args.rot_part_dir,
                         augment=args.augment, hard_target=args.hard_target,
                         preloaded_rot=preloaded_rot)
    collate = make_collate(args.hard_target)
    loader = DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers,
        collate_fn=collate, pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
    )

    log("warming up: pulling first batch ...")
    t = time.time()
    batch_iter = iter(loader)
    batch = next(batch_iter)
    log(f"  first batch ready in {time.time()-t:.1f}s")

    log(f"==== TRAINING START ==== {args.steps} steps, batch={args.batch}, "
        f"amp={args.amp}, hard_target={args.hard_target}")

    model.train()
    t0 = time.time()
    step = 0
    while step < args.steps:
        try:
            fs = batch["fs_mask"].to(device, non_blocking=True)
            rp = batch["rot_part_mask"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)
            B = fs.shape[0]

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(fs, rp)                              # (B, 128, 128)
                logits_flat = logits.reshape(B, -1)                 # (B, 16384)
                if args.hard_target:
                    loss = F.cross_entropy(logits_flat, tgt)
                else:
                    target_flat = tgt.reshape(B, -1)
                    loss = soft_ce(logits_flat, target_flat)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            sched.step()

            if step < 5 or step % args.log_every == 0:
                elapsed = time.time() - t0
                ips = (step + 1) / max(elapsed, 1e-3)
                gpu_msg = ""
                if device.type == "cuda":
                    mem = torch.cuda.memory_allocated() / 1e9
                    gpu_msg = f"  gpu_alloc={mem:.2f}GB"
                with torch.no_grad():
                    pred_flat = logits_flat.argmax(dim=-1)
                    if args.hard_target:
                        train_top1 = (pred_flat == tgt).float().mean().item()
                    else:
                        gt_argmax = tgt.reshape(B, -1).argmax(dim=-1)
                        train_top1 = (pred_flat == gt_argmax).float().mean().item()
                log(f"step {step:6d}  loss={loss.item():.4f}  "
                    f"top1={train_top1:.3f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}  {ips:.2f} step/s{gpu_msg}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/top1", train_top1, step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)

            if step > 0 and step % args.val_every == 0:
                tv = time.time()
                metrics = evaluate(model, data["val"], args.rot_part_dir,
                                   device, n_max=args.val_max,
                                   preloaded_rot=preloaded_rot)
                log(f"  [val@{step}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    + f"  ({time.time()-tv:.1f}s)")
                for k, v in metrics.items():
                    writer.add_scalar(k, v, step)

            if step > 0 and step % args.ckpt_every == 0:
                ckpt = os.path.join(args.ckpt_dir, f"step_{step:06d}.pt")
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args)}, ckpt)
                log(f"  [ckpt] saved {ckpt}")

            step += 1
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
        except KeyboardInterrupt:
            log("KeyboardInterrupt — stopping early")
            break
        except Exception:
            log("EXCEPTION in training step:")
            traceback.print_exc()
            raise

    log("Final eval on full val set ...")
    metrics = evaluate(model, data["val"], args.rot_part_dir, device, n_max=None,
                       preloaded_rot=preloaded_rot)
    log("  final: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    for k, v in metrics.items():
        writer.add_scalar(k, v, args.steps)

    final = os.path.join(args.ckpt_dir, "final.pt")
    torch.save({"model": model.state_dict(), "step": step,
                "args": vars(args), "final_metrics": metrics}, final)
    log(f"==== DONE. Final ckpt: {final}  total_time={time.time()-t0:.1f}s ====")


if __name__ == "__main__":
    main()
