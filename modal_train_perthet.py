"""
Modal training for the per-(pair, θ) placement UNet (no tile head, no cell head).

Inputs: (fs_mask, rotated_part_mask).  Output: (128, 128) per-pixel reward logits.
At inference: 36 forward passes per pair, argmax over (θ, r, c).

Volume layout (`nestingrl-data`):
  /vol/hier_training_data_soft.pkl                 (14.55 GB, fs + heatmap_fp16)
  /vol/rot_part_masks_theta36/pair_NNNNN.npy       (~7 GB, 12k files, 36×128×128 uint8)
  /vol/checkpoints/perthet/{step_*,final}.pt
  /vol/logs/perthet/

Train:
  modal run modal_train_perthet.py --steps 5000 --batch 256
"""
import os
import modal

app = modal.App("nestingrl-perthet")

_HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=2.0,<3",
        "scipy",
        "shapely>=2.0",
        "tqdm",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "matplotlib",
        "tensorboard",
    )
    .add_local_dir(os.path.join(_HERE, "src"), remote_path="/root/project/src")
    .add_local_dir(os.path.join(_HERE, "scripts"), remote_path="/root/project/scripts")
)

volume = modal.Volume.from_name("nestingrl-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=98304,
    timeout=7200,
    volumes={"/vol": volume},
)
def train(
    steps: int = 5000,
    batch: int = 256,
    lr: float = 3e-4,
    base: int = 32,
    val_every: int = 250,
    val_max: int = 400,
    ckpt_every: int = 1000,
    workers: int = 4,
    amp: bool = True,
    augment: bool = False,
    hard_target: bool = False,
    ckpt_dir: str = "/vol/checkpoints/perthet",
    log_dir: str = "/vol/logs/perthet",
    resume: str = "",
    preload_rot: bool = True,
    data_path: str = "/vol/hier_training_data_soft.pkl",
    rot_part_dir: str = "/vol/rot_part_masks_theta36",
):
    import os
    import sys
    import subprocess
    import time

    sys.path.insert(0, "/root/project")

    import torch
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
    print(f"torch {torch.__version__}  CUDA {torch.version.cuda}", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free", flush=True)

    rot_dir = rot_part_dir
    for p, label in [(data_path, "data pkl"), (rot_dir, "rot_part dir")]:
        if not os.path.exists(p):
            raise RuntimeError(f"missing {label}: {p}")
    print(f"Data: {data_path}  ({os.path.getsize(data_path)/1e9:.2f} GB)", flush=True)
    print(f"Rot:  {rot_dir}  ({len(os.listdir(rot_dir))} files)", flush=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        "python", "-u", "-m", "scripts.train_perthet",
        "--data", data_path,
        "--rot-part-dir", rot_dir,
        "--steps", str(steps),
        "--batch", str(batch),
        "--lr", str(lr),
        "--base", str(base),
        "--val-every", str(val_every),
        "--val-max", str(val_max),
        "--ckpt-every", str(ckpt_every),
        "--workers", str(workers),
        "--ckpt-dir", ckpt_dir,
        "--log-dir", log_dir,
        "--device", "cuda",
        "--log-every", "25",
        "--out-log", os.path.join(log_dir, "run.log"),
    ]
    if amp:
        cmd.append("--amp")
    if augment:
        cmd.append("--augment")
    if hard_target:
        cmd.append("--hard-target")
    if resume:
        cmd += ["--resume", resume]
    if preload_rot:
        cmd.append("--preload-rot")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Launching:", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/root/project", env=env)
    elapsed = time.time() - t0
    print(f"Training exit: {proc.returncode}  wall={elapsed:.1f}s "
          f"({elapsed/60:.1f} min)", flush=True)
    volume.commit()
    return {"exit_code": proc.returncode, "wall_seconds": elapsed,
            "ckpt_dir": ckpt_dir, "log_dir": log_dir}


@app.function(image=image, volumes={"/vol": volume}, timeout=600)
def fetch_best():
    import os
    out = {}
    for rel in [
        "checkpoints/perthet/final.pt",
        "logs/perthet/run.log",
    ]:
        p = os.path.join("/vol", rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[rel] = f.read()
            print(f"  read {p}  ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
    ckpt_dir = "/vol/checkpoints/perthet"
    if os.path.isdir(ckpt_dir) and "checkpoints/perthet/final.pt" not in out:
        steps = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("step_"))
        if steps:
            p = os.path.join(ckpt_dir, steps[-1])
            with open(p, "rb") as f:
                out[f"checkpoints/perthet/{steps[-1]}"] = f.read()
    return out


@app.local_entrypoint()
def main(
    steps: int = 5000,
    batch: int = 256,
    lr: float = 3e-4,
    base: int = 32,
    augment: bool = False,
    hard_target: bool = False,
    val_every: int = 250,
    ckpt_every: int = 1000,
    ckpt_dir: str = "/vol/checkpoints/perthet",
    log_dir: str = "/vol/logs/perthet",
    resume: str = "",
    preload_rot: bool = True,
    data_path: str = "/vol/hier_training_data_soft.pkl",
    rot_part_dir: str = "/vol/rot_part_masks_theta36",
):
    print(f"Launching Modal A100-80GB per-(pair,θ): steps={steps}, batch={batch}, "
          f"lr={lr}, base={base}, augment={augment}, hard_target={hard_target}, "
          f"preload_rot={preload_rot}, data={data_path}")
    result = train.remote(steps=steps, batch=batch, lr=lr, base=base,
                          val_every=val_every, ckpt_every=ckpt_every,
                          augment=augment, hard_target=hard_target,
                          ckpt_dir=ckpt_dir, log_dir=log_dir, resume=resume,
                          preload_rot=preload_rot,
                          data_path=data_path, rot_part_dir=rot_part_dir)
    print(f"\nTraining result: {result}")

    print("\nFetching results ...")
    files = fetch_best.remote()
    for rel, data in files.items():
        local = os.path.join(".", rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            f.write(data)
        print(f"  {local}  ({len(data)/1e6:.1f} MB)")

    print("\nDone.")
