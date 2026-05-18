"""
Tiny Modal helper to list contents of the nestingrl-data volume from inside
Modal, since the local `modal volume ls` CLI is flaking out with
"Connection lost" right now.

Run with:
    modal run modal_check_volume.py
"""
import modal
import os

app = modal.App("nestingrl-check-volume")
vol = modal.Volume.from_name("nestingrl-data", create_if_missing=False)

img = modal.Image.debian_slim(python_version="3.11")


@app.function(image=img, volumes={"/vol": vol}, timeout=60)
def list_vol():
    out_lines = []
    for root, dirs, files in os.walk("/vol"):
        for f in files:
            p = os.path.join(root, f)
            try:
                sz = os.path.getsize(p)
            except OSError as e:
                sz = f"ERR: {e}"
            out_lines.append(f"{p}  {sz}")
    if not out_lines:
        return "VOLUME EMPTY"
    return "\n".join(out_lines)


@app.local_entrypoint()
def main():
    print(list_vol.remote())
