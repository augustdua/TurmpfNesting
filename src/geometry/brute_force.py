"""Brute-force reward-heatmap computation for one (fs, part) pair.

Pure CPU, torch-free. Used both as the report's on-the-fly fallback and as the
per-worker function in scripts/precompute_bf.py.

For each rotation theta in N_THETA equispaced bins of [0, 360):
  1. Compute the IFP (locus of valid centroid translations).
  2. Rasterize the IFP to a RES x RES binary mask.
  3. For each pixel in the IFP mask, Shapely-score the placement.
Pixels outside the IFP stay zero.
"""
import numpy as np
from shapely.affinity import translate as shp_translate

from src.geometry.rewards import compute_reward_exp
from src.geometry.ifp import compute_ifp_exact
from scripts.rasterize_ifp_union import rasterize_polygon

RES = 128
N_THETA = 36


def _pix_to_world(r, c, res=RES):
    x = 2.0 * c / (res - 1) - 1.0
    y = 1.0 - 2.0 * r / (res - 1)
    return float(x), float(y)


def compute_bf_array(fs, part, k=10.0, verbose=False):
    """Return the raw (N_THETA, RES, RES) BF heatmap. No argmax bookkeeping."""
    cx, cy = part.centroid.coords[0]
    part_centered = shp_translate(part, -cx, -cy)

    thetas = np.linspace(0.0, 360.0, N_THETA, endpoint=False, dtype=np.float32)
    bf = np.zeros((N_THETA, RES, RES), dtype=np.float32)

    for t_idx, theta in enumerate(thetas):
        try:
            ifp = compute_ifp_exact(fs, part, float(theta))
        except Exception:
            continue
        if ifp.is_empty or ifp.area < 1e-8:
            continue
        ifp_mask = rasterize_polygon(ifp, RES) > 0
        if not ifp_mask.any():
            continue
        rc = np.argwhere(ifp_mask)
        if verbose:
            print(f"      theta {theta:6.1f}deg: {len(rc):5d} IFP px",
                  flush=True)
        for row, col in rc:
            x = col / (RES - 1) * 2 - 1
            y = 1 - row / (RES - 1) * 2
            try:
                bf[t_idx, row, col] = float(
                    compute_reward_exp(fs, part, x, y, float(theta), k=k)
                )
            except Exception:
                pass

    return bf


def summarize_bf(bf):
    """(heatmap, t_idx, r_idx, c_idx, theta_deg, x, y, r_star) from a (N_THETA, H, W) array."""
    flat = int(bf.reshape(-1).argmax())
    t_idx = flat // (RES * RES)
    rest = flat % (RES * RES)
    r_idx = rest // RES
    c_idx = rest % RES
    x_star, y_star = _pix_to_world(r_idx, c_idx, RES)
    theta_deg = 360.0 * t_idx / N_THETA
    r_star = float(bf[t_idx, r_idx, c_idx])
    return bf, t_idx, r_idx, c_idx, theta_deg, x_star, y_star, r_star


def compute_brute_force_on_the_fly(fs, part, k=10.0, verbose=False):
    """compute_bf_array + summarize_bf in one call (matches the report's old API)."""
    return summarize_bf(compute_bf_array(fs, part, k=k, verbose=verbose))
