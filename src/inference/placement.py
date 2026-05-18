"""
Inference-time placement: model forward pass + bounded Shapely refinement.

Usage:
    from src.inference.placement import PlacementModel
    pm = PlacementModel(ckpt="checkpoints/perthet_combined/final.pt", device="cuda")
    theta, x, y, reward = pm.place(fs_poly, part_poly,
                                   refine_pixels=10, refine_thetas=2)

If refine_pixels=0 and refine_thetas=0, only the model prediction is returned
(scored by one Shapely call). Otherwise, a grid search of size
(2*refine_pixels+1)^2 * (2*refine_thetas+1) around the predicted (θ̂, r̂, ĉ) is
brute-forced via Shapely and the best is returned.

Default window: 21*21*5 = 2205 Shapely calls per pair ≈ 165 ms refinement.
"""
import numpy as np
import torch
from shapely.affinity import rotate as shp_rotate
from shapely.affinity import translate as shp_translate

from src.models.neural_bo_policy import _SmallUNet
from src.geometry.rewards import compute_reward_exp
from scripts.rasterize_ifp_union import rasterize_polygon

RES = 128
N_THETA = 36


def _pix_to_world(r, c, res=RES):
    """Inverse of rasterize: pixel row/col -> world (x, y) in [-1, 1]."""
    x = 2.0 * c / (res - 1) - 1.0
    y = 1.0 - 2.0 * r / (res - 1)
    return float(x), float(y)


def _center_part(part_poly):
    cx, cy = part_poly.centroid.coords[0]
    return shp_translate(part_poly, -cx, -cy)


class PlacementModel:
    """Wraps the trained per-(pair, θ) UNet for inference."""

    def __init__(self, ckpt, device="cuda", base=32, n_theta=N_THETA, res=RES):
        self.device = torch.device(device if torch.cuda.is_available()
                                   else "cpu")
        self.n_theta = n_theta
        self.res = res
        self.thetas_deg = np.linspace(0.0, 360.0, n_theta, endpoint=False,
                                      dtype=np.float32)
        # Always load to CPU first then move (avoids Windows CUDA OOM at load).
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = _SmallUNet(in_ch=2, base=base, out_ch=1)
        sd = ck.get("model", ck)
        # Strip "unet." prefix if present (from PerThetaPlacementUNet wrapper).
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("unet."):
                new_sd[k[len("unet."):]] = v
            else:
                new_sd[k] = v
        self.model.load_state_dict(new_sd, strict=True)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def _model_forward(self, fs_mask, rot_part_masks):
        """
        fs_mask:        (res, res) float32 numpy
        rot_part_masks: (n_theta, res, res) float32 numpy
        Returns:        (n_theta, res, res) float32 numpy — per-pixel logits.
        """
        fs_t = torch.from_numpy(fs_mask).to(self.device)
        fs_t = fs_t.unsqueeze(0).expand(self.n_theta, -1, -1).contiguous()
        rp_t = torch.from_numpy(rot_part_masks).to(self.device)
        # _SmallUNet takes a stacked (B, 2, H, W); concat fs+rp on channel dim.
        x = torch.stack([fs_t, rp_t], dim=1)
        logits = self.model(x).squeeze(1)            # (n_theta, res, res)
        return logits.cpu().numpy()

    def place(
        self,
        fs_poly,
        part_poly,
        refine_pixels=10,
        refine_thetas=2,
        k=10.0,
    ):
        """
        Predict a placement for (fs_poly, part_poly).

        Returns (theta_deg, x, y, shapely_reward).
        """
        # ---- model forward ----
        fs_mask = rasterize_polygon(fs_poly, self.res).astype(np.float32)
        part_c = _center_part(part_poly)
        rot_parts = np.stack([
            rasterize_polygon(
                shp_rotate(part_c, float(t), origin=(0.0, 0.0),
                           use_radians=False),
                self.res,
            )
            for t in self.thetas_deg
        ]).astype(np.float32)

        logits = self._model_forward(fs_mask, rot_parts)   # (36, 128, 128)
        flat = int(logits.reshape(-1).argmax())
        pred_t = flat // (self.res * self.res)
        rest = flat % (self.res * self.res)
        pred_r = rest // self.res
        pred_c = rest % self.res

        # ---- score the model's pick (always 1 Shapely call) ----
        x0, y0 = _pix_to_world(pred_r, pred_c, self.res)
        t0 = float(self.thetas_deg[pred_t])
        try:
            best_reward = compute_reward_exp(fs_poly, part_poly, x0, y0, t0, k=k)
        except Exception:
            best_reward = -1.0
        best = (t0, x0, y0, best_reward)

        # ---- bounded refinement ----
        if refine_pixels > 0 or refine_thetas > 0:
            for dt in range(-refine_thetas, refine_thetas + 1):
                t_idx = (pred_t + dt) % self.n_theta
                t = float(self.thetas_deg[t_idx])
                for dr in range(-refine_pixels, refine_pixels + 1):
                    r = pred_r + dr
                    if r < 0 or r >= self.res:
                        continue
                    for dc in range(-refine_pixels, refine_pixels + 1):
                        c = pred_c + dc
                        if c < 0 or c >= self.res:
                            continue
                        if dt == 0 and dr == 0 and dc == 0:
                            continue        # already scored
                        x, y = _pix_to_world(r, c, self.res)
                        try:
                            rw = compute_reward_exp(fs_poly, part_poly,
                                                    x, y, t, k=k)
                        except Exception:
                            continue
                        if rw > best[3]:
                            best = (t, x, y, rw)
        return best
