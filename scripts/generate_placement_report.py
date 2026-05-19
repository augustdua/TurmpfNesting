"""
Generate a *modern-LaTeX* visualization report of the placement pipeline.

Outputs
-------
visualizations/report/placement_pipeline_report.pdf      (final, via pdflatex)
visualizations/report/placement_pipeline_report.tex      (LaTeX source)
visualizations/report/figs/pair_NNNNN.png                (per-pair figures)
visualizations/report/figs/summary.png                   (summary chart)

Each example page shows, for one convex val pair:
  * input free-space polygon
  * input part polygon (centered)
  * predicted reward heatmap @ theta_hat, with refine window + refined target
  * brute-force reward heatmap @ theta_star (ground truth optimum)
  * placement at model argmax (no refine)
  * placement after Shapely refinement
  * placement at brute-force argmax (ground truth)

The methodology section is transcribed verbatim (in LaTeX form) from
METHODOLOGY.md so the report is self-contained.

Usage:
    python -m scripts.generate_placement_report
    python -m scripts.generate_placement_report --n 16 --seed 1 --no-compile
"""
import argparse
import os
import pickle
import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.wkt import loads as wkt_loads
from shapely.affinity import rotate as shp_rotate
from shapely.affinity import translate as shp_translate

from src.inference.placement import (
    PlacementModel, _pix_to_world, _center_part, RES, N_THETA,
)
from src.geometry.rewards import compute_reward_exp
from src.geometry.ifp import compute_ifp_exact
from scripts.rasterize_ifp_union import rasterize_polygon

CONVEX_VAL_START = 0
CONVEX_VAL_END = 12000
CONCAVE_VAL_START = 0
CONCAVE_VAL_END = 10000
BF_CHUNKS_DIR = "data/reward_heatmaps_exp_k10_inside.npy_chunks"


# ----------------------------------------------------------------------- #
# Plotting helpers
# ----------------------------------------------------------------------- #

def plot_poly(ax, poly, **kw):
    if poly is None or poly.is_empty:
        return
    geoms = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, **kw)


def plot_placement(ax, fs, part_c, theta_deg, x, y, title, edge_color="red"):
    placed = shp_translate(
        shp_rotate(part_c, theta_deg, origin=(0.0, 0.0)), x, y
    )
    plot_poly(ax, fs, alpha=0.20, color="steelblue",
              edgecolor="steelblue", linewidth=1.8)
    plot_poly(ax, placed, alpha=0.65, color="darkorange",
              edgecolor=edge_color, linewidth=2.0)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)


# ----------------------------------------------------------------------- #
# Inference + brute-force resolution
# ----------------------------------------------------------------------- #

def run_pipeline_timed(pm, fs, part, refine_pixels, refine_thetas, k=10.0):
    """Replay PlacementModel.place(), breaking out per-step timing."""
    part_c = _center_part(part)

    t0 = time.time()
    fs_mask = rasterize_polygon(fs, RES).astype(np.float32)
    rot_parts = np.stack([
        rasterize_polygon(
            shp_rotate(part_c, float(t), origin=(0.0, 0.0)),
            RES,
        )
        for t in pm.thetas_deg
    ]).astype(np.float32)
    t_raster = time.time() - t0

    t0 = time.time()
    logits = pm._model_forward(fs_mask, rot_parts)
    t_forward = time.time() - t0

    flat = int(logits.reshape(-1).argmax())
    pred_t = flat // (RES * RES)
    rest = flat % (RES * RES)
    pred_r = rest // RES
    pred_c = rest % RES

    x0, y0 = _pix_to_world(pred_r, pred_c, RES)
    t0_deg = float(pm.thetas_deg[pred_t])
    try:
        r_noref = float(compute_reward_exp(fs, part, x0, y0, t0_deg, k=k))
    except Exception:
        r_noref = -1.0

    best = (t0_deg, x0, y0, r_noref, pred_t, pred_r, pred_c)
    n_calls = 1

    t0 = time.time()
    for dt in range(-refine_thetas, refine_thetas + 1):
        t_idx = (pred_t + dt) % N_THETA
        t_deg = float(pm.thetas_deg[t_idx])
        for dr in range(-refine_pixels, refine_pixels + 1):
            r = pred_r + dr
            if r < 0 or r >= RES:
                continue
            for dc in range(-refine_pixels, refine_pixels + 1):
                c = pred_c + dc
                if c < 0 or c >= RES:
                    continue
                if dt == 0 and dr == 0 and dc == 0:
                    continue
                xw, yw = _pix_to_world(r, c, RES)
                try:
                    rw = float(
                        compute_reward_exp(fs, part, xw, yw, t_deg, k=k)
                    )
                except Exception:
                    continue
                n_calls += 1
                if rw > best[3]:
                    best = (t_deg, xw, yw, rw, t_idx, r, c)
    t_refine = time.time() - t0

    return dict(
        logits=logits,
        pred_t=pred_t, pred_r=pred_r, pred_c=pred_c,
        t0_deg=t0_deg, x0=x0, y0=y0, r_noref=r_noref,
        ref_t_deg=best[0], ref_x=best[1], ref_y=best[2],
        r_ref=best[3], ref_t_idx=best[4],
        ref_pred_r=best[5], ref_pred_c=best[6],
        t_raster=t_raster, t_forward=t_forward, t_refine=t_refine,
        n_calls=n_calls,
    )


def _summarize_bf(bf):
    """(heatmap, t_idx, r_idx, c_idx, theta_deg, x, y, r_star) from a (36, H, W) array."""
    flat = int(bf.reshape(-1).argmax())
    t_idx = flat // (RES * RES)
    rest = flat % (RES * RES)
    r_idx = rest // RES
    c_idx = rest % RES
    x_star, y_star = _pix_to_world(r_idx, c_idx, RES)
    theta_deg = 360.0 * t_idx / N_THETA
    r_star = float(bf[t_idx, r_idx, c_idx])
    return bf, t_idx, r_idx, c_idx, theta_deg, x_star, y_star, r_star


def load_brute_force_convex(idx):
    """Load precomputed brute-force heatmap chunk for a convex pair."""
    path = os.path.join(BF_CHUNKS_DIR, f"pair_{idx:05d}.npy")
    return _summarize_bf(np.load(path).astype(np.float32))


def compute_brute_force_on_the_fly(fs, part, k=10.0, verbose=False):
    """Compute (36, 128, 128) brute-force reward heatmap for ANY (fs, part) pair.

    Per the precompute pipeline (modal_concave_precompute.py): for each rotation
    theta, compute the IFP, rasterize it, and Shapely-score every pixel inside.
    Pixels outside the IFP stay zero.
    """
    from shapely.affinity import translate as shp_translate
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

    return _summarize_bf(bf)


# ----------------------------------------------------------------------- #
# Per-pair figure
# ----------------------------------------------------------------------- #

def render_pair_figure(fig_path, pm, rec, label, bf_tuple,
                       refine_pixels, refine_thetas):
    """Render one pair page.

    label    : string used in plots, tex section title, and dict["idx"].
    bf_tuple : the 8-tuple returned by _summarize_bf (heatmap + argmax info).
    """
    fs = wkt_loads(rec["fs_poly_wkt"])
    part = wkt_loads(rec["part_poly_wkt"])
    part_c = _center_part(part)

    info = run_pipeline_timed(pm, fs, part, refine_pixels, refine_thetas)
    bf, bf_t_idx, bf_r, bf_c, bf_t_deg, x_star, y_star, r_star = bf_tuple

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # ---- Top row ----
    # (0, 0) predicted heatmap @ theta_hat
    ax = axes[0, 0]
    slice_pred = info["logits"][info["pred_t"]]
    im = ax.imshow(slice_pred, extent=(-1, 1, -1, 1),
                   origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter([info["x0"]], [info["y0"]], s=100, marker="x",
               color="red", linewidths=2.5, label="model argmax", zorder=5)
    pix = 2.0 / (RES - 1)
    half_w = (refine_pixels + 0.5) * pix
    ax.add_patch(Rectangle(
        (info["x0"] - half_w, info["y0"] - half_w),
        2 * half_w, 2 * half_w,
        fill=False, edgecolor="red", linewidth=1.4, linestyle="--",
        label="refine window",
    ))
    if info["ref_t_idx"] == info["pred_t"]:
        ax.scatter([info["ref_x"]], [info["ref_y"]],
                   s=130, marker="o", facecolor="none",
                   edgecolor="lime", linewidths=2.5,
                   label="refined", zorder=5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Predicted reward map @ "
        r"$\hat{\theta}$="
        f"{info['t0_deg']:.0f}"
        r"$^\circ$",
        fontsize=11,
    )
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

    # (0, 1) brute-force heatmap @ theta_star
    ax = axes[0, 1]
    slice_bf = bf[bf_t_idx]
    im = ax.imshow(slice_bf, extent=(-1, 1, -1, 1),
                   origin="upper", cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.scatter([x_star], [y_star], s=110, marker="*",
               color="cyan", edgecolors="black", linewidths=1.3,
               label=r"brute-force argmax", zorder=5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Brute-force reward map @ "
        r"$\theta^\ast$="
        f"{bf_t_deg:.0f}"
        r"$^\circ$",
        fontsize=11,
    )
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)

    # (0, 2) part polygon (centered)
    ax = axes[0, 2]
    plot_poly(ax, part_c, alpha=0.55, color="darkorange",
              edgecolor="darkorange", linewidth=2)
    ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title("Part polygon (centered)", fontsize=11)

    # ---- Bottom row: placements ----
    plot_placement(
        axes[1, 0], fs, part_c,
        info["t0_deg"], info["x0"], info["y0"],
        f"Model only (no refine)  r = {info['r_noref']:.2f}",
        edge_color="red",
    )
    plot_placement(
        axes[1, 1], fs, part_c,
        info["ref_t_deg"], info["ref_x"], info["ref_y"],
        f"After refinement  r = {info['r_ref']:.2f}",
        edge_color="forestgreen",
    )
    plot_placement(
        axes[1, 2], fs, part_c,
        bf_t_deg, x_star, y_star,
        f"Brute force (ground truth)  $r^\\ast$ = {r_star:.2f}",
        edge_color="black",
    )

    fig.suptitle(
        f"Pair {label}  (area ratio {rec.get('area_ratio', float('nan')):.3f})",
        fontsize=14, y=0.995,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return dict(
        idx=label,
        area_ratio=rec.get("area_ratio", float("nan")),
        t0_deg=info["t0_deg"], x0=info["x0"], y0=info["y0"],
        r_noref=info["r_noref"],
        ref_t_deg=info["ref_t_deg"], ref_x=info["ref_x"], ref_y=info["ref_y"],
        r_ref=info["r_ref"],
        bf_t_deg=bf_t_deg, x_star=x_star, y_star=y_star, r_star=r_star,
        rec_noref=info["r_noref"] / r_star if r_star > 0 else 0.0,
        rec_ref=info["r_ref"] / r_star if r_star > 0 else 0.0,
        t_raster_ms=info["t_raster"] * 1000,
        t_forward_ms=info["t_forward"] * 1000,
        t_refine_ms=info["t_refine"] * 1000,
        n_calls=info["n_calls"],
    )


# ----------------------------------------------------------------------- #
# Summary figure
# ----------------------------------------------------------------------- #

def render_summary_figure(fig_path, rows):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # bar chart: per-pair recovery (model-only vs refined)
    ax = axes[0]
    rs = sorted(rows, key=lambda r: r["rec_noref"])
    xs = np.arange(len(rs))
    ax.bar(xs - 0.2, [r["rec_noref"] for r in rs], width=0.4,
           color="cornflowerblue", label="model only")
    ax.bar(xs + 0.2, [r["rec_ref"] for r in rs], width=0.4,
           color="seagreen", label="refined")
    ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--",
               label="brute force ($r^\\ast$)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["idx"]) for r in rs],
                       rotation=45, fontsize=8)
    ax.set_ylabel("Recovery  $r / r^\\ast$")
    ax.set_title("Recovery per pair  (sorted by model-only)")
    ax.set_ylim(0, max(1.05, max(r["rec_ref"] for r in rs) + 0.05))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # bar chart: refinement uplift
    ax = axes[1]
    rs2 = sorted(rows, key=lambda r: -(r["rec_ref"] - r["rec_noref"]))
    deltas = [r["rec_ref"] - r["rec_noref"] for r in rs2]
    colors = ["seagreen" if d > 0 else "indianred" for d in deltas]
    ax.bar(range(len(rs2)), deltas, color=colors)
    ax.set_xticks(range(len(rs2)))
    ax.set_xticklabels([str(r["idx"]) for r in rs2],
                       rotation=45, fontsize=8)
    ax.set_ylabel("$\\Delta$ recovery  (refined $-$ model only)")
    ax.set_title("Refinement uplift  (sorted)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- #
# LaTeX
# ----------------------------------------------------------------------- #

LATEX_PREAMBLE = r"""
\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{float}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black]{hyperref}
\usepackage{siunitx}
\sisetup{detect-all}

\titleformat{\section}{\Large\bfseries\sffamily}{\thesection}{0.6em}{}
\titleformat{\subsection}{\large\bfseries\sffamily}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\sffamily}{\thesubsubsection}{0.6em}{}

\newcommand{\code}[1]{\texttt{#1}}
\setlength{\parskip}{0.4em}
"""


# Methodology transcribed from METHODOLOGY.md.  Kept faithful; mechanical
# markdown -> LaTeX conversion of the parts that aren't already raw LaTeX
# math.
METHODOLOGY_LATEX = r"""
\section{Placement Network --- Methodology}

A learned placement function that, given a free-space polygon $F$ and a part polygon $P$, returns a placement $(\theta, x, y)$ maximizing a Shapely-defined reward.  Trained on 22\,000 pairs (12\,k convex--convex + 10\,k concave-fs--convex), achieves val recovery $\approx 0.72$ with one small U-Net and one feed-forward pass per rotation.

\subsection{Formal problem statement}

\subsubsection{Notation}

Let
\begin{itemize}[leftmargin=*,nosep]
  \item $F \subset \mathbb{R}^2$ be a simple polygon (the \emph{free space}), possibly non-convex.
  \item $P \subset \mathbb{R}^2$ be a simple convex polygon (the \emph{part}), centered at the origin so that its centroid coincides with $(0,0)$.
  \item $R_\theta : \mathbb{R}^2 \to \mathbb{R}^2$ denote rotation by angle $\theta$ about the origin.
  \item $T_{(x,y)}$ denote translation by $(x,y)$.
  \item The \emph{placement} of $P$ at position $(x,y)$ with rotation $\theta$ is the set
  \[
    P_{\theta,x,y} \;=\; T_{(x,y)}\bigl(R_\theta(P)\bigr).
  \]
\end{itemize}

We discretize:
\begin{itemize}[leftmargin=*,nosep]
  \item Rotations: $\theta \in \Theta = \{0, \tfrac{2\pi}{36}, \tfrac{4\pi}{36}, \dots, \tfrac{70\pi}{36}\}$, indexed $i = 0, \dots, 35$.
  \item Positions: $(x,y) \in [-1,1]^2$ on a $128 \times 128$ raster grid. Each grid cell is indexed by row $r \in \{0, \dots, 127\}$ and column $c \in \{0, \dots, 127\}$. The pixel-to-world map is
  \[
    \operatorname{pix}^{-1}(r, c) \;=\; \left(\,\tfrac{2c}{127} - 1,\;\; 1 - \tfrac{2r}{127}\,\right) \in [-1,1]^2,
  \]
  so column $c$ is the $x$-axis (left $\to$ right) and row $r$ is the $y$-axis (top $\to$ bottom, sign-flipped to match image convention). Total pixels $\lvert\Omega\rvert = 128^2 = 16\,384$.
\end{itemize}

The full discrete action space has size $\lvert\Theta\rvert \cdot \lvert\Omega\rvert = 36 \cdot 16\,384 = 589\,824$.

\subsubsection{Validity and reward}

Define the \textbf{Inner Fit Polygon} at rotation $\theta$ as the locus of valid centroid positions:
\[
  \operatorname{IFP}(F, P, \theta) \;=\; \{ (x,y) \in \mathbb{R}^2 : P_{\theta,x,y} \subseteq F \}.
\]
Equivalently, $\operatorname{IFP}(F, P, \theta) = F \ominus R_\theta(P)$ (Minkowski erosion). It is well-defined and computable for any $F$ (convex or concave) and convex $P$.

\paragraph{Contact and reward functions}
For a free-space polygon $F$ and a placed-part polygon $Q \subseteq F$, define the \textbf{contact score}
\[
  \operatorname{contact}(F, Q) \;=\; \alpha\,\operatorname{adherence}(F, Q) \;+\; \beta\,\operatorname{proximity}(F, Q) \;+\; \gamma,
\]
where each term is normalized to lie in $[0, 1]$:
\begin{itemize}[leftmargin=*,nosep]
  \item \textbf{Adherence.} Fraction of $Q$'s perimeter that lies within $\varepsilon$ of $F$'s boundary:
  \[
    \operatorname{adherence}(F, Q) \;=\; \frac{\mathcal{H}^1\!\left(\{x \in \partial Q : \operatorname{dist}(x, \partial F) \leq \varepsilon\}\right)}{\mathcal{H}^1(\partial Q)},
  \]
  where $\mathcal{H}^1$ is 1-dimensional Hausdorff measure (arc-length) and $\partial$ denotes polygon boundary. We use $\varepsilon = 5 \cdot 10^{-3}$ in normalized $[-1,1]$ coordinates.
  \item \textbf{Proximity.} Mean closeness of $Q$'s perimeter to $\partial F$, with an exponential decay:
  \[
    \operatorname{proximity}(F, Q) \;=\; \frac{1}{\mathcal{H}^1(\partial Q)} \int_{\partial Q} \exp\!\left(-\frac{\operatorname{dist}(x, \partial F)}{\tau}\right) \, d\mathcal{H}^1(x),
  \]
  with characteristic length $\tau = 0.05$.
\end{itemize}

Weights $(\alpha, \beta, \gamma) = (0.5, 0.2, 0.3)$ are fixed; defined in \code{src/geometry/rewards.py::compute\_reward}.  By construction $\operatorname{contact}(F, Q) \in [\gamma, \gamma + \alpha + \beta] = [0.3, 1.0]$ for any $Q \subseteq F$.

The \textbf{scalar reward} at a placement is
\[
  r(F, P, \theta, x, y) \;=\;
  \begin{cases}
    \exp\!\left(k \cdot \operatorname{contact}(F, P_{\theta,x,y})\right) & \text{if } (x,y) \in \operatorname{IFP}(F,P,\theta) \\
    0 & \text{otherwise}
  \end{cases}
\]
where $k = 10$ is a sharpening constant.  The exponentiation turns additive contact differences into multiplicative reward differences, so the softmax-normalized heatmap (Section~\ref{sec:target}) concentrates mass near the best-contact placements.

\paragraph{Reward heatmap $H_\theta$}
For fixed $(F, P, \theta)$, define the discrete reward heatmap as the function
\[
  H_\theta : \{0, \dots, 127\}^2 \to \mathbb{R}_{\geq 0}, \qquad H_\theta[r, c] \;=\; r\!\left(F,\; P,\; \theta,\; \operatorname{pix}^{-1}(r, c)\right).
\]
Concretely, $H_\theta[r, c]$ is the reward of placing $R_\theta(P)$ with its centroid at world coordinates
\[
  \operatorname{pix}^{-1}(r, c) \;=\; \left(\frac{2c}{127} - 1,\;\; 1 - \frac{2r}{127}\right) \in [-1, 1]^2.
\]
Properties:

\begin{center}
\begin{tabular}{ll}
  \toprule
  property & value \\
  \midrule
  Tensor shape                & $128 \times 128$ \\
  Support                     & rasterized $\operatorname{IFP}(F, P, \theta) \cap \operatorname{pix}^{-1}(\{0,\dots,127\}^2)$ \\
  Range on support            & $[\exp(0.3 k),\, \exp(k)] = [\exp(3),\, \exp(10)] \approx [20,\, 22\,026]$ \\
  Range outside support       & $\{0\}$ \\
  Stacked over 36 rotations   & $\mathbf{H} \in \mathbb{R}_{\geq 0}^{36 \times 128 \times 128}$ \\
  \bottomrule
\end{tabular}
\end{center}

The full per-pair reward tensor is $\mathbf{H} = (H_{\theta_0}, \dots, H_{\theta_{35}})$, computed once per pair in the precompute stage and stored as float16 in the training pkl.  The brute-force optimum is $r^\ast = \max_{i,r,c} \mathbf{H}[i, r, c]$.

\subsubsection{Optimization objective}

The brute-force optimum for a pair is
\[
  (\theta^\ast, x^\ast, y^\ast) \;=\; \arg\max_{(\theta, x, y) \in \Theta \times \Omega} \; r(F, P, \theta, x, y),
  \qquad r^\ast = r(F, P, \theta^\ast, x^\ast, y^\ast).
\]
Computing it requires evaluating Shapely on every valid pixel --- too expensive for inference.  The placement network $\pi_\phi$ instead predicts a near-optimal placement in a single forward pass per rotation.

\subsubsection{Quality metric}

For a predicted placement $(\hat\theta, \hat x, \hat y)$:
\[
  \operatorname{recovery} \;=\; \frac{r(F, P, \hat\theta, \hat x, \hat y)}{r^\ast} \;\in\; [0, 1].
\]
We report mean recovery on a 2200-pair validation set.

\subsection{Model}

\subsubsection{Inputs}

For a given query rotation $\theta_i$:
\begin{itemize}[leftmargin=*,nosep]
  \item $\mathbf{m}_F \in \{0,1\}^{128\times128}$: rasterized free-space mask.
  \item $\mathbf{m}_{P,\theta_i} \in \{0,1\}^{128\times128}$: rasterized mask of $R_{\theta_i}(P)$, with its centroid placed at the image center.
\end{itemize}
Stacked: $\mathbf{x}_i = \operatorname{stack}(\mathbf{m}_F, \mathbf{m}_{P,\theta_i}) \in \mathbb{R}^{2\times128\times128}$.

\subsubsection{Architecture: $\pi_\phi$}

A 4-level U-Net (\code{\_SmallUNet} in \code{src/models/neural\_bo\_policy.py}):
\[
  \pi_\phi : \mathbb{R}^{2\times128\times128} \to \mathbb{R}^{128\times128}.
\]

\paragraph{Building blocks}
Let $C \in \mathbb{R}^{c_{\text{in}} \times H \times W}$ denote a tensor with $c_{\text{in}}$ channels at spatial resolution $H \times W$.

\textbf{(a) ConvBlock} $\mathcal{C}_{c_{\text{out}}}$.  Two stacked stages of ($3{\times}3$ conv with same-padding, BatchNorm2d, ReLU):
\[
  \mathcal{C}_{c_{\text{out}}}(C) \;=\; \operatorname{ReLU}\!\left(\operatorname{BN}\!\left(W_2 \star \operatorname{ReLU}\!\left(\operatorname{BN}\!\left(W_1 \star C\right)\right)\right)\right),
\]
where $W_1 \in \mathbb{R}^{c_{\text{out}}\times c_{\text{in}}\times 3\times 3}$, $W_2 \in \mathbb{R}^{c_{\text{out}}\times c_{\text{out}}\times 3\times 3}$, and $\star$ denotes 2D convolution with padding 1 (output preserves $H, W$).

\textbf{(b) Down}: max-pool with stride 2:
\[
  \operatorname{Down}(C) \in \mathbb{R}^{c_{\text{in}}\times H/2 \times W/2}, \qquad
  \operatorname{Down}(C)[c, i, j] = \max_{a,b\in\{0,1\}} C[c,\, 2i+a,\, 2j+b].
\]

\textbf{(c) Up}: bilinear upsample by factor 2:
\[
  \operatorname{Up}(C) \in \mathbb{R}^{c_{\text{in}}\times 2H \times 2W}.
\]

\textbf{(d) SkipCat}: channel-wise concatenation with the corresponding encoder feature at matched resolution:
\[
  \operatorname{SkipCat}(U, E) \;=\; [U;\,E] \;\in\; \mathbb{R}^{(c_U+c_E)\times H \times W}.
\]

\paragraph{Forward pass}
With base width $b = 32$:

\textbf{Encoder.}
\[
\begin{aligned}
e_1 &= \mathcal{C}_{b}(\mathbf{x}_i)            && \in\,\mathbb{R}^{32\times128\times128}\\
e_2 &= \mathcal{C}_{2b}(\operatorname{Down}(e_1))     && \in\,\mathbb{R}^{64\times64\times64}\\
e_3 &= \mathcal{C}_{4b}(\operatorname{Down}(e_2))     && \in\,\mathbb{R}^{128\times32\times32}\\
e_4 &= \mathcal{C}_{8b}(\operatorname{Down}(e_3))     && \in\,\mathbb{R}^{256\times16\times16}
\end{aligned}
\]

\textbf{Bottleneck.}
\[
  g \;=\; \mathcal{C}_{8b}(e_4) \;\in\;\mathbb{R}^{256\times16\times16}.
\]

\textbf{Decoder.} (mirror of encoder, with skip concatenation)
\[
\begin{aligned}
d_3 &= \mathcal{C}_{4b}\!\left(\operatorname{SkipCat}(\operatorname{Up}(g),\,e_3)\right)   && \in\,\mathbb{R}^{128\times32\times32}\\
d_2 &= \mathcal{C}_{2b}\!\left(\operatorname{SkipCat}(\operatorname{Up}(d_3),\,e_2)\right) && \in\,\mathbb{R}^{64\times64\times64}\\
d_1 &= \mathcal{C}_{b}\!\left(\operatorname{SkipCat}(\operatorname{Up}(d_2),\,e_1)\right)  && \in\,\mathbb{R}^{32\times128\times128}
\end{aligned}
\]

\textbf{Output head.}  $1{\times}1$ convolution to a single channel:
\[
  \mathbf{z}_i \;=\; W_{\text{out}} \star d_1 \;\in\; \mathbb{R}^{1\times128\times128},
\]
with $W_{\text{out}} \in \mathbb{R}^{1\times32\times1\times1}$.  Squeezing the singleton channel gives $\mathbf{z}_i \in \mathbb{R}^{128\times128}$, the per-pixel logit map for rotation $\theta_i$.

\paragraph{Receptive field and parameter count}
Receptive field at the bottleneck spans the full input ($e_4$ has spatial $16{\times}16$ covering $128{\times}128$ -- each bottleneck cell ``sees'' all of the input through the 4-level downsampling). This is necessary because the optimum placement depends on global free-space geometry, not just local pixel neighborhoods.

Parameter table (with $b = 32$, $K = 9$ being the kernel-element count for a $3{\times}3$ conv):

\begin{center}
\begin{tabular}{lr}
  \toprule
  stage & params (approx) \\
  \midrule
  $\mathcal{C}_b$ on $c_\text{in}=2$           & $2 \cdot b \cdot K + b \cdot b \cdot K = 9\,792$ \\
  $\mathcal{C}_{2b}$ on $b$                    & $b \cdot 2b \cdot K + 2b \cdot 2b \cdot K = 55\,296$ \\
  $\mathcal{C}_{4b}$ on $2b$                   & $2b \cdot 4b \cdot K + 4b \cdot 4b \cdot K = 221\,184$ \\
  $\mathcal{C}_{8b}$ on $4b$                   & $4b \cdot 8b \cdot K + 8b \cdot 8b \cdot K = 884\,736$ \\
  Bottleneck $\mathcal{C}_{8b}$ on $8b$        & $8b \cdot 8b \cdot K + 8b \cdot 8b \cdot K = 1\,179\,648$ \\
  Decoder $\mathcal{C}_{4b}$ on $8b+4b$        & $12b \cdot 4b \cdot K + 4b \cdot 4b \cdot K = 663\,552$ \\
  Decoder $\mathcal{C}_{2b}$ on $4b+2b$        & $6b \cdot 2b \cdot K + 2b \cdot 2b \cdot K = 165\,888$ \\
  Decoder $\mathcal{C}_{b}$ on $2b+b$          & $3b \cdot b \cdot K + b \cdot b \cdot K = 36\,864$ \\
  Output $1{\times}1$ conv: $b \to 1$          & $32 + 1 = 33$ \\
  BatchNorm affines + biases (small)           & $\sim 6\,000$ \\
  \midrule
  \textbf{Total}                               & $\approx 3.13 \times 10^6$ \\
  \bottomrule
\end{tabular}
\end{center}

\subsubsection{Why this conditioning works}

The model never has to \emph{infer} rotation from the un-rotated part.  Conditioning the input on $\theta_i$ via $\mathbf{m}_{P,\theta_i}$ collapses the prediction task into a translation-equivariant question: ``given this oriented part and this free space, where does it fit best?'' This is the standard inductive bias of a U-Net.

\subsection{Training objective}

\subsubsection{Target distribution}\label{sec:target}

For each $(F, P, \theta_i)$ training example, define the \emph{normalized soft target}
\[
  \tilde{H}_{\theta_i}[r,c] \;=\; \frac{H_{\theta_i}[r,c]}{\sum_{r',c'} H_{\theta_i}[r',c']}.
\]
Since the precomputed $H_{\theta_i}$ uses $r = \exp(k \cdot \operatorname{contact})$, $\tilde{H}_{\theta_i}$ is a temperature-$1/k$ softmax of the underlying contact field over the IFP.

\subsubsection{Loss}

Let $\mathbf{p}_i = \operatorname{softmax}(\mathbf{z}_i) \in \Delta^{16383}$ (softmax over the 16\,384 flattened pixels).  The per-example soft cross-entropy loss is
\[
  \mathcal{L}_i(\phi) \;=\; -\sum_{r,c} \tilde{H}_{\theta_i}[r,c] \, \log \mathbf{p}_i[r,c].
\]
Equivalently, $\mathcal{L}_i = D_{\operatorname{KL}}(\tilde H_{\theta_i} \,\|\, \mathbf{p}_i) + \operatorname{H}(\tilde H_{\theta_i})$, where the entropy term is an example-dependent constant.

Compared to hard-label CE (one-hot at $\arg\max$), the soft loss provides gradient on every cell weighted by reward, which empirically lifts val recovery from $0.43 \to 0.72$ in our experiments.

\subsubsection{Training-step distribution}

Let $\mathcal{D} = \{(F_j, P_j, H^{(j)})\}_{j=1}^{N}$ be the corpus of $N = 22\,000$ training pairs (each with its 36-rotation heatmap tensor).  Sampling for each SGD step:
\[
  j \sim \operatorname{Uniform}\{1, \dots, N\}, \qquad i \sim \operatorname{Uniform}\{0, \dots, 35\},
\]
and the per-step empirical objective is $\mathbb{E}_{j,i}[\mathcal{L}_{i}(\phi; F_j, P_j)]$.  Roughly $N \cdot 36 = 7.92 \times 10^5$ distinct $(j, i)$ examples are addressable.

\subsubsection{Augmentation: dihedral group $D_2$}

Applied at the dataloader level.  Let $\sigma_h$ (horizontal flip) and $\sigma_v$ (vertical flip) act on the rasterized image plane.  Each is applied with probability $\tfrac12$, independently:
\begin{itemize}[leftmargin=*,nosep]
  \item $\sigma_h$: spatial axis-1 reversal; rotation reindex $i \mapsto (-i) \bmod 36$.
  \item $\sigma_v$: spatial axis-0 reversal; rotation reindex $i \mapsto (18 - i) \bmod 36$.
\end{itemize}

The transform is applied jointly to $(\mathbf{m}_F, \mathbf{m}_{P,\theta_i}, \tilde H_{\theta_i})$, preserving the input--target consistency.  Under reflection of the entire scene, the placement problem maps to itself, so the loss is unchanged in distribution.  Effective dataset multiplier: $\lvert D_2 \rvert = 4$.

\subsubsection{Optimization}

\begin{itemize}[leftmargin=*,nosep]
  \item Optimizer: AdamW, weight decay $10^{-4}$.
  \item Learning rate: $\eta_0 = 3 \times 10^{-4}$, cosine-annealed to 0 over $T = 8000$ steps.
  \item Batch size: 256.
  \item Mixed precision (FP16 amp).
  \item Hardware: Modal A100-80GB, $\sim 4.2$ steps/sec, $\sim 30$ min wall.
\end{itemize}

\subsection{Inference}\label{sec:inference}

For a deployment query on $(F, P)$:
\begin{enumerate}[leftmargin=*,nosep]
  \item Rasterize $\mathbf{m}_F$.
  \item For each $i \in \{0, \dots, 35\}$: rasterize $\mathbf{m}_{P,\theta_i}$, run $\mathbf{z}_i = \pi_\phi(\mathbf{x}_i)$.
  \item Stack: $\mathbf{Z} = [\mathbf{z}_0, \dots, \mathbf{z}_{35}] \in \mathbb{R}^{36\times128\times128}$.
  \item Predict $(\hat\imath, \hat r, \hat c) = \arg\max_{i,r,c} \mathbf{Z}[i,r,c]$, giving $\hat\theta = \theta_{\hat\imath}$, $(\hat x, \hat y) = \operatorname{pix}^{-1}(\hat r, \hat c)$.
  \item \textbf{Optional refinement.}  Evaluate Shapely reward on a local window
  \[
    \mathcal{W} \;=\; \left\{(\theta_{\hat\imath + \delta_\theta},\, \hat r + \delta_r,\, \hat c + \delta_c) : \lvert\delta_\theta\rvert \leq K,\ \lvert\delta_r\rvert, \lvert\delta_c\rvert \leq N\right\}
  \]
  and return $\arg\max_{\mathcal W} r(\cdot)$.  With $N = 10, K = 2$, $\lvert\mathcal{W}\rvert = 5 \cdot 21^2 = 2205$ Shapely calls at $\sim 75\,\mu s$ each $\approx 165$ ms.  Closes the model-vs-brute-force gap.
\end{enumerate}

Latency without refinement: 36 forward passes on a 3M-param U-Net, dominated by per-rotation Shapely-free rasterization $\sim 30$ ms total on a CPU; the pure model forward passes are $< 50$ ms on GPU.

\subsection{Data pipeline}

\subsubsection{Pair generation}
\begin{itemize}[leftmargin=*,nosep]
  \item \textbf{Convex--convex.}  Random convex hulls from $n$ uniform points; 12\,000 pairs sampled by \code{generate\_bc\_dataset.py}.
  \item \textbf{Concave-fs / convex-part.}  From \code{bo\_train\_pool\_10k.pkl}: 10\,000 pairs with non-convex $F$ (random star polygons) and convex $P$.
\end{itemize}

\subsubsection{IFP precompute}
For each pair and each $\theta_i$: compute $\operatorname{IFP}(F, P, \theta_i)$ via Minkowski erosion using \code{pyclipper} (\code{src/geometry/ifp.py}).  Rasterize to $\{0,1\}^{128\times128}$.

\subsubsection{Reward heatmap precompute}
For each pixel inside the rasterized IFP, evaluate $r(F, P, \theta_i, x, y)$ via Shapely.  Pixels outside IFP are 0.  Stored as $(36, 128, 128)$ float32 per pair.

Total Shapely queries: $\sim 7.1 \times 10^8$ for convex + $\sim 3.1 \times 10^8$ for concave $\approx 10^9$ queries.  Executed in parallel across 100 Modal containers $\times$ 8 cores.  Total wall $\approx 25$ minutes; cost $\approx \$40$.

\subsubsection{Rotated-part-mask precompute}
For each pair: rasterize $R_{\theta_i}(P)$ centered at origin, for $i = 0, \dots, 35$.  Stored as $(36, 128, 128)$ uint8 per pair.  Used as the second input channel at training and inference.

\subsubsection{Combined corpus}
$N_{\text{train}} = 19\,800$ pairs (10\,800 convex + 9\,000 concave), $N_{\text{val}} = 2\,200$ (1200 convex + 1000 concave).  Combined pkl: $26.68$ GB (fp16 heatmaps + uint8 masks).

\subsection{Results}

Model selection: best mean val recovery on a fixed 2200-pair validation set, evaluated every 250 steps.  Compare model-only recovery (no Shapely refine):

\begin{center}
\begin{tabular}{lr}
  \toprule
  Configuration & Val recovery \\
  \midrule
  Hard label, hierarchical (tile + cell)                          & 0.04 \\
  Soft label, hierarchical                                         & 0.43 \\
  Soft label, hierarchical, tile-only                              & 0.32 \\
  \textbf{Per-(pair, $\theta$) U-Net, convex-only ($N=10\,800$)}   & \textbf{0.72} \\
  \textbf{Per-(pair, $\theta$) U-Net, combined ($N=19\,800$)}      & \textbf{0.73} \\
  \bottomrule
\end{tabular}
\end{center}

The convex-only and combined models are nearly tied on a mixed val set; combined generalizes to concave free spaces from a single checkpoint.  With Shapely refinement, both push to near-$1.0$ recovery (analytically: refinement window contains the true argmax with high probability if the model lands within $\pm 10$ pixels and $\pm 2$ rotation bins).

\subsection{Implementation pointers}

\begin{center}
\begin{tabular}{ll}
  \toprule
  concern & file \\
  \midrule
  Model                          & \code{src/models/neural\_bo\_policy.py::\_SmallUNet} \\
  Training script                & \code{scripts/train\_perthet.py} \\
  Modal training entrypoint      & \code{modal\_train\_perthet.py} \\
  IFP exact (handles concave fs) & \code{src/geometry/ifp.py::compute\_ifp\_exact} \\
  Reward function                & \code{src/geometry/rewards.py::compute\_reward\_exp} \\
  Concave precompute             & \code{modal\_concave\_precompute.py} \\
  Combined-pkl build             & \code{modal\_build\_combined.py} \\
  Best ckpt --- convex           & \code{checkpoints/perthet/final.pt} \\
  Best ckpt --- combined         & \code{checkpoints/perthet\_combined/final.pt} \\
  \bottomrule
\end{tabular}
\end{center}
"""


def _tex_escape(s):
    return str(s).replace("_", r"\_")


def latex_example_section(row, fig_relpath, refine_pixels, refine_thetas):
    t_total = row["t_raster_ms"] + row["t_forward_ms"] + row["t_refine_ms"]
    n_calls_target = (2 * refine_pixels + 1) ** 2 * (2 * refine_thetas + 1)
    return r"""
\subsection{Pair """ + _tex_escape(row["idx"]) + r""" \quad\small\textmd{(area ratio """ + f"{row['area_ratio']:.3f}" + r""")}}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{""" + fig_relpath + r"""}
\end{figure}

\begin{center}\small
\begin{tabular}{lrrrr}
  \toprule
  Method & $\theta$ (deg) & $(x, y)$ & Reward & Recovery \\
  \midrule
  Brute force ($r^\ast$)    & """ + f"{row['bf_t_deg']:.0f}" + r""" & $(""" + f"{row['x_star']:+.3f}, {row['y_star']:+.3f}" + r""")$ & """ + f"{row['r_star']:.3f}" + r""" & 1.000 \\
  Model only                & """ + f"{row['t0_deg']:.0f}"  + r""" & $(""" + f"{row['x0']:+.3f}, {row['y0']:+.3f}" + r""")$       & """ + f"{row['r_noref']:.3f}" + r""" & """ + f"{row['rec_noref']:.3f}" + r""" \\
  Refined ($\pm""" + str(refine_pixels) + r"""$\,px, $\pm""" + str(refine_thetas) + r"""$\,$\theta$-bins) & """ + f"{row['ref_t_deg']:.0f}" + r""" & $(""" + f"{row['ref_x']:+.3f}, {row['ref_y']:+.3f}" + r""")$ & """ + f"{row['r_ref']:.3f}" + r""" & """ + f"{row['rec_ref']:.3f}" + r""" \\
  \bottomrule
\end{tabular}
\end{center}

\begin{center}\small
\begin{tabular}{lr}
  \toprule
  Cost & Value \\
  \midrule
  Rasterize (36 rotated part masks)              & """ + f"{row['t_raster_ms']:.1f}"  + r""" ms \\
  Model forward (36 passes, batched)             & """ + f"{row['t_forward_ms']:.1f}" + r""" ms \\
  Shapely refinement (""" + str(row['n_calls']) + r""" calls, target """ + str(n_calls_target) + r""")  & """ + f"{row['t_refine_ms']:.1f}"  + r""" ms \\
  \midrule
  Total                                          & """ + f"{t_total:.1f}" + r""" ms \\
  \bottomrule
\end{tabular}
\end{center}
"""


def latex_summary_section(rows, fig_relpath, refine_pixels, refine_thetas):
    deltas = np.array([r["rec_ref"] - r["rec_noref"] for r in rows])
    rec_noref = np.array([r["rec_noref"] for r in rows])
    rec_ref = np.array([r["rec_ref"] for r in rows])
    return r"""
\section{Summary}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{""" + fig_relpath + r"""}
\caption{Left: recovery $r / r^\ast$ per pair (model-only vs.\ refined). Right: refinement uplift sorted descending.}
\end{figure}

\begin{center}\small
\begin{tabular}{lr}
  \toprule
  Statistic & Value \\
  \midrule
  Pairs reported                          & """ + str(len(rows)) + r""" \\
  Mean recovery (model only)              & """ + f"{rec_noref.mean():.4f}" + r""" \\
  Mean recovery (refined)                 & """ + f"{rec_ref.mean():.4f}" + r""" \\
  Median recovery (model only)            & """ + f"{float(np.median(rec_noref)):.4f}" + r""" \\
  Median recovery (refined)               & """ + f"{float(np.median(rec_ref)):.4f}" + r""" \\
  Mean $\Delta$ recovery                  & """ + f"{deltas.mean():+.4f}" + r""" \\
  Pairs where refinement helped           & """ + f"{int((deltas > 0).sum())}/{len(rows)}" + r""" \\
  \midrule
  Mean time per pair (rasterize)          & """ + f"{np.mean([r['t_raster_ms'] for r in rows]):.1f}"  + r""" ms \\
  Mean time per pair (model forward)      & """ + f"{np.mean([r['t_forward_ms'] for r in rows]):.1f}" + r""" ms \\
  Mean time per pair (refinement)         & """ + f"{np.mean([r['t_refine_ms'] for r in rows]):.1f}"  + r""" ms \\
  Mean Shapely calls per pair             & """ + f"{np.mean([r['n_calls'] for r in rows]):.0f}"      + r""" (expected """ + f"{(2*refine_pixels+1)**2 * (2*refine_thetas+1)}" + r""") \\
  \bottomrule
\end{tabular}
\end{center}
"""


def write_tex(out_tex, ckpt, convex_rows, concave_rows,
              refine_pixels, refine_thetas):
    n_calls = (2 * refine_pixels + 1) ** 2 * (2 * refine_thetas + 1)
    parts = [LATEX_PREAMBLE]
    parts.append(r"""
\title{\textsf{\Huge Placement Pipeline}\\[0.3em]\textsf{\Large Visualization Report}}
\author{August Dua\\\textit{\normalsize MSc Math, TUM (Candidate)}}
\date{\today}

\begin{document}
\maketitle

\begin{center}
\small
Checkpoint: \texttt{""" + ckpt.replace("_", r"\_") + r"""}\\[0.3em]
""" + str(len(convex_rows)) + r""" convex val pairs (orig\_idx $\in [""" + str(CONVEX_VAL_START) + ", " + str(CONVEX_VAL_END) + r""")$),
""" + str(len(concave_rows)) + r""" concave-fs val pairs (\code{bo\_train\_pool\_10k} position $\in [""" + str(CONCAVE_VAL_START) + ", " + str(CONCAVE_VAL_END) + r""")$).\\
Refinement window: $\pm""" + str(refine_pixels) + r"""$\,px $\times \pm""" + str(refine_thetas) + r"""$\,$\theta$-bins
$\Rightarrow$ """ + str(n_calls) + r""" Shapely calls per pair.
\end{center}

\tableofcontents
\newpage
""")
    parts.append(METHODOLOGY_LATEX)

    if convex_rows:
        parts.append(r"""
\newpage
\section{Examples --- Convex--convex pairs}
The free-space polygon and the part polygon are both convex.  Brute-force ground truth $r^\ast$ comes from precomputed heatmap chunks (\code{data/reward\_heatmaps\_exp\_k10\_inside.npy\_chunks/}). For each pair we show, top row left to right: the model's predicted reward map at $\hat\theta$ (with the refinement window outlined and the refined target circled), the brute-force ground-truth reward map at $\theta^\ast$, and the part polygon (centered).  Bottom row: the model-only placement, the placement after Shapely refinement, and the brute-force placement.
""")
        for row in convex_rows:
            fig_rel = f"figs/{row['idx']}.png"
            parts.append(latex_example_section(row, fig_rel,
                                               refine_pixels, refine_thetas))

    if concave_rows:
        parts.append(r"""
\newpage
\section{Examples --- Concave free-space pairs}
The free-space polygon is non-convex (random star polygon from \code{bo\_train\_pool\_10k.pkl}); the part remains convex.  The model and refinement use the same pipeline; brute-force $r^\ast$ is computed on the fly via \code{compute\_ifp\_exact} + exhaustive Shapely scoring inside the IFP (Section \ref{sec:inference}).  Same six-panel layout as before.
""")
        for row in concave_rows:
            fig_rel = f"figs/{row['idx']}.png"
            parts.append(latex_example_section(row, fig_rel,
                                               refine_pixels, refine_thetas))

    all_rows = convex_rows + concave_rows
    if all_rows:
        parts.append(latex_summary_section(all_rows, "figs/summary.png",
                                           refine_pixels, refine_thetas))
    parts.append("\n\\end{document}\n")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def compile_pdf(tex_path):
    out_dir = os.path.dirname(tex_path) or "."
    name = os.path.basename(tex_path)
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        name,
    ]
    # Two passes for TOC.
    for i in (1, 2):
        print(f"  pdflatex pass {i} ...", flush=True)
        res = subprocess.run(
            cmd, cwd=out_dir, capture_output=True, text=True, timeout=600
        )
        if res.returncode != 0:
            print("--- pdflatex stdout (tail) ---", flush=True)
            print("\n".join(res.stdout.splitlines()[-40:]), flush=True)
            print("--- pdflatex stderr (tail) ---", flush=True)
            print("\n".join(res.stderr.splitlines()[-40:]), flush=True)
            raise RuntimeError(f"pdflatex failed (pass {i})")
    # Cleanup aux files.
    base, _ = os.path.splitext(name)
    for ext in (".aux", ".log", ".out", ".toc"):
        p = os.path.join(out_dir, base + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/perthet_combined/final.pt")
    ap.add_argument("--convex-source",
                    default="data/bc_snapshot_raster128.pkl")
    ap.add_argument("--concave-source",
                    default="data/bo_train_pool_10k.pkl")
    ap.add_argument("--out-dir", default="visualizations/report")
    ap.add_argument("--n-convex", type=int, default=12)
    ap.add_argument("--n-concave", type=int, default=6)
    ap.add_argument("--refine-pixels", type=int, default=10)
    ap.add_argument("--refine-thetas", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true",
                    help="Emit .tex + figures but skip pdflatex.")
    args = ap.parse_args()

    figs_dir = os.path.join(args.out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    print(f"Loading model from {args.ckpt} ...", flush=True)
    pm = PlacementModel(ckpt=args.ckpt, device=args.device)
    print(f"  device={pm.device}", flush=True)

    rng = np.random.default_rng(args.seed)

    # ---- Convex examples ----
    convex_rows = []
    if args.n_convex > 0:
        print(f"Loading convex source {args.convex_source} ...", flush=True)
        with open(args.convex_source, "rb") as f:
            convex_records = pickle.load(f)
        print(f"  {len(convex_records)} pairs", flush=True)

        pool = list(range(CONVEX_VAL_START,
                          min(CONVEX_VAL_END, len(convex_records))))
        picks = rng.choice(pool,
                           size=min(args.n_convex, len(pool)),
                           replace=False)
        picks = sorted(int(p) for p in picks)
        print(f"  convex picks: {picks}", flush=True)

        for k, idx in enumerate(picks):
            label = f"convex_{idx:05d}"
            fig_path = os.path.join(figs_dir, f"{label}.png")
            print(f"  [convex {k+1}/{len(picks)}] pair {idx}",
                  flush=True)
            bf_tuple = load_brute_force_convex(idx)
            row = render_pair_figure(fig_path, pm, convex_records[idx],
                                     label, bf_tuple,
                                     args.refine_pixels, args.refine_thetas)
            convex_rows.append(row)
        del convex_records  # free RAM before loading concave source

    # ---- Concave examples (brute force computed on the fly) ----
    concave_rows = []
    if args.n_concave > 0:
        print(f"Loading concave source {args.concave_source} ...", flush=True)
        with open(args.concave_source, "rb") as f:
            concave_records = pickle.load(f)
        print(f"  {len(concave_records)} pairs", flush=True)

        pool = list(range(CONCAVE_VAL_START,
                          min(CONCAVE_VAL_END, len(concave_records))))
        picks = rng.choice(pool,
                           size=min(args.n_concave, len(pool)),
                           replace=False)
        picks = sorted(int(p) for p in picks)
        print(f"  concave picks: {picks}", flush=True)

        for k, idx in enumerate(picks):
            label = f"concave_{idx:05d}"
            fig_path = os.path.join(figs_dir, f"{label}.png")
            rec = concave_records[idx]
            fs = wkt_loads(rec["fs_poly_wkt"])
            part = wkt_loads(rec["part_poly_wkt"])
            t0 = time.time()
            print(f"  [concave {k+1}/{len(picks)}] pair {idx}: "
                  f"computing brute-force heatmap ...", flush=True)
            bf_tuple = compute_brute_force_on_the_fly(fs, part, k=10.0)
            print(f"    bf done in {time.time()-t0:.1f}s, "
                  f"r* = {bf_tuple[-1]:.3f}", flush=True)
            row = render_pair_figure(fig_path, pm, rec, label, bf_tuple,
                                     args.refine_pixels, args.refine_thetas)
            concave_rows.append(row)

    all_rows = convex_rows + concave_rows
    if all_rows:
        sum_path = os.path.join(figs_dir, "summary.png")
        print(f"Summary figure -> {sum_path}", flush=True)
        render_summary_figure(sum_path, all_rows)

    tex_path = os.path.join(args.out_dir, "placement_pipeline_report.tex")
    print(f"Writing {tex_path} ...", flush=True)
    write_tex(tex_path, args.ckpt, convex_rows, concave_rows,
              args.refine_pixels, args.refine_thetas)

    if args.no_compile:
        print("Skipping pdflatex (--no-compile).", flush=True)
        return

    print(f"Compiling {tex_path} via pdflatex ...", flush=True)
    compile_pdf(tex_path)
    pdf_path = tex_path.replace(".tex", ".pdf")
    print(f"Done: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
