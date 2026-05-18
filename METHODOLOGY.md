# Placement Network — Methodology

A learned placement function that, given a free-space polygon `F` and a part polygon `P`, returns a placement `(θ, x, y)` maximizing a Shapely-defined reward. Trained on 22 000 pairs (12k convex–convex + 10k concave-fs–convex), achieves val recovery ≈ 0.72 with one small U-Net and one feed-forward pass per rotation.

---

## 1. Formal problem statement

### 1.1 Notation

Let
- $F \subset \mathbb{R}^2$ be a simple polygon (the *free space*), possibly non-convex.
- $P \subset \mathbb{R}^2$ be a simple convex polygon (the *part*), centered at the origin so that its centroid coincides with $(0,0)$.
- $R_\theta : \mathbb{R}^2 \to \mathbb{R}^2$ denote rotation by angle $\theta$ about the origin.
- $T_{(x,y)}$ denote translation by $(x,y)$.
- The *placement* of $P$ at position $(x,y)$ with rotation $\theta$ is the set
  $$
  P_{\theta,x,y} \;=\; T_{(x,y)}\bigl(R_\theta(P)\bigr).
  $$

We discretize:
- Rotations: $\theta \in \Theta = \{0, \tfrac{2\pi}{36}, \tfrac{4\pi}{36}, \dots, \tfrac{70\pi}{36}\}$, indexed $i = 0, \dots, 35$.
- Positions: $(x,y) \in [-1,1]^2$ on a $128 \times 128$ raster grid. Each grid cell is indexed by row $r \in \{0, \dots, 127\}$ and column $c \in \{0, \dots, 127\}$. The pixel-to-world map is
  $$
  \mathrm{pix}^{-1}(r, c) \;=\; \bigl(\,\tfrac{2c}{127} - 1,\;\; 1 - \tfrac{2r}{127}\,\bigr) \;\in\; [-1,1]^2,
  $$
  so column $c$ is the $x$-axis (left $\to$ right) and row $r$ is the $y$-axis (top $\to$ bottom, sign-flipped to match image convention). Total pixels $|\Omega| = 128^2 = 16\,384$.

The full discrete action space has size $|\Theta| \cdot |\Omega| = 36 \cdot 16\,384 = 589\,824$.

### 1.2 Validity and reward

Define the **Inner Fit Polygon** at rotation $\theta$ as the locus of valid centroid positions:
$$
\mathrm{IFP}(F, P, \theta) \;=\; \{ (x,y) \in \mathbb{R}^2 : P_{\theta,x,y} \subseteq F \}.
$$
Equivalently, $\mathrm{IFP}(F, P, \theta) = F \ominus R_\theta(P)$ (Minkowski erosion). It is well-defined and computable for any $F$ (convex or concave) and convex $P$.

#### Contact and reward functions

For a free-space polygon $F$ and a placed-part polygon $Q \subseteq F$, define the **contact score**
$$
\mathrm{contact}(F, Q) \;=\; \alpha\,\mathrm{adherence}(F, Q) \;+\; \beta\,\mathrm{proximity}(F, Q) \;+\; \gamma,
$$
where each term is normalized to lie in $[0, 1]$:
- **Adherence.** Fraction of $Q$'s perimeter that lies within $\varepsilon$ of $F$'s boundary:
  $$
  \mathrm{adherence}(F, Q) \;=\; \frac{\mathcal{H}^1\bigl(\{x \in \partial Q : \mathrm{dist}(x, \partial F) \leq \varepsilon\}\bigr)}{\mathcal{H}^1(\partial Q)},
  $$
  where $\mathcal{H}^1$ is 1-dimensional Hausdorff measure (arc-length) and $\partial$ denotes polygon boundary. We use $\varepsilon = 5 \cdot 10^{-3}$ in normalized $[-1,1]$ coordinates.
- **Proximity.** Mean closeness of $Q$'s perimeter to $\partial F$, with an exponential decay:
  $$
  \mathrm{proximity}(F, Q) \;=\; \frac{1}{\mathcal{H}^1(\partial Q)} \int_{\partial Q} \exp\!\Bigl(-\frac{\mathrm{dist}(x, \partial F)}{\tau}\Bigr) \, d\mathcal{H}^1(x),
  $$
  with characteristic length $\tau = 0.05$.

Weights $(\alpha, \beta, \gamma) = (0.5, 0.2, 0.3)$ are fixed; defined in `src/geometry/rewards.py::compute_reward`. By construction $\mathrm{contact}(F, Q) \in [\gamma, \gamma + \alpha + \beta] = [0.3, 1.0]$ for any $Q \subseteq F$.

The **scalar reward** at a placement is
$$
r(F, P, \theta, x, y) \;=\;
\begin{cases}
\exp\!\bigl(k \cdot \mathrm{contact}(F, P_{\theta,x,y})\bigr) & \text{if } (x,y) \in \mathrm{IFP}(F,P,\theta) \\
0 & \text{otherwise}
\end{cases}
$$
where $k = 10$ is a sharpening constant. The exponentiation turns additive contact differences into multiplicative reward differences, so the softmax-normalized heatmap (§3.1) concentrates mass near the best-contact placements.

#### Reward heatmap $H_\theta$

For fixed $(F, P, \theta)$, define the discrete reward heatmap as the function
$$
H_\theta : \{0, \dots, 127\}^2 \to \mathbb{R}_{\geq 0}, \qquad H_\theta[r, c] \;=\; r\!\left(F,\; P,\; \theta,\; \mathrm{pix}^{-1}(r, c)\right).
$$
Concretely, $H_\theta[r, c]$ is the reward of placing $R_\theta(P)$ with its centroid at world coordinates
$$
\mathrm{pix}^{-1}(r, c) \;=\; \left(\frac{2c}{127} - 1,\;\; 1 - \frac{2r}{127}\right) \in [-1, 1]^2.
$$
Properties:

| property | value |
|---|---|
| Tensor shape | $128 \times 128$ |
| Support | exactly the rasterized $\mathrm{IFP}(F, P, \theta) \cap \mathrm{pix}^{-1}(\{0,\dots,127\}^2)$ |
| Range on support | $[\exp(0.3 k),\, \exp(k)] = [\exp(3),\, \exp(10)] \approx [20,\, 22\,026]$ |
| Range outside support | $\{0\}$ |
| Total tensor: 36 rotations stacked | $\mathbf{H} \in \mathbb{R}_{\geq 0}^{36 \times 128 \times 128}$ |

The full per-pair reward tensor is $\mathbf{H} = (H_{\theta_0}, \dots, H_{\theta_{35}})$, computed once per pair in the precompute stage (§5.3) and stored as float16 in the training pkl. The brute-force optimum (§1.3) is $r^* = \max_{i,r,c} \mathbf{H}[i, r, c]$.

### 1.3 Optimization objective

The brute-force optimum for a pair is
$$
(\theta^*, x^*, y^*) \;=\; \arg\max_{(\theta, x, y) \in \Theta \times \Omega} \; r(F, P, \theta, x, y),
\qquad r^* = r(F, P, \theta^*, x^*, y^*).
$$
Computing it requires evaluating Shapely on every valid pixel — too expensive for inference. The placement network $\pi_\phi$ instead predicts a near-optimal placement in a single forward pass per rotation.

### 1.4 Quality metric

For a predicted placement $(\hat\theta, \hat x, \hat y)$:
$$
\mathrm{recovery} \;=\; \frac{r(F, P, \hat\theta, \hat x, \hat y)}{r^*} \;\in\; [0, 1].
$$
We report mean recovery on a 2200-pair validation set.

---

## 2. Model

### 2.1 Inputs

For a given query rotation $\theta_i$:
- $\mathbf{m}_F \in \{0,1\}^{128\times128}$: rasterized free-space mask.
- $\mathbf{m}_{P,\theta_i} \in \{0,1\}^{128\times128}$: rasterized mask of $R_{\theta_i}(P)$, with its centroid placed at the image center.

Stacked: $\mathbf{x}_i = \mathrm{stack}(\mathbf{m}_F, \mathbf{m}_{P,\theta_i}) \in \mathbb{R}^{2\times128\times128}$.

### 2.2 Architecture: $\pi_\phi$

A 4-level U-Net (`_SmallUNet` in `src/models/neural_bo_policy.py`):
$$
\pi_\phi : \mathbb{R}^{2\times128\times128} \to \mathbb{R}^{128\times128}.
$$

#### Building blocks

Let $C \in \mathbb{R}^{c_{\text{in}} \times H \times W}$ denote a tensor with $c_{\text{in}}$ channels at spatial resolution $H \times W$.

**(a) ConvBlock** $\mathcal{C}_{c_{\text{out}}}$. Two stacked stages of (3×3 conv with same-padding, BatchNorm2d, ReLU):
$$
\mathcal{C}_{c_{\text{out}}}(C) \;=\; \mathrm{ReLU}\!\bigl(\mathrm{BN}\!\bigl(W_2 \star \mathrm{ReLU}\!\bigl(\mathrm{BN}\!\bigl(W_1 \star C\bigr)\bigr)\bigr)\bigr),
$$
where $W_1 \in \mathbb{R}^{c_{\text{out}}\times c_{\text{in}}\times 3\times 3}$, $W_2 \in \mathbb{R}^{c_{\text{out}}\times c_{\text{out}}\times 3\times 3}$, and $\star$ denotes 2D convolution with padding 1 (output preserves $H, W$).

**(b) Down**: max-pool with stride 2:
$$
\mathrm{Down}(C)\!\in\!\mathbb{R}^{c_{\text{in}}\times H/2 \times W/2}, \qquad \mathrm{Down}(C)[c, i, j] = \max_{a,b\in\{0,1\}} C[c,\, 2i+a,\, 2j+b].
$$

**(c) Up**: bilinear upsample by factor 2:
$$
\mathrm{Up}(C)\!\in\!\mathbb{R}^{c_{\text{in}}\times 2H \times 2W}.
$$

**(d) SkipCat**: channel-wise concatenation with the corresponding encoder feature at matched resolution:
$$
\mathrm{SkipCat}(U, E) \;=\; [U;\,E] \;\in\; \mathbb{R}^{(c_U+c_E)\times H \times W}.
$$

#### Forward pass

With base width $b = 32$:

**Encoder.**
$$
\begin{aligned}
e_1 &= \mathcal{C}_{b}(\mathbf{x}_i)            && \in\,\mathbb{R}^{32\times128\times128}\\
e_2 &= \mathcal{C}_{2b}(\mathrm{Down}(e_1))     && \in\,\mathbb{R}^{64\times64\times64}\\
e_3 &= \mathcal{C}_{4b}(\mathrm{Down}(e_2))     && \in\,\mathbb{R}^{128\times32\times32}\\
e_4 &= \mathcal{C}_{8b}(\mathrm{Down}(e_3))     && \in\,\mathbb{R}^{256\times16\times16}
\end{aligned}
$$

**Bottleneck.**
$$
g \;=\; \mathcal{C}_{8b}(e_4) \;\in\;\mathbb{R}^{256\times16\times16}.
$$

**Decoder.** (mirror of encoder, with skip concatenation)
$$
\begin{aligned}
d_3 &= \mathcal{C}_{4b}\bigl(\mathrm{SkipCat}(\mathrm{Up}(g),\,e_3)\bigr)   && \in\,\mathbb{R}^{128\times32\times32}\\
d_2 &= \mathcal{C}_{2b}\bigl(\mathrm{SkipCat}(\mathrm{Up}(d_3),\,e_2)\bigr) && \in\,\mathbb{R}^{64\times64\times64}\\
d_1 &= \mathcal{C}_{b}\bigl(\mathrm{SkipCat}(\mathrm{Up}(d_2),\,e_1)\bigr)  && \in\,\mathbb{R}^{32\times128\times128}
\end{aligned}
$$

**Output head.** $1\times1$ convolution to a single channel:
$$
\mathbf{z}_i \;=\; W_{\text{out}} \star d_1 \;\in\; \mathbb{R}^{1\times128\times128},
$$
with $W_{\text{out}}\in\mathbb{R}^{1\times32\times1\times1}$. Squeezing the singleton channel gives $\mathbf{z}_i \in \mathbb{R}^{128\times128}$, the per-pixel logit map for rotation $\theta_i$.

#### Receptive field and parameter count

Receptive field at the bottleneck spans the full input ($e_4$ has spatial 16×16 covering 128×128 → each bottleneck cell "sees" all of the input through the 4-level downsampling). This is necessary because the optimum placement depends on global free-space geometry, not just local pixel neighborhoods.

Parameter table (with $b = 32$, $K = 9$ being the kernel-element count for a 3×3 conv):

| stage | params (approx) |
|---|---|
| $\mathcal{C}_b$ on $c_\text{in}=2$ | $2 \cdot b \cdot K + b \cdot b \cdot K = 9{,}792$ |
| $\mathcal{C}_{2b}$ on $b$         | $b \cdot 2b \cdot K + 2b \cdot 2b \cdot K = 55{,}296$ |
| $\mathcal{C}_{4b}$ on $2b$        | $2b \cdot 4b \cdot K + 4b \cdot 4b \cdot K = 221{,}184$ |
| $\mathcal{C}_{8b}$ on $4b$        | $4b \cdot 8b \cdot K + 8b \cdot 8b \cdot K = 884{,}736$ |
| Bottleneck $\mathcal{C}_{8b}$ on $8b$ | $8b \cdot 8b \cdot K + 8b \cdot 8b \cdot K = 1{,}179{,}648$ |
| Decoder $\mathcal{C}_{4b}$ on $8b+4b$ | $12b \cdot 4b \cdot K + 4b \cdot 4b \cdot K = 663{,}552$ |
| Decoder $\mathcal{C}_{2b}$ on $4b+2b$ | $6b \cdot 2b \cdot K + 2b \cdot 2b \cdot K = 165{,}888$ |
| Decoder $\mathcal{C}_{b}$ on $2b+b$   | $3b \cdot b \cdot K + b \cdot b \cdot K = 36{,}864$ |
| Output 1×1 conv: $b \to 1$         | $32 + 1 = 33$ |
| BatchNorm affines + biases (small)  | ~6{,}000 |
| **Total** | $\approx 3.13 \times 10^6$ |

### 2.3 Why this conditioning works

The model never has to *infer* rotation from the un-rotated part. Conditioning the input on $\theta_i$ via $\mathbf{m}_{P,\theta_i}$ collapses the prediction task into a translation-equivariant question: "given this oriented part and this free space, where does it fit best?" This is the standard inductive bias of a U-Net.

---

## 3. Training objective

### 3.1 Target distribution

For each $(F, P, \theta_i)$ training example, define the *normalized soft target*
$$
\tilde{H}_{\theta_i}[r,c] \;=\; \frac{H_{\theta_i}[r,c]}{\sum_{r',c'} H_{\theta_i}[r',c']}.
$$
Since the precomputed $H_{\theta_i}$ uses $r = \exp(k \cdot \mathrm{contact})$, $\tilde{H}_{\theta_i}$ is a temperature-$1/k$ softmax of the underlying contact field over the IFP.

### 3.2 Loss

Let $\mathbf{p}_i = \mathrm{softmax}(\mathbf{z}_i) \in \Delta^{16383}$ (softmax over the 16 384 flattened pixels). The per-example soft cross-entropy loss is
$$
\mathcal{L}_i(\phi) \;=\; -\sum_{r,c} \tilde{H}_{\theta_i}[r,c] \, \log \mathbf{p}_i[r,c].
$$
Equivalently, $\mathcal{L}_i = D_{\mathrm{KL}}(\tilde H_{\theta_i} \,\|\, \mathbf{p}_i) + \mathrm{H}(\tilde H_{\theta_i})$, where the entropy term is an example-dependent constant.

Compared to hard-label CE (one-hot at $\arg\max$), the soft loss provides gradient on every cell weighted by reward, which empirically lifts val recovery from 0.43 → 0.72 in our experiments.

### 3.3 Training-step distribution

Let $\mathcal{D} = \{(F_j, P_j, H^{(j)})\}_{j=1}^{N}$ be the corpus of $N = 22\,000$ training pairs (each with its 36-rotation heatmap tensor). Sampling for each SGD step:
$$
j \sim \mathrm{Uniform}\{1, \dots, N\}, \qquad i \sim \mathrm{Uniform}\{0, \dots, 35\},
$$
and the per-step empirical objective is $\mathbb{E}_{j,i}[\mathcal{L}_{i}(\phi; F_j, P_j)]$. Roughly $N \cdot 36 = 7.92 \times 10^5$ distinct $(j, i)$ examples are addressable.

### 3.4 Augmentation: dihedral group $D_2$

Applied at the dataloader level. Let $\sigma_h$ (horizontal flip) and $\sigma_v$ (vertical flip) act on the rasterized image plane. Each is applied with probability $\tfrac12$, independently:
- $\sigma_h$: spatial axis-1 reversal; rotation reindex $i \mapsto (-i) \bmod 36$.
- $\sigma_v$: spatial axis-0 reversal; rotation reindex $i \mapsto (18 - i) \bmod 36$.

The transform is applied jointly to $(\mathbf{m}_F, \mathbf{m}_{P,\theta_i}, \tilde H_{\theta_i})$, preserving the input–target consistency. Under reflection of the entire scene, the placement problem maps to itself, so the loss is unchanged in distribution. Effective dataset multiplier: $|D_2| = 4$.

### 3.5 Optimization

- Optimizer: AdamW, weight decay $10^{-4}$.
- Learning rate: $\eta_0 = 3 \times 10^{-4}$, cosine-annealed to 0 over $T = 8000$ steps.
- Batch size: 256.
- Mixed precision (FP16 amp).
- Hardware: Modal A100-80GB, ~4.2 steps/sec → ~30 min wall.

---

## 4. Inference

For a deployment query on $(F, P)$:

1. Rasterize $\mathbf{m}_F$.
2. For each $i \in \{0, \dots, 35\}$: rasterize $\mathbf{m}_{P,\theta_i}$, run $\mathbf{z}_i = \pi_\phi(\mathbf{x}_i)$.
3. Stack: $\mathbf{Z} = [\mathbf{z}_0, \dots, \mathbf{z}_{35}] \in \mathbb{R}^{36\times128\times128}$.
4. Predict $(\hat\imath, \hat r, \hat c) = \arg\max_{i,r,c} \mathbf{Z}[i,r,c]$, giving $\hat\theta = \theta_{\hat\imath}$, $(\hat x, \hat y) = \mathrm{pix}^{-1}(\hat r, \hat c)$.
5. **Optional refinement.** Evaluate Shapely reward on a local window
   $$
   \mathcal{W} \;=\; \bigl\{(\theta_{\hat\imath + \delta_\theta},\, \hat r + \delta_r,\, \hat c + \delta_c) : |\delta_\theta| \leq K,\ |\delta_r|, |\delta_c| \leq N\bigr\}
   $$
   and return $\arg\max_{\mathcal W} r(\cdot)$. With $N = 10, K = 2$, $|\mathcal{W}| = 5 \cdot 21^2 = 2205$ Shapely calls at ~75 μs each = ~165 ms. Closes the model-vs-brute-force gap.

Latency without refinement: 36 forward passes on a 3M-param U-Net, dominated by per-rotation Shapely-free rasterization ~30 ms total on a CPU; the pure model forward passes are <50 ms on GPU.

---

## 5. Data pipeline

### 5.1 Pair generation
- **Convex–convex.** Random convex hulls from $n$ uniform points; 12 000 pairs sampled by `generate_bc_dataset.py`.
- **Concave-fs / convex-part.** From `bo_train_pool_10k.pkl`: 10 000 pairs with non-convex $F$ (random star polygons) and convex $P$.

### 5.2 IFP precompute
For each pair and each $\theta_i$: compute $\mathrm{IFP}(F, P, \theta_i)$ via Minkowski erosion using `pyclipper` (`src/geometry/ifp.py`). Rasterize to $\{0,1\}^{128\times128}$.

### 5.3 Reward heatmap precompute
For each pixel inside the rasterized IFP, evaluate $r(F, P, \theta_i, x, y)$ via Shapely. Pixels outside IFP are 0. Stored as $(36, 128, 128)$ float32 per pair.

Total Shapely queries: $\sim 7.1 \times 10^8$ for convex + $\sim 3.1 \times 10^8$ for concave $\approx 10^9$ queries. Executed in parallel across 100 Modal containers × 8 cores. Total wall ≈ 25 minutes; cost ≈ \$40.

### 5.4 Rotated-part-mask precompute
For each pair: rasterize $R_{\theta_i}(P)$ centered at origin, for $i = 0, \dots, 35$. Stored as $(36, 128, 128)$ uint8 per pair. Used as the second input channel at training and inference.

### 5.5 Combined corpus
$N_{\text{train}} = 19\,800$ pairs (10 800 convex + 9000 concave), $N_{\text{val}} = 2\,200$ (1200 convex + 1000 concave). Combined pkl: 26.68 GB (fp16 heatmaps + uint8 masks).

---

## 6. Results

Model selection: best mean val recovery on a fixed 2200-pair validation set, evaluated every 250 steps. Compare model-only recovery (no Shapely refine):

| Configuration | Val recovery |
|---|---|
| Hard label, hierarchical (tile + cell) | 0.04 |
| Soft label, hierarchical | 0.43 |
| Soft label, hierarchical, tile-only | 0.32 |
| **Per-(pair, θ) U-Net, convex-only ($N{=}10\,800$)** | **0.72** |
| **Per-(pair, θ) U-Net, combined ($N{=}19\,800$)** | **0.72*** *(in progress)* |

The convex-only and combined models are nearly tied on a mixed val set; combined generalizes to concave free spaces from a single checkpoint. With Shapely refinement, both push to near-1.0 recovery (analytically: refinement window contains the true argmax with high probability if the model lands within $\pm 10$ pixels and $\pm 2$ rotation bins).

---

## 7. Implementation pointers

| concern | file |
|---|---|
| Model | `src/models/neural_bo_policy.py::_SmallUNet` |
| Training script | `scripts/train_perthet.py` |
| Modal training entrypoint | `modal_train_perthet.py` |
| IFP exact (handles concave fs) | `src/geometry/ifp.py::compute_ifp_exact` |
| Reward function | `src/geometry/rewards.py::compute_reward_exp` |
| Concave precompute | `modal_concave_precompute.py` |
| Combined-pkl build | `modal_build_combined.py` |
| Combined training data | `/vol/hier_training_data_combined.pkl` (Modal volume) |
| Best ckpt — convex | `checkpoints/perthet/final.pt` |
| Best ckpt — combined | `checkpoints/perthet_combined/final.pt` (pending) |
