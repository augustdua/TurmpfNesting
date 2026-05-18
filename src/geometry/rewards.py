import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


def place_part(part, x, y, theta):
    """Apply rotation then translation to a part."""
    placed = rotate(part, theta, origin='centroid', use_radians=False)
    cx, cy = placed.centroid.coords[0]
    placed = translate(placed, x - cx, y - cy)
    return placed


def _sample_boundary_distances(placed, free_space, n_samples=64):
    """
    Sample points along the placed part's boundary and measure
    each point's distance to the free space boundary.

    Uses vectorized nearest-point computation for speed.

    Returns:
        distances: array of shape (n_samples,) — distance per sample point
    """
    from shapely import prepare

    boundary = placed.boundary
    total_len = boundary.length

    # Pre-sample all points at once
    fracs = np.linspace(0, total_len, n_samples, endpoint=False)
    pts = [boundary.interpolate(d) for d in fracs]

    # Prepare free space for fast repeated queries
    fs_boundary = free_space.boundary
    prepare(fs_boundary)

    distances = np.array([fs_boundary.distance(p) for p in pts])
    return distances


def compute_shaped_reward(free_space, part, x, y, theta):
    """
    Shaped reward that pushes harder toward the IFP boundary (where adherence lives).
    Vs compute_reward():
      - Base reward reduced (0.3 -> 0.1) so interior placements look less attractive
      - Adherence weight boosted (0.5 -> 0.5)
      - Proximity weight boosted (0.2 -> 0.4) — continuous signal toward walls
      - Invalid penalty softened (-overlap -> -0.5 * overlap) so boundary overshoots
        aren't catastrophic; policy can venture past boundary and recover.
    Total valid reward range: [0.1, 1.0]
    Invalid range: [-0.5, 0]
    """
    placed = place_part(part, x, y, theta)
    if not placed.is_valid:
        return -0.5

    if free_space.contains(placed):
        bounds = free_space.bounds
        max_dist = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2.0
        epsilon = max_dist * 0.02

        try:
            dists = _sample_boundary_distances(placed, free_space)
        except Exception:
            dists = np.array([max_dist])

        reward = 0.1                                          # base (low — don't settle)
        adherence = np.mean(dists < epsilon)
        reward += 0.5 * adherence                             # full adherence weight
        closeness = np.mean(1.0 - np.clip(dists / max_dist, 0, 1))
        reward += 0.4 * closeness                             # boosted proximity (2x)
        return float(min(reward, 1.0))
    else:
        try:
            outside = placed.difference(free_space)
            overlap_ratio = outside.area / placed.area
        except Exception:
            overlap_ratio = 1.0
        return -0.5 * float(overlap_ratio)                    # softer invalid penalty


def compute_reward_exp(free_space, part, x, y, theta, k=3.0):
    """
    Exponentially-sharpened reward. Same structure as compute_reward, but
    adherence and proximity are passed through (exp(k*x) - 1) so the gap
    between interior (low adherence/proximity) and boundary (higher values)
    grows super-linearly. Larger k = sharper gap.

    With k=3: max valid reward ~= 0.3 + 0.5*(e^3-1) + 0.2*(e^3-1) ~= 13.6
    Invalid still in [-1, 0]. Scale changes — tune policy LR accordingly.
    """
    placed = place_part(part, x, y, theta)
    if not placed.is_valid:
        return -1.0

    if free_space.contains(placed):
        bounds = free_space.bounds
        max_dist = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2.0
        epsilon = max_dist * 0.02
        try:
            dists = _sample_boundary_distances(placed, free_space)
        except Exception:
            dists = np.array([max_dist])

        adherence = float(np.mean(dists < epsilon))
        closeness = float(np.mean(1.0 - np.clip(dists / max_dist, 0, 1)))
        reward = 0.3
        reward += 0.5 * (np.exp(k * adherence) - 1.0)
        reward += 0.2 * (np.exp(k * closeness) - 1.0)
        return float(reward)
    else:
        try:
            outside = placed.difference(free_space)
            overlap_ratio = outside.area / placed.area
        except Exception:
            overlap_ratio = 1.0
        return -float(overlap_ratio)


def compute_reward(free_space, part, x, y, theta):
    """
    Compute placement reward.

    Returns:
        reward: float in [-1, 1]
            Negative if invalid (proportional to overlap outside)
            Positive if valid:
                base reward (0.3) for being fully inside
              + boundary adherence bonus (up to 0.5) — fraction of part
                boundary points within epsilon of free space boundary
              + proximity-to-boundary bonus (up to 0.2) — mean closeness
                of all boundary points to the free space boundary
    """
    placed = place_part(part, x, y, theta)

    if not placed.is_valid:
        return -1.0

    if free_space.contains(placed):
        # Measure distances from part boundary samples to free space boundary
        bounds = free_space.bounds
        max_dist = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2.0
        epsilon = max_dist * 0.02  # ~2% of free space extent

        try:
            dists = _sample_boundary_distances(placed, free_space)
        except Exception:
            dists = np.array([max_dist])

        # Base reward for valid placement
        reward = 0.3

        # Adherence: fraction of boundary points within epsilon of free space boundary
        adherence = np.mean(dists < epsilon)
        reward += 0.5 * adherence

        # Proximity: mean closeness across all boundary points
        closeness = np.mean(1.0 - np.clip(dists / max_dist, 0, 1))
        reward += 0.2 * closeness

        return float(min(reward, 1.0))
    else:
        # Invalid placement — shaped negative reward
        try:
            outside = placed.difference(free_space)
            overlap_ratio = outside.area / placed.area
        except Exception:
            overlap_ratio = 1.0
        return -float(overlap_ratio)
