import numpy as np


def compute_boundary_features(points):
    """
    Compute per-point features for boundary points.

    Args:
        points: (N, 2) array of boundary points in order

    Returns:
        features: (N, 6) array
            [relative_x, relative_y, curvature, tangent_x, tangent_y, dist_to_centroid]
    """
    N = len(points)
    centroid = points.mean(axis=0)

    # Position relative to centroid
    rel = points - centroid

    # Distance to centroid
    dist = np.linalg.norm(rel, axis=1, keepdims=True)

    # Tangent direction (central finite difference, wrapping around)
    prev = np.roll(points, 1, axis=0)
    next_ = np.roll(points, -1, axis=0)
    tangent = next_ - prev
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.clip(tangent_norm, 1e-8, None)
    tangent = tangent / tangent_norm

    # Curvature (signed, via cross product of consecutive edge vectors)
    e1 = points - prev
    e2 = next_ - points
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    dot = (e1 * e2).sum(axis=1)
    curvature = np.arctan2(cross, dot).reshape(-1, 1)

    features = np.hstack([rel, curvature, tangent, dist])
    return features  # (N, 6)


def polygon_to_features(poly, n_points=128):
    """Full pipeline: polygon -> sampled boundary -> feature array."""
    from .polygons import sample_boundary_points, normalize_polygon

    poly = normalize_polygon(poly)
    points = sample_boundary_points(poly, n_points)
    features = compute_boundary_features(points)
    return features  # (N, 6)
