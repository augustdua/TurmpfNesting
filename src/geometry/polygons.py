import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate, scale
from shapely import ops


def random_convex_polygon(n_vertices=8, radius=100.0):
    """Generate a random convex polygon using convex hull of random points."""
    from shapely.geometry import MultiPoint

    # Generate random points in a disk and take their convex hull
    # Use more points than needed so the hull has enough vertices
    n_candidates = max(n_vertices * 3, 20)
    angles = np.random.uniform(0, 2 * np.pi, n_candidates)
    r = radius * np.sqrt(np.random.uniform(0.1, 1.0, n_candidates))
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    hull = MultiPoint(list(zip(x, y))).convex_hull
    return hull


def random_star_polygon(n_points=5, outer_r=100.0, inner_r=40.0):
    """Generate a star-shaped non-convex polygon."""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    r = np.where(np.arange(2 * n_points) % 2 == 0, outer_r, inner_r)
    r = r * (0.8 + 0.2 * np.random.uniform(size=2 * n_points))
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    return Polygon(zip(x, y))


def random_irregular_polygon(n_vertices=10, radius=100.0):
    """Generate random irregular polygon by perturbing a convex one."""
    for _ in range(10):
        poly = random_convex_polygon(n_vertices, radius)
        coords = np.array(poly.exterior.coords[:-1])
        noise = np.random.normal(0, radius * 0.15, coords.shape)
        noisy = coords + noise
        poly = Polygon(noisy)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix self-intersections
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)
        if poly.is_valid and not poly.is_empty:
            return poly
    # fallback to convex if all retries fail
    return random_convex_polygon(n_vertices, radius)


def simplify_polygon(poly, tolerance=1.0):
    """Douglas-Peucker simplification via Shapely."""
    return poly.simplify(tolerance, preserve_topology=True)


def sample_boundary_points(poly, n_points=128):
    """Sample n_points uniformly by arc length along the boundary."""
    boundary = poly.exterior
    total_length = boundary.length
    distances = np.linspace(0, total_length, n_points, endpoint=False)
    points = np.array([boundary.interpolate(d).coords[0] for d in distances])
    return points


def normalize_polygon(poly):
    """Center at origin, scale to fit in [-1, 1] x [-1, 1]."""
    centroid = poly.centroid
    centered = translate(poly, -centroid.x, -centroid.y)
    bounds = centered.bounds  # (minx, miny, maxx, maxy)
    max_extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if max_extent > 0:
        centered = scale(centered, 2.0 / max_extent, 2.0 / max_extent, origin=(0, 0))
    return centered


def normalize_polygon_pair(free_space, part):
    """
    Normalize free_space to [-1, 1] and apply the SAME transform to part,
    preserving their relative sizes.
    """
    # Center and scale free_space
    fs_centroid = free_space.centroid
    fs_centered = translate(free_space, -fs_centroid.x, -fs_centroid.y)
    bounds = fs_centered.bounds
    max_extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    scale_factor = 2.0 / max_extent if max_extent > 0 else 1.0
    fs_norm = scale(fs_centered, scale_factor, scale_factor, origin=(0, 0))

    # Apply same transform to part: translate by free_space's centroid, then same scale
    part_centered = translate(part, -fs_centroid.x, -fs_centroid.y)
    part_norm = scale(part_centered, scale_factor, scale_factor, origin=(0, 0))

    return fs_norm, part_norm
