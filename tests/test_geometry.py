import numpy as np
from shapely.geometry import Polygon

from src.geometry.polygons import *
from src.geometry.features import *
from src.geometry.rewards import *


def test_polygon_generation():
    for _ in range(100):
        p = random_convex_polygon()
        assert p.is_valid
        p = random_irregular_polygon()
        assert p.is_valid
    print("PASS: polygon generation")


def test_boundary_sampling():
    p = random_convex_polygon(8, 100)
    points = sample_boundary_points(p, 128)
    assert points.shape == (128, 2)
    print("PASS: boundary sampling")


def test_features():
    p = random_convex_polygon(8, 100)
    feat = polygon_to_features(p, 128)
    assert feat.shape == (128, 6)
    assert not np.any(np.isnan(feat))
    print("PASS: feature computation")


def test_reward_valid_center():
    """Part in center of large free space: valid, base reward + proximity."""
    fs = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    part = Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)])
    r = compute_reward(fs, part, 0.0, 0.0, 0.0)
    assert r >= 0.3, f"Expected base reward >= 0.3, got {r}"
    print(f"PASS: valid center placement reward = {r:.3f}")


def test_reward_valid_near_edge():
    """Part near edge should have higher reward than part in center."""
    fs = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    part = Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)])
    r_center = compute_reward(fs, part, 0.0, 0.0, 0.0)
    r_edge = compute_reward(fs, part, 1.5, 0.0, 0.0)  # near edge but still inside
    assert r_edge > r_center, f"Edge reward {r_edge:.3f} should beat center {r_center:.3f}"
    print(f"PASS: edge reward ({r_edge:.3f}) > center reward ({r_center:.3f})")


def test_reward_invalid():
    fs = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    part = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    r = compute_reward(fs, part, 2.0, 2.0, 0.0)
    assert r < 0, f"Expected invalid placement, got reward {r}"
    print(f"PASS: invalid placement reward = {r:.3f}")


def test_reward_shaped():
    """Reward should be more negative when more of the part is outside."""
    fs = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    part = Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])

    r_slight = compute_reward(fs, part, 0.8, 0.0, 0.0)
    r_far = compute_reward(fs, part, 3.0, 0.0, 0.0)

    assert r_far < r_slight, "Farther placement should have worse reward"
    print(f"PASS: shaped reward (slight={r_slight:.3f}, far={r_far:.3f})")


def test_reward_boundary_touch():
    """Part touching the boundary should get adherence bonus."""
    fs = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    # Small square part, place its centroid near the edge so it touches
    part = Polygon([(-0.3, -0.3), (0.3, -0.3), (0.3, 0.3), (-0.3, 0.3)])
    r_touch = compute_reward(fs, part, 1.7, 0.0, 0.0)  # right edge touching
    r_center = compute_reward(fs, part, 0.0, 0.0, 0.0)  # floating in center
    assert r_touch > r_center, f"Touching ({r_touch:.3f}) should beat center ({r_center:.3f})"
    print(f"PASS: boundary-touching reward ({r_touch:.3f}) > center ({r_center:.3f})")


if __name__ == "__main__":
    test_polygon_generation()
    test_boundary_sampling()
    test_features()
    test_reward_valid_center()
    test_reward_valid_near_edge()
    test_reward_invalid()
    test_reward_shaped()
    test_reward_boundary_touch()
    print("\nAll geometry tests passed.")
