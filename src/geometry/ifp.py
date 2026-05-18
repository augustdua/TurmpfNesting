"""
Exact Inner-Fit Polygon (IFP) for arbitrary fs + convex part.

Uses pyclipper.MinkowskiSum on each fs edge with the reflected part kernel,
unions the swept regions, and subtracts from fs.

Returns a Shapely (Multi)Polygon of valid centroid placements.
"""
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from shapely.affinity import rotate
from shapely.ops import unary_union

_SCALE = 10**6


def _to_int(path):
    return [(int(round(x * _SCALE)), int(round(y * _SCALE))) for x, y in path]


def _from_int(paths):
    polys = []
    for path in paths:
        coords = [(x / _SCALE, y / _SCALE) for x, y in path]
        if len(coords) >= 3:
            try:
                p = Polygon(coords)
                if p.is_valid and p.area > 0:
                    polys.append(p)
            except Exception:
                pass
    if not polys:
        return Polygon()
    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)


def compute_ifp_exact(fs, part, theta=0.0):
    """
    Exact IFP for any fs (convex or concave) and convex part, at rotation theta.

    Args:
        fs:    Shapely Polygon (free space)
        part:  Shapely Polygon (convex part)
        theta: rotation in degrees applied to part around its centroid

    Returns:
        Shapely Polygon or MultiPolygon of valid centroid positions.
    """
    rotated = rotate(part, theta, origin='centroid', use_radians=False)
    cx, cy = rotated.centroid.coords[0]
    K = [(-(x - cx), -(y - cy))
         for (x, y) in rotated.exterior.coords[:-1]]
    K_int = _to_int(K)

    fs_coords = list(fs.exterior.coords)
    swept_paths = []
    for i in range(len(fs_coords) - 1):
        edge_int = [
            (int(round(fs_coords[i][0] * _SCALE)),
             int(round(fs_coords[i][1] * _SCALE))),
            (int(round(fs_coords[i + 1][0] * _SCALE)),
             int(round(fs_coords[i + 1][1] * _SCALE))),
        ]
        result = pyclipper.MinkowskiSum(K_int, edge_int, False)
        swept_paths.extend(result)

    if not swept_paths:
        return fs

    pc = pyclipper.Pyclipper()
    for path in swept_paths:
        pc.AddPath(path, pyclipper.PT_SUBJECT, True)
    forbidden_int = pc.Execute(pyclipper.CT_UNION,
                                pyclipper.PFT_NONZERO,
                                pyclipper.PFT_NONZERO)

    fs_int = _to_int(fs_coords[:-1])
    pc2 = pyclipper.Pyclipper()
    pc2.AddPath(fs_int, pyclipper.PT_SUBJECT, True)
    if forbidden_int:
        pc2.AddPaths(forbidden_int, pyclipper.PT_CLIP, True)
    ifp_int = pc2.Execute(pyclipper.CT_DIFFERENCE,
                           pyclipper.PFT_NONZERO,
                           pyclipper.PFT_NONZERO)

    return _from_int(ifp_int)
