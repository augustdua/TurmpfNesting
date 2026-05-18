"""
Rasterize the precomputed IFP union polygons to 128x128 masks
and add as 'ifp_union_mask' (uint8) to each record.
"""
import argparse
import pickle
import numpy as np
from PIL import Image, ImageDraw
from shapely.wkt import loads as wkt_loads


def rasterize_polygon(poly, resolution=128):
    """Rasterize a (possibly multi-part) shapely polygon in [-1, 1] coords to a binary mask."""
    img = Image.new('L', (resolution, resolution), 0)
    draw = ImageDraw.Draw(img)
    # Handle both Polygon and MultiPolygon
    if poly.geom_type == 'MultiPolygon':
        geoms = poly.geoms
    else:
        geoms = [poly]
    for g in geoms:
        coords = list(g.exterior.coords)
        pixels = []
        for x, y in coords:
            px = int(np.clip((x + 1) * 0.5 * (resolution - 1), 0, resolution - 1))
            py = int(np.clip((1 - (y + 1) * 0.5) * (resolution - 1), 0, resolution - 1))
            pixels.append((px, py))
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=1)
    return np.array(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-data', required=True)
    ap.add_argument('--out-data', required=True)
    ap.add_argument('--resolution', type=int, default=128)
    args = ap.parse_args()

    with open(args.in_data, 'rb') as f:
        records = pickle.load(f)
    print(f"Rasterizing IFP union for {len(records)} pairs at {args.resolution}x{args.resolution}...")

    n_added = 0
    fracs = []
    for i, r in enumerate(records):
        if 'ifp_union_wkt' not in r:
            continue
        poly = wkt_loads(r['ifp_union_wkt'])
        mask = rasterize_polygon(poly, args.resolution)
        r['ifp_union_mask'] = mask
        n_added += 1
        fracs.append(mask.mean())
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(records)}  mean fill: {np.mean(fracs):.3f}")

    with open(args.out_data, 'wb') as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nAdded IFP union mask to {n_added} records")
    print(f"Fill fraction: mean={np.mean(fracs):.3f}  median={np.median(fracs):.3f}  "
          f"range=[{np.min(fracs):.3f}, {np.max(fracs):.3f}]")
    print(f"Saved -> {args.out_data}")


if __name__ == "__main__":
    main()
