"""Threshold at which each wetland polygon connects to the stream network.

Uses an existing threshold-method raster (e.g. from huc12_threshold.py); no
new flood is run. A polygon is treated as wet and freely traversable, and
everything outside it (other polygons included) is judged on the pixel
results:

- Polygons that intersect a flowline (vector test) get 1, and the reach with
  the longest length inside the polygon.
- Others get the highest ``route_threshold`` among the pixels whose centre
  lies inside the polygon. ``route_threshold`` leaves the pixel's own WIP out,
  so this is the best route leaving the polygon from any of its pixels; the
  maximum always falls on a margin pixel. Polygons containing no pixel centre
  fall back to the pixels they touch.

Multipart polygons are scored as one unit. Output columns:

    route_threshold  highest cutoff at which the polygon connects
                     (-1: no allowed route, NaN: no valid pixels)
    reach_id         reach the best route reaches
    source           flowline | centre | touched | none
    best_x, best_y   centre of the pixel the best route leaves from

Example (R2 raster, credentials from R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY):

    python scripts/polygon_threshold.py --polygons wetland_poly.gpkg \\
        --reaches reaches_100.parquet \\
        --connectivity /vsis3/skykomish/outputs/skykomish_17110009_threshold_downstream_wipgw.tif \\
        --s3-endpoint <account>.r2.cloudflarestorage.com --out polygons.gpkg
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from huc12_threshold import configure_r2, grid_window  # noqa: E402

from wetconn.threshold import NEVER  # noqa: E402


def flowline_hits(polys, reaches, id_field):
    """Reach with the longest length inside each polygon that a flowline crosses."""
    pairs = gpd.sjoin(
        gpd.GeoDataFrame(geometry=polys.geometry), gpd.GeoDataFrame({id_field: reaches[id_field].to_numpy()}, geometry=reaches.geometry.values),
        predicate="intersects",
    )
    seg = reaches.geometry.values[pairs["index_right"].to_numpy()]
    pairs["len"] = seg.intersection(polys.geometry.values[pairs.index.to_numpy()]).length
    best = pairs.sort_values("len", ascending=False).groupby(level=0).head(1)
    return best[id_field]


def score_polygon(src, geom):
    # Pad by a pixel: grid_window rounds offset and length separately and can
    # drop the last row or column.
    x0, y0, x1, y1 = geom.bounds
    px = abs(src.transform.a)
    win = grid_window((x0 - px, y0 - px, x1 + px, y1 + px), src.transform, src.shape)
    if win.width <= 0 or win.height <= 0:
        return np.nan, -1, "none", np.nan, np.nan
    route = src.read(2, window=win)
    rid = src.read(3, window=win)
    tf = src.window_transform(win)
    for source, touched in (("centre", False), ("touched", True)):
        inside = rasterize([geom], out_shape=route.shape, transform=tf, all_touched=touched, dtype="uint8") == 1
        vals = np.where(inside & np.isfinite(route), route, np.nan)
        if np.isfinite(vals).any():
            break
    else:
        return np.nan, -1, "none", np.nan, np.nan
    r, c = np.unravel_index(np.nanargmax(vals), vals.shape)
    x, y = tf * (c + 0.5, r + 0.5)
    best = float(vals[r, c])
    reach = int(rid[r, c]) if best != NEVER and np.isfinite(rid[r, c]) else -1
    return best, reach, source, x, y


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--polygons", required=True)
    p.add_argument("--layer", default=None)
    p.add_argument("--reaches", required=True, help="Flowlines (.parquet or any vector file).")
    p.add_argument("--id-field", default="reach_id")
    p.add_argument("--connectivity", required=True, help="Threshold raster (threshold, route_threshold, reach_id).")
    p.add_argument("--out", required=True, help="Output GeoPackage.")
    p.add_argument("--s3-endpoint", default=None, help="R2 endpoint host for /vsis3/ paths.")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    if a.s3_endpoint:
        configure_r2(a.s3_endpoint)

    t0 = time.time()
    polys = gpd.read_file(a.polygons, layer=a.layer)
    read = gpd.read_parquet if a.reaches.endswith(".parquet") else gpd.read_file
    reaches = read(a.reaches)
    reaches = reaches.set_geometry(reaches.geometry.name).to_crs(polys.crs)

    with rasterio.open(a.connectivity) as src:
        if src.descriptions[:3] != ("threshold", "route_threshold", "reach_id"):
            raise ValueError(f"unexpected bands: {src.descriptions}")
        if src.crs != polys.crs:
            polys = polys.to_crs(src.crs)
            reaches = reaches.to_crs(src.crs)

    n = len(polys)
    route = np.full(n, np.nan)
    reach = np.full(n, -1, np.int64)
    source = np.full(n, "none", object)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)

    hits = flowline_hits(polys, reaches, a.id_field)
    pos = polys.index.get_indexer(hits.index)
    route[pos] = 1.0
    reach[pos] = hits.to_numpy()
    source[pos] = "flowline"
    rest = np.flatnonzero(source == "none")
    print(f"{len(pos)} of {n} polygons intersect a flowline; scoring {len(rest)} from pixels")

    local = threading.local()

    def work(i):
        if not hasattr(local, "src"):
            local.src = rasterio.open(a.connectivity)
        return i, score_polygon(local.src, polys.geometry.iloc[i])

    with ThreadPoolExecutor(a.workers) as ex:
        for i, (rt, rid, s, x, y) in ex.map(work, rest):
            route[i], reach[i], source[i], bx[i], by[i] = rt, rid, s, x, y

    out = polys.copy()
    out["route_threshold"] = route
    out["reach_id"] = reach
    out["source"] = source
    out["best_x"] = bx
    out["best_y"] = by
    out.to_file(a.out, driver="GPKG", layer="polygon_threshold")

    area = out.area.to_numpy() / 1e4
    print(f"sources: {out['source'].value_counts().to_dict()}")
    print(f"no allowed route: {(route == NEVER).sum()}, no valid pixels: {np.isnan(route).sum()}")
    for t in (0.75, 0.5, 0.25):
        m = route >= t
        print(f"connected at {t:.2f}: {m.sum()} polygons ({area[m].sum():,.0f} of {area.sum():,.0f} ha)")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
