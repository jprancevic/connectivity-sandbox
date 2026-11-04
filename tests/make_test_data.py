"""Build a synthetic test landscape for the wetland drainage tools.

    make_test_data(OUT_DIR, rows=1500, cols=1200)

Writes to OUT_DIR:
  dem.tif         3 m float32 DEM (EPSG:26910) with a ragged nodata border, many natural pits, and
                  three planted flat cases (see below)
  dem_rounded.tif the same DEM rounded to 0.05 m, like lidar stored at limited precision: many exact
                  slope ties and small flats, which exercise the tie-break rules
  wetlands.gpkg   ~125 polygons, field "Wetlands ID" = integers starting at 0, including awkward cases
  streams.shp     a few random flowlines
  cases.npz       masks of the planted flat cases, for spot checks

Planted flat cases:
  lake    a flat lake at its lowest rim elevation: has an outlet, must not become a pit
  bowl    a closed flat depression with no wetland: must become exactly one pit
  around  a closed flat bowl whose only way out is a wetland sitting in it: must drain into that wetland

Awkward wetland cases: overlapping polygons with different IDs, a sub-pixel polygon, one off the DEM,
and two IDs shared by two polygons each.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, gaussian_filter
from shapely import affinity
from shapely.geometry import LineString, Point


def make_test_data(out_dir, rows=1500, cols=1200):
    """Writes the test landscape to out_dir (see the module docstring)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nr, nc = rows, cols
    rng = np.random.default_rng(42)
    cs, x0, y0 = 3.0, 600000.0, 5300000.0
    tr = from_origin(x0, y0, cs, cs)

    # --- terrain: regional slope + ridges + multi-scale noise (noise makes plenty of natural pits) ---
    r = np.arange(nr)[:, None]
    c = np.arange(nc)[None, :]
    z = 0.02 * c * cs + 0.01 * r * cs + 25 * np.abs(np.sin(r / nr * 3 * np.pi))
    for sigma, amp in [(120, 40), (40, 8), (10, 1.5), (2, 0.3)]:
        n = gaussian_filter(rng.standard_normal((nr, nc)), sigma)
        z = z + amp * n / n.std()
    z = z.astype("float32")
    edge = np.minimum.reduce([r + 0 * c, nr - 1 - r + 0 * c, c + 0 * r, nc - 1 - c + 0 * r]).astype(float)
    nd = edge < 40 + 600 * gaussian_filter(rng.standard_normal((nr, nc)), 60)
    z[nd] = -9999

    # --- wetlands: random ellipses spread over elevations ---
    valid = np.argwhere(~nd)
    pick = valid[rng.choice(len(valid), 4000, replace=False)]
    pick = pick[np.argsort(z[pick[:, 0], pick[:, 1]])[:: 10]][:120]
    geoms = []
    for i, j in pick:
        g = Point(x0 + (j + 0.5) * cs, y0 - (i + 0.5) * cs).buffer(rng.uniform(6, 60))
        geoms.append(affinity.rotate(affinity.scale(g, rng.uniform(0.4, 1.0), 1.0), rng.uniform(0, 180)))
    ids = list(range(len(geoms)))
    geoms.append(Point(geoms[5].centroid.x + 5, geoms[5].centroid.y).buffer(8)); ids.append(len(ids))  # overlap
    geoms.append(Point(x0 + 900, y0 - 900).buffer(0.8)); ids.append(len(ids))                          # sub-pixel
    geoms.append(Point(x0 - 5000, y0 + 5000).buffer(30)); ids.append(len(ids))                         # off the DEM
    geoms.append(geoms[10].centroid.buffer(4)); ids.append(10)                                         # shared ID
    d2 = geoms[20].centroid
    geoms.append(Point(d2.x + 300, d2.y + 300).buffer(20)); ids.append(20)                             # shared ID
    wl = gpd.GeoDataFrame({"Wetlands ID": ids}, geometry=geoms, crs="EPSG:26910")
    wl.to_file(out / "wetlands.gpkg", driver="GPKG")

    # --- planted flats ---
    rr, cc = np.indices(z.shape)
    wet = features.rasterize(((g, 1) for g in wl.geometry), out_shape=z.shape, transform=tr).astype(bool)

    def disk(r0, c0, rad):
        return ((rr - r0) ** 2 + (cc - c0) ** 2 <= rad ** 2) & ~nd

    def ring(m):
        return binary_dilation(m, np.ones((3, 3), bool)) & ~m & ~nd

    def free_spot(rad, avoid):
        """First grid spot whose disk touches no wetland, no nodata and nothing in `avoid`."""
        for r0 in range(rad + 50, nr - rad - 50, 25):
            for c0 in range(rad + 50, nc - rad - 50, 25):
                m = (rr - r0) ** 2 + (cc - c0) ** 2 <= (rad + 3) ** 2
                if not (wet[m].any() or nd[m].any() or avoid[m].any()):
                    return r0, c0
        raise RuntimeError("no free spot for a planted case; use a bigger grid")

    lake = disk(*free_spot(40, np.zeros_like(nd)), 40)
    z[lake] = z[ring(lake)].min()                        # flat at its lowest rim -> has an outlet
    bowl = disk(*free_spot(15, binary_dilation(lake, iterations=10)), 15)
    z[bowl] = z[ring(bowl)].min() - 5.0                  # closed flat, no wetland -> one pit
    w = features.rasterize([(wl.geometry.iloc[49], 1)], out_shape=z.shape, transform=tr).astype(bool) & ~nd
    around = binary_dilation(w, np.ones((3, 3), bool), iterations=6) & ~w & ~nd
    z[around | w] = z[ring(around | w)].min() - 1.0      # closed flat bowl, only exit is the wetland
    np.savez(out / "cases.npz", lake=lake, bowl=bowl, around=around, around_wetland_id=49)

    with rasterio.open(out / "dem.tif", "w", driver="GTiff", height=nr, width=nc, count=1, dtype="float32",
                       nodata=-9999, crs="EPSG:26910", transform=tr, tiled=True, compress="deflate") as d:
        d.write(z, 1)
    zr = np.where(nd, -9999, np.round(z / 0.05) * 0.05).astype("float32")
    with rasterio.open(out / "dem_rounded.tif", "w", driver="GTiff", height=nr, width=nc, count=1, dtype="float32",
                       nodata=-9999, crs="EPSG:26910", transform=tr, tiled=True, compress="deflate") as d:
        d.write(zr, 1)

    # --- streams ---
    srng = np.random.default_rng(1)
    lines = [LineString(np.cumsum(srng.normal(0, 60, (40, 2)), 0)
                        + [x0 + srng.uniform(600, 3000), y0 - srng.uniform(600, 4000)]) for _ in range(6)]
    gpd.GeoDataFrame({"id": range(6)}, geometry=lines, crs="EPSG:26910").to_file(out / "streams.shp")
    print(f"wrote {out}: DEM {nc} x {nr}, {len(wl)} wetland polygons ({wl['Wetlands ID'].nunique()} IDs), 6 streams")
