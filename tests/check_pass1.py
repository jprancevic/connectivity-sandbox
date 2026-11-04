"""Independent check of pass 1: rebuilds the pass-1 labels with a separate, plain implementation and
compares them cell by cell with the tool's <prefix>_pass1.tif.

    check_pass1(DEM, WETLANDS, OUT_PREFIX, streams=None, all_touched=False) -> bool

OUT_PREFIX is the output path prefix, e.g. out/test  ->  out/test_pass1.tif

The independent version differs from the tool on purpose:
  * D8 is vectorised numpy, not a numba loop
  * flats are routed with a Python breadth-first search, closed flats are found with scipy.ndimage.label
  * labels are assigned with the sorted-elevation walk: rank every cell by (elevation, distance across
    its flat), walk from lowest to highest, and copy the label of each cell's downslope neighbour,
    which must already be labelled
It is slow (Python loops); keep test DEMs to a few million cells.
"""
from collections import deque

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy import ndimage


def check_pass1(dem, wetlands, out_prefix, streams=None, id_field="Wetlands ID", all_touched=False):
    """Returns True if the tool's pass-1 labels match the independent version."""
    NODATA, UNSET = -2147483648, -2147483647

    src = rasterio.open(dem)
    z = src.read(1, masked=True).astype(np.float64).filled(np.nan)
    nr, nc = z.shape
    dx, dy = src.res
    valid = ~np.isnan(z)

    # stop cells: wetland ID + 1 (smaller polygons burned last), then streams = 0, then nodata
    wl = gpd.read_file(wetlands).to_crs(src.crs)
    wl = wl.loc[wl.area.sort_values(ascending=False).index]
    lab0 = features.rasterize(zip(wl.geometry, wl[id_field].astype(np.int64) + 1), out_shape=z.shape,
                              transform=src.transform, fill=UNSET, dtype="int32", all_touched=all_touched)
    if streams:
        st = features.rasterize(((g, 1) for g in gpd.read_file(streams).to_crs(src.crs).geometry),
                                out_shape=z.shape, transform=src.transform, all_touched=True).astype(bool)
        lab0[st] = 0
    lab0[~valid] = NODATA
    stop = valid & (lab0 != UNSET)

    # vectorised D8 (neighbour order NE, E, SE, S, SW, W, NW, N; first maximum wins)
    DR = [-1, 0, 1, 1, 1, 0, -1, -1]
    DC = [1, 1, 1, 0, -1, -1, -1, 0]
    zp = np.pad(z, 1, constant_values=np.nan)
    slopes = np.full((8, nr, nc), -np.inf)
    edge = np.zeros((nr, nc), bool)
    for k in range(8):
        zn = zp[1 + DR[k]:1 + DR[k] + nr, 1 + DC[k]:1 + DC[k] + nc]
        s = (z - zn) / np.hypot(DR[k] * dy, DC[k] * dx)
        slopes[k] = np.where(np.isnan(s), -np.inf, s)
        edge |= np.isnan(zn)                                   # off-grid or nodata neighbour
    best = slopes.argmax(0)
    has_lower = slopes.max(0) > 0
    free = valid & ~stop
    rr, cc = np.indices((nr, nc))
    recv = np.full((nr, nc), -1, np.int64)
    m = free & has_lower
    recv[m] = (rr[m] + np.take(DR, best[m])) * nc + (cc[m] + np.take(DC, best[m]))
    out = free & ~has_lower & edge
    flat = free & ~has_lower & ~edge

    # flats: breadth-first from same-elevation exits
    dist = np.zeros((nr, nc), np.int64)
    exitable = stop | (free & has_lower) | out
    done = np.zeros((nr, nc), bool)
    q = deque()
    for r, c in np.argwhere(flat):
        for k in range(8):
            r2, c2 = r + DR[k], c + DC[k]
            if 0 <= r2 < nr and 0 <= c2 < nc and exitable[r2, c2] and z[r2, c2] == z[r, c]:
                recv[r, c], dist[r, c], done[r, c] = r2 * nc + c2, 1, True
                q.append((r, c))
                break
    while q:
        r, c = q.popleft()
        for k in range(8):
            r2, c2 = r + DR[k], c + DC[k]
            if 0 <= r2 < nr and 0 <= c2 < nc and flat[r2, c2] and not done[r2, c2] and z[r2, c2] == z[r, c]:
                recv[r2, c2], dist[r2, c2], done[r2, c2] = r * nc + c, dist[r, c] + 1, True
                q.append((r2, c2))
    pit_floor = flat & ~done
    pit_lab, n_pits = ndimage.label(pit_floor, structure=np.ones((3, 3)))

    # the sorted-elevation walk
    L = lab0.ravel().astype(np.int64).copy()
    L[out.ravel()] = 0
    L[pit_floor.ravel()] = -pit_lab.ravel()[pit_floor.ravel()]
    vidx = np.flatnonzero(valid.ravel())
    order = vidx[np.lexsort((dist.ravel()[vidx], z.ravel()[vidx]))]
    rv = recv.ravel()
    for i in order:
        if L[i] == UNSET:
            assert L[rv[i]] != UNSET, "downslope neighbour not labelled yet: ordering broken"
            L[i] = L[rv[i]]
    L = L.reshape(nr, nc)

    # compare (pit IDs compared as a partition, so numbering order doesn't matter)
    T = rasterio.open(f"{out_prefix}_pass1.tif").read(1).astype(np.int64)
    pit_e, pit_t = L < 0, (T < 0) & (T != NODATA)
    non_pit_bad = int(((L != T) & ~(pit_e & pit_t) & valid).sum() + (T[~valid] != NODATA).sum())
    pairs = set(zip(L[pit_e & pit_t].tolist(), T[pit_e & pit_t].tolist()))
    bijective = len(pairs) == n_pits == len({t for _, t in pairs})
    print(f"pits: independent {n_pits}, tool {-T[pit_t].min() if pit_t.any() else 0}; pit catchments match as a "
          f"partition: {bijective}")
    print(f"cells that differ: {non_pit_bad:,} of {L.size:,}")
    ok = non_pit_bad == 0 and bijective
    return ok
