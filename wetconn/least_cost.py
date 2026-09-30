"""Method 1: most-probable path to the stream network.

For each pixel we find the 8-connected path to a stream pixel that maximises
the product of wetland probabilities along the way, i.e. minimises the sum of
``-ln(wip)`` weighted by the length travelled through each pixel. Paths may
run in any direction, or optionally only downhill/level toward the stream.
"""

import numpy as np
from numba import njit

from .flowpath import fill_depressions
from .grid import OFFSETS, attenuation, check_wip, step_lengths


@njit(cache=True)
def _sift_up(heap, pos, key, i):
    node = heap[i]
    k = key[node]
    while i > 0:
        parent = (i - 1) >> 1
        pnode = heap[parent]
        if key[pnode] <= k:
            break
        heap[i] = pnode
        pos[pnode] = i
        i = parent
    heap[i] = node
    pos[node] = i


@njit(cache=True)
def _sift_down(heap, pos, key, i, size):
    node = heap[i]
    k = key[node]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and key[heap[child + 1]] < key[heap[child]]:
            child += 1
        cnode = heap[child]
        if key[cnode] >= k:
            break
        heap[i] = cnode
        pos[cnode] = i
        i = child
    heap[i] = node
    pos[node] = i


@njit(cache=True)
def _dijkstra(lam, is_target, label, z, max_rise, offsets, lengths):
    """Multi-source Dijkstra from the target pixels.

    Edge weight between neighbours a and b is ``0.5 * (lam[a] + lam[b]) * d``
    (trapezoid rule for the line integral of ``lam`` between pixel centres).
    ``lam`` must be 0 on targets; ``inf`` marks barriers.

    A step toward the stream is only allowed if it rises by at most
    ``max_rise * d``. With ``z`` all zeros every step is allowed.

    Returns the accumulated cost from each pixel centre to the nearest target
    centre, the length of that path, and the label of the target reached.
    """
    nr, nc = lam.shape
    n = nr * nc
    lamf = lam.ravel()
    tgt = is_target.ravel()
    labf = label.ravel()
    zf = z.ravel()
    cost = np.full(n, np.inf)
    plen = np.full(n, np.inf)
    lab = np.full(n, -1, np.int64)
    heap = np.empty(n, np.int64)
    pos = np.full(n, -1, np.int64)  # -1: never queued, -2: settled

    size = 0
    for i in range(n):
        if tgt[i]:
            cost[i] = 0.0
            plen[i] = 0.0
            lab[i] = labf[i]
            heap[size] = i
            pos[i] = size
            size += 1

    while size > 0:
        u = heap[0]
        pos[u] = -2
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            pos[heap[0]] = 0
            _sift_down(heap, pos, cost, 0, size)

        r = u // nc
        c = u - r * nc
        cu = cost[u]
        lu = lamf[u]
        for k in range(8):
            rr = r + offsets[k, 0]
            cc = c + offsets[k, 1]
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            v = rr * nc + cc
            if pos[v] == -2:
                continue
            lv = lamf[v]
            if lv == np.inf:
                continue
            # Travel runs v -> u (toward the stream): limit the climb.
            if zf[u] - zf[v] > max_rise * lengths[k]:
                continue
            nd = cu + 0.5 * (lu + lv) * lengths[k]
            nl = plen[u] + lengths[k]
            if nd == cost[v] and nl < plen[v]:
                # Equal cost: prefer the geometrically shorter path.
                plen[v] = nl
                lab[v] = lab[u]
            elif nd < cost[v]:
                cost[v] = nd
                plen[v] = nl
                lab[v] = lab[u]
                if pos[v] == -1:
                    heap[size] = v
                    pos[v] = size
                    size += 1
                    _sift_up(heap, pos, cost, size - 1)
                else:
                    _sift_up(heap, pos, cost, pos[v])

    return cost.reshape(nr, nc), plen.reshape(nr, nc), lab.reshape(nr, nc)


@njit(cache=True)
def _exclude_source(cost, plen, lab, lam, is_target, z, max_rise, offsets, lengths):
    """Drop each pixel's own half-step from its path cost.

    ``cost`` (from ``_dijkstra``) includes half of the first step, spent
    inside the source pixel. That pixel is accounted for by the at-site term
    ``wip``, so here we re-choose the first step without it:
    ``H[j] = min_m cost[m] + 0.5 * lam[m] * d(j, m)``.
    """
    nr, nc = cost.shape
    h = np.full((nr, nc), np.inf)
    hl = np.full((nr, nc), np.inf)
    hlab = np.full((nr, nc), -1, np.int64)
    for r in range(nr):
        for c in range(nc):
            if is_target[r, c]:
                h[r, c] = 0.0
                hl[r, c] = 0.0
                hlab[r, c] = lab[r, c]
                continue
            best = np.inf
            for k in range(8):
                rr = r + offsets[k, 0]
                cc = c + offsets[k, 1]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                    continue
                cm = cost[rr, cc]
                if cm == np.inf or z[rr, cc] - z[r, c] > max_rise * lengths[k]:
                    continue
                cand = cm + 0.5 * lam[rr, cc] * lengths[k]
                cl = plen[rr, cc] + lengths[k]
                if cand < best or (cand == best and cl < hl[r, c]):
                    best = cand
                    hl[r, c] = cl
                    hlab[r, c] = lab[rr, cc]
            h[r, c] = best
    return h, hl, hlab


def least_cost_connectivity(
    wip,
    streams,
    dx,
    dy,
    ref_length=None,
    valid=None,
    labels=None,
    min_wip=0.0,
    path_threshold=1.0,
    dem=None,
    dem_valid=None,
    downstream_only=False,
    max_uphill_slope=0.01,
    depressions="fill",
):
    """Probability that each pixel is a wetland connected to the stream network.

    Parameters
    ----------
    wip : 2-D array of wetland probability in [0, 1].
    streams : 2-D bool array, True on stream (target) pixels.
    dx, dy : column and row spacing in map units (metres).
    ref_length : length over which a pixel's probability applies once (see
        ``grid.attenuation``). Defaults to the pixel size, which reduces to a
        plain sum of ``-ln(wip)`` over pixels on an orthogonal path. Fix it to
        a constant when comparing rasters of different resolution.
    valid : 2-D bool array of usable pixels (default: finite ``wip``).
    labels : 2-D int array of stream ids at stream pixels (default: 1).
    min_wip : optional floor on ``wip`` so zero-probability pixels become
        passable at high cost instead of acting as barriers.
    path_threshold : probability at or above which a pixel costs nothing to
        cross; travel uses ``min(wip / path_threshold, 1)``. The at-site term
        in ``p_joint`` always uses the raw ``wip``.
    dem, dem_valid : elevation on the same grid, needed for ``downstream_only``.
    downstream_only : only allow paths that run downhill or level toward the
        stream (water-borne transport). Default False: any direction
        (e.g. habitat connectivity).
    max_uphill_slope : with ``downstream_only``, the steepest climb (rise over
        step length, 0.01 = 1%) a single step may make, to tolerate DEM noise.
    depressions : with ``downstream_only``, ``"fill"`` fills closed
        depressions to their spill point so they drain out; ``"sink"`` keeps
        them, so pixels that can't climb out within ``max_uphill_slope``
        are not connected.

    Returns
    -------
    dict of 2-D float arrays:
        ``p_joint``     wip x p_conn: wetland here AND connected.
        ``p_conn``      probability of connection given this pixel is wetland.
        ``cost``        -ln(p_conn).
        ``path_length`` length of the chosen path (map units).
        ``reach_id``    label of the stream pixel reached.
    """
    wip = np.asarray(wip, dtype=np.float64)
    streams = np.asarray(streams, dtype=bool)
    if valid is None:
        valid = np.isfinite(wip)
    if ref_length is None:
        ref_length = float(np.sqrt(dx * dy))
    if labels is None:
        labels = np.ones(wip.shape, np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    check_wip(wip, valid)

    if downstream_only:
        if dem is None:
            raise ValueError("downstream_only needs a dem.")
        z, dem_valid = _condition_dem(dem, dem_valid, streams, depressions)
        valid = valid & dem_valid
        max_rise = float(max_uphill_slope)
    else:
        z = np.zeros(wip.shape)
        max_rise = 0.0

    lam = attenuation(wip, valid, ref_length, min_wip, path_threshold)
    # Stream pixels are connected by definition, even where wip is missing.
    lam[streams] = 0.0

    lengths = step_lengths(dx, dy)
    cost, plen, lab = _dijkstra(lam, streams, labels, z, max_rise, OFFSETS, lengths)
    h, hl, hlab = _exclude_source(
        cost, plen, lab, lam, streams, z, max_rise, OFFSETS, lengths
    )

    out_mask = ~(valid | streams)
    p_conn = np.exp(-h)
    p_joint = np.where(valid, wip, np.nan) * p_conn
    result = {
        "p_joint": p_joint,
        "p_conn": p_conn,
        "cost": h,
        "path_length": np.where(np.isfinite(h), hl, np.nan),
        "reach_id": np.where(hlab >= 0, hlab, np.nan).astype(np.float64),
    }
    for arr in result.values():
        arr[out_mask] = np.nan
    return result


def _condition_dem(dem, dem_valid, streams, depressions):
    """Elevations for the downstream-only rule.

    Stream pixels without DEM data are set to ``-inf`` so they can always be
    stepped into; other nodata pixels are left NaN (they are barriers anyway).
    """
    dem = np.asarray(dem, dtype=np.float64)
    if dem.shape != streams.shape:
        raise ValueError("wip and dem must be on the same grid.")
    if dem_valid is None:
        dem_valid = np.isfinite(dem)
    if depressions == "fill":
        z, _ = fill_depressions(dem, dem_valid)
    elif depressions == "sink":
        z = np.where(dem_valid, dem, np.nan)
    else:
        raise ValueError("depressions must be 'fill' or 'sink'.")
    z[streams & ~dem_valid] = -np.inf
    return z, dem_valid
