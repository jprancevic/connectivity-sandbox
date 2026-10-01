"""Method 4: the wetness threshold at which each pixel joins the stream network.

WIP is read as the condition needed to wet a pixel: at threshold ``t`` the
pixels with ``wip >= t`` are wet. Lowering ``t`` wets more of the landscape,
and a pixel connects once it and some route of wet pixels to a stream are
all at or above ``t``. Its connection threshold is therefore the highest
such ``t``: the lowest WIP on its best route (including itself), maximised
over routes. One priority flood from the streams gives that for every
threshold at once; the connected wetland map at any ``t`` is
``threshold >= t``.
"""

import numpy as np
from numba import njit

from .grid import OFFSETS, check_wip, step_lengths
from .least_cost import downstream_elevations

NEVER = -1.0  # written where no allowed route exists (downstream-only)


@njit(cache=True)
def _before(a, b, key, plen):
    """Heap order: widest route first, then the shorter one."""
    return key[a] < key[b] or (key[a] == key[b] and plen[a] < plen[b])


@njit(cache=True)
def _up(heap, pos, key, plen, i):
    node = heap[i]
    while i > 0:
        parent = (i - 1) >> 1
        if not _before(node, heap[parent], key, plen):
            break
        heap[i] = heap[parent]
        pos[heap[i]] = i
        i = parent
    heap[i] = node
    pos[node] = i


@njit(cache=True)
def _down(heap, pos, key, plen, i, size):
    node = heap[i]
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and _before(heap[child + 1], heap[child], key, plen):
            child += 1
        if not _before(heap[child], node, key, plen):
            break
        heap[i] = heap[child]
        pos[heap[i]] = i
        i = child
    heap[i] = node
    pos[node] = i


@njit(cache=True)
def _widest_routes(wip, passable, is_target, label, z, max_rise, offsets, lengths):
    """Bottleneck (max-min) flood outward from the target pixels.

    ``key`` holds minus the bottleneck so the heap pops the widest route
    first; among equally wide routes the shorter wins, which makes
    ``reach_id`` the nearest stream. Targets start at +inf: reaching any
    stream pixel connects. A step toward the stream may rise at most
    ``max_rise * d``.
    """
    nr, nc = wip.shape
    n = nr * nc
    wf = wip.ravel()
    ok = passable.ravel()
    tgt = is_target.ravel()
    labf = label.ravel()
    zf = z.ravel()
    key = np.full(n, np.inf)
    plen = np.full(n, np.inf)
    lab = np.full(n, -1, np.int64)
    heap = np.empty(n, np.int64)
    pos = np.full(n, -1, np.int64)  # -1: never queued, -2: settled

    size = 0
    for i in range(n):
        if tgt[i]:
            key[i] = -np.inf
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
            _down(heap, pos, key, plen, 0, size)
        r = u // nc
        c = u - r * nc
        for k in range(8):
            rr = r + offsets[k, 0]
            cc = c + offsets[k, 1]
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            v = rr * nc + cc
            if pos[v] == -2 or not ok[v]:
                continue
            # Travel runs v -> u (toward the stream): limit the climb.
            if zf[u] - zf[v] > max_rise * lengths[k]:
                continue
            cand = max(key[u], -wf[v])  # minus min(bottleneck(u), wip(v))
            cl = plen[u] + lengths[k]
            if cand < key[v] or (cand == key[v] and cl < plen[v]):
                key[v] = cand
                plen[v] = cl
                lab[v] = lab[u]
                if pos[v] == -1:
                    heap[size] = v
                    pos[v] = size
                    size += 1
                    _up(heap, pos, key, plen, size - 1)
                else:
                    _up(heap, pos, key, plen, pos[v])
    return (-key).reshape(nr, nc), plen.reshape(nr, nc), lab.reshape(nr, nc)


@njit(cache=True)
def _route_only(width, plen, lab, is_target, z, max_rise, offsets, lengths):
    """Best route through a neighbour, leaving the pixel itself out."""
    nr, nc = width.shape
    route = np.full((nr, nc), -np.inf)
    rlen = np.full((nr, nc), np.inf)
    rlab = np.full((nr, nc), -1, np.int64)
    for r in range(nr):
        for c in range(nc):
            if is_target[r, c]:
                route[r, c] = np.inf
                rlen[r, c] = 0.0
                rlab[r, c] = lab[r, c]
                continue
            for k in range(8):
                rr = r + offsets[k, 0]
                cc = c + offsets[k, 1]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                    continue
                if z[rr, cc] - z[r, c] > max_rise * lengths[k]:
                    continue
                w = width[rr, cc]
                cl = plen[rr, cc] + lengths[k]
                if w > route[r, c] or (w == route[r, c] and cl < rlen[r, c]):
                    route[r, c] = w
                    rlen[r, c] = cl
                    rlab[r, c] = lab[rr, cc]
    return route, rlab


def threshold_connectivity(
    wip,
    streams,
    dx,
    dy,
    valid=None,
    labels=None,
    dem=None,
    dem_valid=None,
    downstream_only=False,
    max_uphill_slope=0.01,
    depressions="fill",
):
    """Highest wetness threshold at which each pixel is wet and connected.

    Parameters are as for ``least_cost_connectivity``.

    Returns
    -------
    dict of 2-D float arrays:
        ``threshold``        min(own wip, route_threshold): the highest
                             ``t`` at which the pixel is wet and joined to a
                             stream. ``NEVER`` (-1) if no allowed route exists.
        ``route_threshold``  the same for the route alone, i.e. the threshold
                             at which the pixel would connect if it were wet
                             (1 next to a stream).
        ``reach_id``         label of the stream the best route reaches (the
                             nearest one among equally good routes).
    Stream pixels have ``threshold = wip`` and ``route_threshold = 1``.
    """
    wip = np.asarray(wip, dtype=np.float64)
    streams = np.asarray(streams, dtype=bool)
    if valid is None:
        valid = np.isfinite(wip)
    if labels is None:
        labels = np.ones(wip.shape, np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    check_wip(wip, valid)

    if downstream_only:
        if dem is None:
            raise ValueError("downstream_only needs a dem.")
        z, dem_valid = downstream_elevations(dem, dem_valid, streams, depressions)
        valid = valid & dem_valid
        max_rise = float(max_uphill_slope)
    else:
        z = np.zeros(wip.shape)
        max_rise = 0.0

    w = np.where(valid, np.minimum(wip, 1.0), 0.0)
    lengths = step_lengths(dx, dy)
    width, plen, lab = _widest_routes(w, valid, streams, labels, z, max_rise, OFFSETS, lengths)
    route, rlab = _route_only(width, plen, lab, streams, z, max_rise, OFFSETS, lengths)

    route = np.minimum(route, 1.0)
    connected = np.isfinite(route) & (route >= 0)
    threshold = np.where(connected, np.minimum(np.where(valid, wip, np.nan), route), NEVER)
    threshold[streams] = np.where(valid, wip, np.nan)[streams]
    result = {
        "threshold": threshold,
        "route_threshold": np.where(connected, route, NEVER),
        "reach_id": np.where(connected & (rlab >= 0), rlab, np.nan).astype(np.float64),
    }
    out_mask = ~(valid | streams)
    for arr in result.values():
        arr[out_mask] = np.nan
    return result
