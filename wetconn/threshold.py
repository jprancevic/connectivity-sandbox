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


def focal_mean(wip, valid, size):
    """Mean of the valid pixels in a ``size`` x ``size`` window around each pixel.

    Nodata and pixels beyond the raster edge are left out of the mean, so
    windows near the edge of the domain average fewer pixels. Invalid pixels
    stay NaN. Uses summed-area tables, so the cost doesn't grow with ``size``.
    """
    if size < 1 or size % 2 == 0:
        raise ValueError("window size must be a positive odd number of pixels.")
    if size == 1:
        return np.where(valid, wip, np.nan)

    def box_sum(a):
        h = size // 2
        s = np.pad(np.pad(a, h).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]

    total = box_sum(np.where(valid, wip, 0.0))
    count = box_sum(valid.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.clip(total / count, 0.0, 1.0)
    return np.where(valid, mean, np.nan)


def multiscale_threshold(
    wip,
    streams,
    dx,
    dy,
    windows=(1, 3, 5, 7, 9, 11, 13, 15, 17),
    valid=None,
    labels=None,
    on_window=None,
    **kwargs,
):
    """Connection threshold with routes judged on focal means of growing size.

    For each window size ``k`` the route to the stream is evaluated on the
    ``k`` x ``k`` focal mean of ``wip``, so short dry breaks inside wetter
    ground are averaged over. The pixel itself always uses its raw ``wip``:
    ``threshold_k = min(wip, route_threshold on the focal mean)``, so no
    scale can exceed the pixel's own value. Window 1 is the plain method.

    ``on_window(k, result)`` is called after each scale with that scale's
    ``threshold``, ``route_threshold``, ``reach_id`` and ``wip_avg``, so
    callers can write each one out without keeping them all in memory.
    Other keyword arguments go to ``threshold_connectivity``.

    Returns
    -------
    dict of 2-D float arrays:
        ``threshold``  the highest threshold over all scales.
        ``window``     the smallest window (pixels) that achieves it.
        ``reach_id``   the stream reached at that window.
    """
    wip = np.asarray(wip, dtype=np.float64)
    streams = np.asarray(streams, dtype=bool)
    if valid is None:
        valid = np.isfinite(wip)
    raw = np.where(valid, np.minimum(wip, 1.0), np.nan)

    best = best_window = best_reach = None
    for size in sorted(set(windows)):
        avg = focal_mean(wip, valid, size)
        r = threshold_connectivity(avg, streams, dx, dy, valid, labels, **kwargs)
        route = r["route_threshold"]
        threshold = np.where(route == NEVER, NEVER, np.minimum(raw, route))
        threshold[np.isnan(route)] = np.nan
        threshold[streams] = raw[streams]
        result = {
            "threshold": threshold,
            "route_threshold": route,
            "reach_id": r["reach_id"],
            "wip_avg": avg,
        }
        if on_window is not None:
            on_window(size, result)
        if best is None:
            best = threshold.copy()
            best_window = np.where(np.isnan(threshold), np.nan, float(size))
            best_reach = r["reach_id"].copy()
        else:
            better = threshold > best  # strict: ties keep the smaller window
            best[better] = threshold[better]
            best_window[better] = size
            best_reach[better] = r["reach_id"][better]
    return {"threshold": best, "window": best_window, "reach_id": best_reach}
