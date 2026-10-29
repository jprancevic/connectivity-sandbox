"""Method 2: probability of connection along the D8 steepest-descent flow path.

Each pixel follows its single steepest-descent path (slope measured as drop
over the true step length, so diagonals are compared fairly) until it reaches
a stream pixel. The cost along that path uses the same length-weighted
``-ln(wip)`` as method 1.
"""

import numpy as np
from numba import njit

from .grid import OFFSETS, attenuation, check_wip, step_lengths

# Status codes written to the ``status`` output.
NODATA = 0
REACHES_STREAM = 1
LEAVES_DOMAIN = 2  # flows off the raster edge or into a nodata pixel
CLOSED_DEPRESSION = 3  # ends in a pit (only when depressions="sink")


@njit(cache=True)
def _heap_push(hz, hi, size, z, i):
    j = size
    while j > 0:
        p = (j - 1) >> 1
        if hz[p] <= z:
            break
        hz[j] = hz[p]
        hi[j] = hi[p]
        j = p
    hz[j] = z
    hi[j] = i
    return size + 1


@njit(cache=True)
def _heap_pop(hz, hi, size):
    z0 = hz[0]
    i0 = hi[0]
    size -= 1
    if size > 0:
        z = hz[size]
        i = hi[size]
        j = 0
        while True:
            c = 2 * j + 1
            if c >= size:
                break
            if c + 1 < size and hz[c + 1] < hz[c]:
                c += 1
            if hz[c] >= z:
                break
            hz[j] = hz[c]
            hi[j] = hi[c]
            j = c
        hz[j] = z
        hi[j] = i
    return z0, i0, size


@njit(cache=True)
def _boundary(valid, offsets):
    """Valid pixels on the raster edge or next to a nodata pixel."""
    nr, nc = valid.shape
    out = np.zeros((nr, nc), np.bool_)
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            if r == 0 or c == 0 or r == nr - 1 or c == nc - 1:
                out[r, c] = True
                continue
            for k in range(8):
                if not valid[r + offsets[k, 0], c + offsets[k, 1]]:
                    out[r, c] = True
                    break
    return out


@njit(cache=True)
def _priority_flood(z, valid, boundary, offsets):
    """Priority-Flood + epsilon (Barnes et al., 2014).

    Fills every depression to its spill point and adds the smallest float64
    increment across flats, so every pixel has a strictly lower neighbour
    leading to the domain boundary.
    """
    nr, nc = z.shape
    out = z.copy()
    closed = np.zeros((nr, nc), np.bool_)
    n = nr * nc
    hz = np.empty(n, np.float64)
    hi = np.empty(n, np.int64)
    size = 0
    for r in range(nr):
        for c in range(nc):
            if boundary[r, c]:
                closed[r, c] = True
                size = _heap_push(hz, hi, size, out[r, c], r * nc + c)
    while size > 0:
        zc, i, size = _heap_pop(hz, hi, size)
        r = i // nc
        c = i - r * nc
        for k in range(8):
            rr = r + offsets[k, 0]
            cc = c + offsets[k, 1]
            if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                continue
            if closed[rr, cc] or not valid[rr, cc]:
                continue
            closed[rr, cc] = True
            if out[rr, cc] <= zc:
                out[rr, cc] = np.nextafter(zc, np.inf)
            size = _heap_push(hz, hi, size, out[rr, cc], rr * nc + cc)
    return out


def fill_depressions(z, valid):
    """Fill closed depressions to their spill point (Priority-Flood + epsilon).

    Returns ``(filled, boundary)``. Filled cells are raised by the smallest
    float64 steps toward the outlet, far below any real slope.
    """
    boundary = _boundary(valid, OFFSETS)
    filled = _priority_flood(np.where(valid, z, np.nan), valid, boundary, OFFSETS)
    return filled, boundary


@njit(cache=True)
def _d8(z, valid, offsets, lengths):
    """Steepest-descent receiver (flat index) and direction index per pixel."""
    nr, nc = z.shape
    receiver = np.full(nr * nc, -1, np.int64)
    direction = np.full(nr * nc, -1, np.int64)
    for r in range(nr):
        for c in range(nc):
            if not valid[r, c]:
                continue
            best = 0.0
            for k in range(8):
                rr = r + offsets[k, 0]
                cc = c + offsets[k, 1]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc or not valid[rr, cc]:
                    continue
                s = (z[r, c] - z[rr, cc]) / lengths[k]
                if s > best:
                    best = s
                    receiver[r * nc + c] = rr * nc + cc
                    direction[r * nc + c] = k
    return receiver, direction


@njit(cache=True)
def _accumulate(receiver, direction, lam, is_target, wip_valid, dem_valid, boundary, label, lengths):
    """Walk every pixel's flow path downstream, memoising along the way.

    ``A`` is the trapezoid-rule cost from a pixel centre to the stream,
    including the pixel's own half-step; ``H`` is the same without it (the
    source pixel is covered by the at-site ``wip`` term).
    """
    n = receiver.size
    lamf = lam.ravel()
    tgt = is_target.ravel()
    wv = wip_valid.ravel()
    dv = dem_valid.ravel()
    bnd = boundary.ravel()
    labf = label.ravel()

    a = np.full(n, np.inf)
    h = np.full(n, np.inf)
    plen = np.full(n, np.inf)
    lab = np.full(n, -1, np.int64)
    status = np.zeros(n, np.int8)
    done = np.zeros(n, np.bool_)
    stack = np.empty(n, np.int64)

    for i in range(n):
        if done[i] or not dv[i]:
            continue
        top = 0
        j = i
        # Walk down until we hit a pixel whose outcome is already known.
        while not done[j]:
            if tgt[j]:
                a[j] = 0.0
                h[j] = 0.0
                plen[j] = 0.0
                lab[j] = labf[j]
                status[j] = 1
                done[j] = True
                break
            if not wv[j]:
                status[j] = 2
                done[j] = True
                break
            r = receiver[j]
            if r < 0:
                status[j] = 2 if bnd[j] else 3
                done[j] = True
                break
            stack[top] = j
            top += 1
            j = r
        # Unwind back up the path.
        while top > 0:
            top -= 1
            j = stack[top]
            r = receiver[j]
            status[j] = status[r]
            if status[r] == 1:
                d = lengths[direction[j]]
                lr = 0.0 if tgt[r] else lamf[r]
                a[j] = a[r] + 0.5 * (lamf[j] + lr) * d
                h[j] = a[r] + 0.5 * lr * d
                plen[j] = plen[r] + d
                lab[j] = lab[r]
            done[j] = True
    return h, plen, lab, status


def flowpath_connectivity(
    wip,
    dem,
    streams,
    dx,
    dy,
    ref_length=None,
    wip_valid=None,
    dem_valid=None,
    labels=None,
    depressions="fill",
    burn_depth=0.0,
    min_wip=0.0,
    path_threshold=1.0,
):
    """Probability that each pixel is a wetland connected along its flow path.

    Parameters are as for ``least_cost_connectivity`` plus:

    dem : 2-D elevation array on the same grid as ``wip``.
    dem_valid : 2-D bool array of usable DEM pixels (default: finite ``dem``).
    depressions : ``"fill"`` routes flow out of closed depressions through
        their spill point (fill-and-spill); ``"sink"`` leaves them as pits, so
        pixels draining into them are not connected (``p_conn = 0``).
    burn_depth : lower stream pixels by this much (DEM units) before routing,
        to force flow paths onto the mapped flowlines.

    Returns the same dict as ``least_cost_connectivity`` plus ``status``
    (see module constants). ``p_conn`` is NaN where the flow path leaves the
    domain before reaching a stream, since the outcome is unknown.
    """
    wip = np.asarray(wip, dtype=np.float64)
    dem = np.asarray(dem, dtype=np.float64)
    streams = np.asarray(streams, dtype=bool)
    if wip.shape != dem.shape:
        raise ValueError("wip and dem must be on the same grid.")
    if wip_valid is None:
        wip_valid = np.isfinite(wip)
    if dem_valid is None:
        dem_valid = np.isfinite(dem)
    if ref_length is None:
        ref_length = float(np.sqrt(dx * dy))
    if labels is None:
        labels = np.ones(wip.shape, np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if depressions not in ("fill", "sink"):
        raise ValueError("depressions must be 'fill' or 'sink'.")
    check_wip(wip, wip_valid)

    lengths = step_lengths(dx, dy)
    z = np.where(dem_valid, dem, np.nan)
    if burn_depth:
        z[streams & dem_valid] -= burn_depth
    if depressions == "fill":
        z, boundary = fill_depressions(z, dem_valid)
    else:
        boundary = _boundary(dem_valid, OFFSETS)
    receiver, direction = _d8(z, dem_valid, OFFSETS, lengths)

    lam = attenuation(wip, wip_valid, ref_length, min_wip, path_threshold)
    lam[streams] = 0.0
    targets = streams & dem_valid
    h, plen, lab, status = _accumulate(
        receiver, direction, lam, targets, wip_valid, dem_valid, boundary, labels, lengths
    )
    shape = wip.shape
    h = h.reshape(shape)
    plen = plen.reshape(shape)
    lab = lab.reshape(shape)
    status = status.reshape(shape)

    reached = status == REACHES_STREAM
    p_conn = np.full(shape, np.nan)
    p_conn[reached] = np.exp(-h[reached])
    p_conn[status == CLOSED_DEPRESSION] = 0.0
    result = {
        "p_joint": np.where(wip_valid, wip, np.nan) * p_conn,
        "p_conn": p_conn,
        "cost": np.where(reached, h, np.nan),
        "path_length": np.where(reached, plen, np.nan),
        "reach_id": np.where(reached & (lab >= 0), lab, np.nan).astype(np.float64),
        "status": status.astype(np.float64),
    }
    out_mask = ~dem_valid | ~(wip_valid | targets)
    for name, arr in result.items():
        if name != "status":
            arr[out_mask] = np.nan
    result["status"][~dem_valid] = NODATA
    return result
