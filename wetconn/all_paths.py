"""Method 3: probability that at least one wetland route to the stream exists.

Monte Carlo over random wetland maps. In each map every pixel is passable
with probability ``min(wip / path_threshold, 1)``, independently of the
others. A pixel is connected in that map if, were it wetland, some chain of
passable pixels would link it to a stream. Averaging over maps credits every
route at once, so several routes, or a wide corridor, raise the probability
above the single best path of method 1.
"""

import numpy as np
from numba import njit

from .grid import OFFSETS, check_wip, step_lengths
from .least_cost import downstream_elevations


@njit(cache=True)
def _simulate(p_pass, is_target, z, max_rise, offsets, lengths, n_sims, seed):
    """Count, per pixel, the maps in which it would be connected if wetland.

    Each map is a breadth-first search outward from the stream pixels over
    passable pixels. A pixel's passability is drawn the first time the search
    touches it in that map, so untouched ground costs nothing. Per-map marks
    store the map index, so nothing is reset between maps.

    A step toward the stream may rise at most ``max_rise * d``; with ``z`` all
    zeros every step is allowed.
    """
    np.random.seed(seed)
    nr, nc = p_pass.shape
    n = nr * nc
    pp = p_pass.ravel()
    tgt = is_target.ravel()
    zf = z.ravel()
    targets = np.nonzero(tgt)[0]

    count = np.zeros(n, np.int32)
    drawn = np.full(n, -1, np.int32)  # map index in which passability was drawn
    passable = np.zeros(n, np.bool_)
    reached = np.full(n, -1, np.int32)  # map index in which the search reached it
    connected = np.full(n, -1, np.int32)  # map index in which it was counted
    queue = np.empty(n, np.int64)

    for s in range(n_sims):
        head = 0
        tail = 0
        for i in targets:
            reached[i] = s
            queue[tail] = i
            tail += 1
        while head < tail:
            u = queue[head]
            head += 1
            r = u // nc
            c = u - r * nc
            for k in range(8):
                rr = r + offsets[k, 0]
                cc = c + offsets[k, 1]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                    continue
                v = rr * nc + cc
                # Travel runs v -> u (toward the stream): limit the climb.
                if zf[u] - zf[v] > max_rise * lengths[k]:
                    continue
                # v would be connected if it were wetland.
                if connected[v] != s:
                    connected[v] = s
                    count[v] += 1
                if reached[v] == s:
                    continue
                if drawn[v] != s:
                    drawn[v] = s
                    passable[v] = np.random.random() < pp[v]
                if passable[v]:
                    reached[v] = s
                    queue[tail] = v
                    tail += 1
    return count.reshape(nr, nc)


def all_paths_connectivity(
    wip,
    streams,
    dx,
    dy,
    valid=None,
    min_wip=0.0,
    path_threshold=1.0,
    dem=None,
    dem_valid=None,
    downstream_only=False,
    max_uphill_slope=0.01,
    depressions="fill",
    n_sims=500,
    seed=0,
):
    """Probability that each pixel is a wetland with some wetland route to a stream.

    Parameters are as for ``least_cost_connectivity``, plus:

    n_sims : number of random wetland maps. The standard error of ``p_conn``
        is at most ``0.5 / sqrt(n_sims)`` (0.022 for 500).
    seed : random seed; the same seed and inputs give identical results.

    Pixels are independent in each map, which matches method 1 with
    ``ref_length`` equal to the pixel size. Diagonal and orthogonal steps
    both cross one pixel here (method 1 charges diagonals by length).

    Returns
    -------
    dict of 2-D float arrays:
        ``p_joint``    wip x p_conn: wetland here AND connected.
        ``p_conn``     probability that some route exists, given this pixel
                       is wetland.
        ``p_conn_se``  Monte Carlo standard error of ``p_conn``.
    """
    wip = np.asarray(wip, dtype=np.float64)
    streams = np.asarray(streams, dtype=bool)
    if valid is None:
        valid = np.isfinite(wip)
    if not 0 < path_threshold <= 1:
        raise ValueError("path_threshold must be in (0, 1].")
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1.")
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

    p = np.where(valid, np.minimum(wip, 1.0), 0.0)
    if min_wip > 0:
        p = np.where(valid, np.maximum(p, min_wip), 0.0)
    p_pass = np.minimum(p / path_threshold, 1.0)
    p_pass[streams] = 1.0

    count = _simulate(
        p_pass, streams, z, max_rise, OFFSETS, step_lengths(dx, dy), int(n_sims), int(seed)
    )
    p_conn = count / n_sims
    p_conn[streams] = 1.0
    result = {
        "p_joint": np.where(valid, wip, np.nan) * p_conn,
        "p_conn": p_conn,
        "p_conn_se": np.sqrt(p_conn * (1 - p_conn) / n_sims),
    }
    out_mask = ~(valid | streams)
    for arr in result.values():
        arr[out_mask] = np.nan
    return result
