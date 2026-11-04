"""Numba kernels for wetland drainage areas: D8 routing (pass 1) and pit draining (pass 2).

Copied unchanged from wetland_drainage_areas.ipynb. The constants stay at module level because numba
freezes globals at compile time.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

EXTERNAL = 0                     # leaves the DEM without reaching a wetland
NODATA = -2147483648
UNSET = -2147483647              # not labelled yet (internal)
DR = np.array([-1, 0, 1, 1, 1, 0, -1, -1])   # neighbours NE, E, SE, S, SW, W, NW, N
DC = np.array([1, 1, 1, 0, -1, -1, -1, 0])
FLAT, OUT, WET = 8, 9, 10


@njit
def pass1(z, lab, dx, dy):
    """D8 routing. Fills lab in place; returns pit floor locations and sizes."""
    nr, nc = z.shape
    dist = np.sqrt((DR * dy) ** 2 + (DC * dx) ** 2)
    dirs = np.full((nr, nc), 255, dtype=np.uint8)
    d, l = dirs.ravel(), lab.ravel()             # flat views

    # steepest strictly-lower neighbour; edge cells with none drain out of the DEM
    n_flat = 0
    for r in range(nr):
        for c in range(nc):
            if np.isnan(z[r, c]):
                continue
            if lab[r, c] != UNSET:          # wetland or stream: flow stops here
                dirs[r, c] = WET
                continue
            edge = r == 0 or r == nr - 1 or c == 0 or c == nc - 1
            best, best_s = -1, 0.0
            for k in range(8):
                rr, cc = r + DR[k], c + DC[k]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc:
                    continue
                if np.isnan(z[rr, cc]):
                    edge = True
                    continue
                s = (z[r, c] - z[rr, cc]) / dist[k]
                if s > best_s:
                    best, best_s = k, s
            if best >= 0:
                dirs[r, c] = best
            elif edge:
                dirs[r, c] = OUT
                lab[r, c] = EXTERNAL
            else:
                dirs[r, c] = FLAT
                n_flat += 1

    # flats: breadth-first from same-elevation exits (stored as k + 16 until the scan ends,
    # so a flat cell assigned during the scan isn't mistaken for an exit)
    queue = np.empty(max(n_flat, 1), dtype=np.int64)
    head = tail = 0
    for r in range(nr):
        for c in range(nc):
            if dirs[r, c] != FLAT:
                continue
            for k in range(8):
                rr, cc = r + DR[k], c + DC[k]
                if 0 <= rr < nr and 0 <= cc < nc and z[rr, cc] == z[r, c] and \
                        (dirs[rr, cc] <= 7 or dirs[rr, cc] == OUT or dirs[rr, cc] == WET):
                    dirs[r, c] = k + 16
                    queue[tail] = r * nc + c
                    tail += 1
                    break
    for t in range(tail):
        d[queue[t]] -= 16
    while head < tail:
        r, c = divmod(queue[head], nc)
        head += 1
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if 0 <= rr < nr and 0 <= cc < nc and dirs[rr, cc] == FLAT and z[rr, cc] == z[r, c]:
                dirs[rr, cc] = (k + 4) % 8
                queue[tail] = rr * nc + cc
                tail += 1

    # pits: each remaining closed flat or single cell gets an ID; its cells are labelled -ID
    n_pits = 0
    pit_cell = np.empty(n_flat + 1, dtype=np.int64)
    pit_floor = np.zeros(n_flat + 1, dtype=np.int64)
    for i in range(nr * nc):
        if d[i] != FLAT:
            continue
        n_pits += 1
        pit_cell[n_pits] = i
        d[i] = WET                  # closed from here on
        l[i] = -n_pits
        head, tail = 0, 1
        queue[0] = i
        while head < tail:
            r, c = divmod(queue[head], nc)
            head += 1
            pit_floor[n_pits] += 1
            for k in range(8):
                rr, cc = r + DR[k], c + DC[k]
                if 0 <= rr < nr and 0 <= cc < nc and dirs[rr, cc] == FLAT and z[rr, cc] == z[r, c]:
                    dirs[rr, cc] = WET
                    lab[rr, cc] = -n_pits
                    queue[tail] = rr * nc + cc
                    tail += 1

    # every other cell: follow the path down to the first labelled cell, then label the whole path
    off = DR * nc + DC
    path = np.empty(1 << 16, dtype=np.int64)
    for i in range(nr * nc):
        if l[i] != UNSET or d[i] > 7:
            continue
        n, j = 0, i
        while l[j] == UNSET:
            if n == path.size:
                path = np.concatenate((path, np.empty_like(path)))
            path[n] = j
            n += 1
            j += off[d[j]]
        for t in range(n):
            l[path[t]] = l[j]
    return n_pits, pit_cell[1:n_pits + 1], pit_floor[1:n_pits + 1]


@njit
def find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit
def pass2(z, lab, n_pits, dx, dy):
    """Drain pits from the lowest outlet up. Relabels pit cells in lab; returns per-pit outlet info."""
    nr, nc = z.shape

    # lowest crossing from each pit catchment into each neighbouring label
    slot = Dict.empty(types.int64, types.int64)
    cap = 1 << 16
    e_pit = np.empty(cap, np.int64)
    e_to = np.empty(cap, np.int64)
    e_z = np.empty(cap, np.float64)
    e_slope = np.empty(cap, np.float64)
    e_cell = np.empty(cap, np.int64)
    e_k = np.empty(cap, np.int64)
    dist = np.sqrt((DR * dy) ** 2 + (DC * dx) ** 2)
    n = 0
    for r in range(nr):
        for c in range(nc):
            p = -np.int64(lab[r, c])
            if p <= 0 or lab[r, c] == NODATA:
                continue
            for k in range(8):
                rr, cc = r + DR[k], c + DC[k]
                if rr < 0 or rr >= nr or cc < 0 or cc >= nc or lab[rr, cc] == NODATA:
                    to, s, g = EXTERNAL, z[r, c], 0.0        # spills off the DEM here
                else:
                    to = np.int64(lab[rr, cc])
                    if to == -p:
                        continue
                    s = max(z[r, c], z[rr, cc])
                    g = (z[r, c] - z[rr, cc]) / dist[k]     # ties: steepest way out wins
                key = (p << 32) | (to & 0xFFFFFFFF)
                if key not in slot:
                    if n == cap:
                        cap *= 2
                        e_pit = np.concatenate((e_pit, np.empty(n, np.int64)))
                        e_to = np.concatenate((e_to, np.empty(n, np.int64)))
                        e_z = np.concatenate((e_z, np.empty(n, np.float64)))
                        e_slope = np.concatenate((e_slope, np.empty(n, np.float64)))
                        e_cell = np.concatenate((e_cell, np.empty(n, np.int64)))
                        e_k = np.concatenate((e_k, np.empty(n, np.int64)))
                    slot[key] = n
                    e_pit[n], e_to[n], e_z[n], e_slope[n], e_cell[n], e_k[n] = p, to, s, g, r * nc + c, k
                    n += 1
                else:
                    j = slot[key]
                    if s < e_z[j] or (s == e_z[j] and g > e_slope[j]):
                        e_z[j], e_slope[j], e_cell[j], e_k[j] = s, g, r * nc + c, k

    # take crossings from lowest to highest
    parent = np.arange(n_pits + 1)
    final = np.full(n_pits + 1, UNSET, np.int64)
    spill = np.full(n_pits + 1, np.nan)
    outlet = np.full(n_pits + 1, -1, np.int64)
    # order: lowest spill, then steepest, then (exact ties) inside cell in raster order, then direction
    order = np.argsort(e_k[:n], kind="mergesort")
    order = order[np.argsort(e_cell[:n][order], kind="mergesort")]
    order = order[np.argsort(-e_slope[:n][order], kind="mergesort")]
    order = order[np.argsort(e_z[:n][order], kind="mergesort")]
    for t in order:
        a = find(parent, e_pit[t])
        if final[a] != UNSET:
            continue                                        # already drained
        to = e_to[t]
        if to < 0:                                          # crossing into another pit
            b = find(parent, -to)
            if a == b:
                continue
            if final[b] == UNSET:                           # both still closed: they merge
                for q in (a, b):
                    if np.isnan(spill[q]):
                        spill[q], outlet[q] = e_z[t], e_cell[t]
                parent[a] = b
                continue
            to = final[b]
        if np.isnan(spill[a]):
            spill[a], outlet[a] = e_z[t], e_cell[t]
        final[a] = to

    drains_to = np.empty(n_pits + 1, np.int64)
    for p in range(1, n_pits + 1):
        drains_to[p] = final[find(parent, p)]
    l = lab.ravel()
    for i in range(nr * nc):
        v = l[i]
        if v < 0 and v != NODATA and drains_to[-v] != UNSET:
            l[i] = drains_to[-v]
    return spill[1:], outlet[1:], drains_to[1:]
