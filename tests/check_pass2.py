"""Independent check of pass 2: drains the pits of <prefix>_pass1.tif with a literal, pit-by-pit
procedure and compares the result with the tool's <prefix>_labels.tif and <prefix>_pits.csv.

    python check_pass2.py DEM OUT_PREFIX

The literal procedure: take pits in order of floor elevation. For a pit not yet drained, find the lowest
crossing out of its catchment (spill height = the higher of the cells on either side; off the DEM or into
nodata = the inside cell's elevation; ties broken by the steepest drop across the crossing, then by the
inside cell's raster order, then by neighbour direction NE, E, SE, S, SW, W, NW, N). If the cell
across belongs to a pit that hasn't drained, merge the two and let the merged pit drain on its own turn;
otherwise every cell of the pit takes that cell's label.
"""
import argparse

import numpy as np
import pandas as pd
import rasterio

ap = argparse.ArgumentParser()
ap.add_argument("dem")
ap.add_argument("out_prefix")
a = ap.parse_args()
NODATA = -2147483648

src = rasterio.open(a.dem)
dx, dy = src.res
z = src.read(1, masked=True).astype(np.float64).filled(np.nan)
nr, nc = z.shape
L1 = rasterio.open(f"{a.out_prefix}_pass1.tif").read(1).astype(np.int64)
LF = rasterio.open(f"{a.out_prefix}_labels.tif").read(1).astype(np.int64)
pits = pd.read_csv(f"{a.out_prefix}_pits.csv")
zf, l1 = z.ravel(), L1.ravel()

pc = np.flatnonzero((l1 < 0) & (l1 != NODATA))
pid = -l1[pc]
o = np.argsort(pid, kind="stable")
pc, pid = pc[o], pid[o]
starts = np.searchsorted(pid, np.arange(1, len(pits) + 2))
members = {p: [pc[starts[p - 1]:starts[p]]] for p in range(1, len(pits) + 1)}
root = np.arange(len(pits) + 1)
final = {}
DR = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
DC = np.array([1, 1, 1, 0, -1, -1, -1, 0])
ties = 0

for p in pits.sort_values(["floor_elevation", "pit_id"], kind="stable").pit_id:
    if root[p] != p or p in final:
        continue
    cells = np.concatenate(members[p])
    r, c = np.divmod(cells, nc)
    S, TO, G, CELL, K = [], [], [], [], []
    for k in range(8):
        rr, cc = r + DR[k], c + DC[k]
        off = (rr < 0) | (rr >= nr) | (cc < 0) | (cc >= nc)
        nb = np.where(off, 0, rr.clip(0, nr - 1) * nc + cc.clip(0, nc - 1))
        nl = np.where(off, NODATA, l1[nb])
        ext = off | (nl == NODATA)
        s = np.where(ext, zf[cells], np.maximum(zf[cells], zf[nb]))
        g = np.where(ext, 0.0, (zf[cells] - zf[nb]) / np.hypot(DR[k] * dy, DC[k] * dx))
        to = np.where(ext, 0, nl)
        inside = (to < 0) & (root[np.where(to < 0, -to, 0)] == p)
        S.append(s[~inside]); TO.append(to[~inside]); G.append(g[~inside])
        CELL.append(cells[~inside]); K.append(np.full((~inside).sum(), k))
    s, to, g, cell, kk = (np.concatenate(x) for x in (S, TO, G, CELL, K))
    at = np.flatnonzero(s == s.min())
    at = at[g[at] == g[at].max()]
    cand = to[at]
    first = at[np.lexsort((kk[at], cell[at]))[0]]      # exact ties: first inside cell, then direction

    def meaning(t):
        if t >= 0:
            return ("label", t)
        b = root[-t]
        return ("label", final[b]) if b in final else ("pit", b)

    if len({meaning(t) for t in cand}) > 1:
        ties += 1
    kind, v = meaning(to[first])
    if kind == "label":
        final[p] = v
    else:
        root[root == p] = v
        members[v] += members[p]

drains = np.array([final.get(root[p], -999) for p in range(1, len(pits) + 1)])
exp = l1.copy()
exp[pc] = drains[pid - 1]
bad_cells = int((exp.reshape(nr, nc) != LF).sum())
bad_pits = int((drains != pits.drains_to.to_numpy()).sum())
print(f"pits: {len(pits)}, left undrained: {(drains == -999).sum()}, "
      f"exact ties settled by raster order: {ties}")
print(f"cells that differ: {bad_cells:,} of {LF.size:,}; pits whose destination differs: {bad_pits}")
ok = bad_cells == 0 and bad_pits == 0 and (drains != -999).all()
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
