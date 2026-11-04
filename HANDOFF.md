# Wetland drainage areas: handoff

Tools that label every DEM cell with the first wetland its D8 flow path reaches, so you can map which parts of the landscape drain through which wetland polygons. Built and verified in a Claude session; this file carries everything decided there, because that conversation isn't available to you.

## What's in this folder

| file | what it is |
|---|---|
| `wetland_drainage_areas.ipynb` | Reference implementation. Settings cell, one cell of numba kernels (`pass1`, `find`, `pass2`), then the pass 1 cell, the pass 2 cell, and an optional polygons cell. |
| `tests/make_test_data.py` | Builds a synthetic test landscape: DEM, rounded copy of the DEM, wetlands, streams, planted flat cases. |
| `tests/check_pass1.py` | Independent re-implementation of pass 1 (vectorised D8, Python flat routing, sorted-elevation walk). Compares cell by cell. |
| `tests/check_pass2.py` | Independent, literal pit-by-pit pass 2. Compares cell by cell and pit by pit. |
| `tests/check_cases.py` | Spot checks on the planted flat cases and the value scheme. |
| `tests/run_notebook.py` | Runs the notebook headless with Settings overridden (used by the tests until the notebook becomes a package). |
| `tests/run_all.sh` | Runs everything. Takes about a minute. |
| `requirements.txt` | Dependencies. |

Inputs used in the real project (Skykomish, HUC8 17110009):
- DEM: `17110009_Skykomish_DEM_3m_huc8.tif`, 3 m, about 3 GB.
- Wetlands: `tw_wetland_polygons_mid.gpkg`, field `Wetlands ID`, non-negative integers starting at 0.
- Streams: `/Users/Jeff/Dropbox/Documents/TealWaters/Data/skykomish/reaches_100m/reach_sky100a.shp`.

## The algorithm

### Pass 1: route every cell

1. **Burn stop cells into a label grid.** The grid starts as `UNSET`.
   - Wetland polygons are burned as **wetland ID + 1**. Larger polygons go first, so a smaller polygon wins where two overlap.
   - If `STREAMS_PATH` is set, flowlines are burned **after** wetlands, as **0**, with `all_touched=True`. That setting closes the corner-only gaps a diagonal D8 step could slip through. Because streams go second, they overwrite wetland cells where the two overlap.
   - DEM nodata becomes `NODATA` (int32 minimum).
2. **Flow directions.** Every other cell gets a D8 direction to its steepest strictly-lower neighbour: the largest drop ÷ distance, with diagonals at √2 × cell size. On an exact tie the first neighbour in the order NE, E, SE, S, SW, W, NW, N wins. A cell that's already labelled (wetland or stream) is a stop cell and has no direction.
3. **No lower neighbour, on the edge.** A cell with no lower neighbour that sits on the domain edge (grid edge, or next to nodata) drains out of the DEM and gets **0**.
4. **Flats.** A cell with no lower neighbour that's not on the edge is a flat cell. Flat cells drain toward the nearest same-elevation exit (a cell with a direction, a stop cell, or an off-DEM cell) by breadth-first search across equal elevations.
   - The first ring is marked as `k + 16` during its scan, so a flat cell assigned in the same scan isn't mistaken for an exit.
   - This is what makes filled and hydro-flattened DEMs work, and why a flat around a wetland drains into it.
5. **Pits.** Flat cells left over (closed flats, or single-cell pits) are grouped by 8-connectivity at equal elevation. Each group is one pit, numbered in raster order of its first cell and labelled **−pit ID**.
6. **Labelling.** Every remaining cell follows its directions down to the first labelled cell, and the whole path takes that label. This gives the same result as ranking all cells by elevation and walking up the list copying each cell's downslope label (that's what `check_pass1.py` does), without needing the sort.

### Pass 2: drain pits from the lowest outlet up

1. **Candidate crossings.** For every pit catchment (all cells labelled −p) and every label it borders, the lowest crossing is recorded:
   - **Into another cell:** the spill height is the higher of the cells on either side, because water has to rise over whichever is higher.
   - **Off the DEM or into nodata:** the spill height is the inside cell's elevation, and the crossing drains to 0.
2. **Order.** Crossings are taken in this order:
   - lowest spill height first;
   - then the steepest drop across the crossing;
   - then, for exact ties, the inside cell's raster position;
   - then the neighbour direction.
   Exact ties are common on DEMs stored at limited precision (30 of 1,281 pits on the rounded test DEM), so the last two keys keep results deterministic.
3. **Draining.** For each crossing, if the pit (or merged pit group) hasn't drained yet:
   - **Into a wetland, a stream, 0, or an already-drained pit:** the group takes that label.
   - **Into a pit that hasn't drained:** the two groups merge, using union–find, and drain together through their next crossing.
4. **Relabel.** Every pit cell takes its group's final label.

## Outputs

`<PREFIX>_pass1.tif` and `<PREFIX>_labels.tif` are int32, with these values:

| value | `_pass1.tif` | `_labels.tif` (final) |
|---|---|---|
| ≥ 1 | wetland ID + 1 | wetland ID + 1 |
| 0 | reaches a stream, or leaves the DEM, without reaching a wetland | same |
| ≤ −1 | −pit ID | none left |
| −2147483648 | nodata | nodata |

`<PREFIX>_pits.csv` has one row per pit:
- pit ID
- floor row and column
- floor elevation
- floor cells
- catchment cells
- outlet row and column
- spill elevation
- `drains_to`, as a label value

`<PREFIX>_summary.csv` has one row per wetland ID:
- label value
- wetland cells
- drainage cells after pass 1
- drainage cells after pass 2
- drainage area in m²

`<PREFIX>_drainage_areas.gpkg` is optional: dissolved polygons of the final labels.

## Decisions and why

- **D8, not D-infinity.** With D8 each cell has one path, so "the first wetland it hits" has one answer.
- **No Whitebox.** An early version used WhiteboxTools: conditioning, D8, then a seeded watershed. It was dropped:
  - The numba version is simpler and faster (17 s vs 49 s for a 64 M-cell run, measured before pass 2 existed), and uses about 10 bytes per cell instead of about 40.
  - Whitebox silently misreads GeoTIFFs written with a compression predictor.
  - On the same conditioned DEM, the two gave identical labels.
- **No depression filling.** Pits are kept and labelled so you can see where they are. Pass 2 then drains them, which does in effect what filling would do, but keeps the pits on record.
- **Filled DEMs work too.** On a filled DEM there are no pits, so pass 2 does nothing.
- **Filled DEM and Whitebox differ in one case.** A filled depression that contains a wetland becomes a flat with the wetland on it, and the whole flat drains into that wetland. Whitebox's flat routing sends most of it past the wetland toward the spill point. In depressions without a wetland, the two agree on 97% of cells.
- **Value scheme.** Wetland ID + 1 frees 0 for "doesn't reach a wetland". Pit IDs are negative in pass 1, so nodata moved to int32 minimum (it was −9999 earlier, which would collide with pit 9999).
- **Streams overwrite wetlands.** This is a choice. Burning streams first instead would keep wetland cells as wetland.
- **0 means two things.** It covers both "reached a stream" and "left the DEM". Give streams their own value if the two need separating.

## Verification

`bash tests/run_all.sh` builds the test data, runs the notebook on three cases (no streams, streams, rounded DEM with streams), and runs every check. It must print `ALL CHECKS PASSED`.

What the checks establish:
- **Pass 1:** identical, cell for cell on 1.8 M cells, to an independent implementation using the sorted-elevation walk. Pit catchments match as a partition.
- **Pass 2:** identical to a literal pit-by-pit procedure, which takes pits by floor elevation, finds each pit's lowest outlet, and merges when it spills into an undrained pit. Every pit's destination matches.
- **Planted cases:** a flat lake with an outlet produces no pit, a closed flat bowl becomes exactly one pit, and a flat bowl around a wetland drains into the wetland. Wetland ID 0 is stored as 1, and the cell counts add up.
- **The checks can fail.** A deliberately broken D8 tie-break is caught (on the rounded DEM), and so is a broken spill-height rule.

Performance on 64 M cells, 2 cores: pass 1 took 24 s and pass 2 took 12 s, with about 1.1 GB peak memory. Memory is about 10 bytes per cell plus about 1 GB, so the 3 GB DEM (about 750 M cells) should need roughly 8–9 GB.

## Open items

1. **Pass 2 memory on the real DEM.** It keeps one record per (pit, neighbouring label) pair plus a numba dict. A raw 3 m lidar DEM can have millions of pits, which could add a few GB. If that's a problem, store crossings in arrays sorted by key instead of the dict.
2. **Flat handling, still to decide.** Option: drop flat routing from pass 1 and treat every flat as a pit, so pass 2 drains them all. That's simpler code, but a flat with a wetland on it would then usually drain past the wetland to a lower exit instead of into it. The plan was to run both on the real DEM and compare.
3. **Polygonizing is slow** on very large rasters, which is why it's off by default.
4. Possibly give stream cells a value different from "left the DEM".

## Turning this into a package

Suggested layout:

```
wetland_drainage/
  routing.py     # pass1, find, pass2 kernels, copied unchanged from the notebook
  run.py         # run(dem_path, wetlands_path, out_dir, prefix, id_field=..., streams_path=None,
                 #     all_touched=False, write_polygons=False) -> dict of outputs; body = notebook cells
  cli.py         # thin argparse wrapper around run()
tests/           # these scripts; replace run_notebook.py with direct calls to run(), wrap checks in pytest
```

Ground rules for the refactor:
- Keep the three kernels' logic identical. The tests compare cell for cell, so any behaviour change shows up.
- Keep the numba constants (`DR`, `DC`, `FLAT`, `OUT`, `WET`, `NODATA`, `UNSET`, `EXTERNAL`) at module level in `routing.py`. Numba freezes globals at compile time.
- Make the tests pass before and after the refactor.

Prompt to start the session with:

> Read HANDOFF.md. Run `bash tests/run_all.sh` to confirm everything passes. Then turn `wetland_drainage_areas.ipynb` into the package layout in HANDOFF.md, converting the tests to pytest that call `run()` directly, and keep all checks passing.
