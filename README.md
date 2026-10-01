# wetconn: wetland–drainage connectivity

Estimates, for every pixel, the probability that it is a wetland **and**
connects to the drainage network through wetland, by multiplying wetland
probabilities (WIP) along a path to the nearest flowline.

| Method | Path | Status |
| --- | --- | --- |
| 1. `least-cost` | Most probable 8-connected path, any direction or downstream-only | done |
| 2. `flowpath` | D8 steepest-descent flow path | done |
| 3. `all-paths` | Probability that any wetland route exists (Monte Carlo), any direction or downstream-only | done |
| 4. `threshold` | Highest WIP threshold at which a pixel is wet and connected, any direction or downstream-only | done |

## Setup

```bash
pip install -e ".[test]"
pytest
```

## Usage

```bash
# Method 1, any direction (e.g. habitat connectivity)
python -m wetconn least-cost --wip proctor_wip.tif --streams proctor_flowlines.gpkg \
    -o proctor_least_cost.tif --ref-length 3 --min-wip 0.005 --path-threshold 75

# Method 1, downstream-only (water-borne transport)
python -m wetconn least-cost --wip proctor_wip.tif --dem proctor_dem.tif \
    --streams proctor_flowlines.gpkg -o proctor_least_cost_down.tif \
    --ref-length 3 --min-wip 0.005 --path-threshold 75 --downstream-only

# Method 2 (the DEM is resampled onto the WIP grid if it differs)
python -m wetconn flowpath --wip proctor_wip.tif --dem proctor_dem.tif \
    --streams proctor_flowlines.gpkg -o proctor_flowpath.tif --ref-length 3 --min-wip 0.005
```

WIP stored as percent (0–100, as in `proctor_wip.tif`) is detected and divided
by 100; `--wip-scale` sets the divisor explicitly. `--min-wip 0.005` is half
the 1% rounding step: a stored 0 means "under 0.5%", not impossible.
`--path-threshold 75` is in the raster's units (see below).

```bash
# Method 3: any route (add --dem ... --downstream-only for water-borne transport)
python -m wetconn all-paths --wip proctor_wip.tif --streams proctor_flowlines.gpkg \
    -o proctor_all_paths.tif --min-wip 0.005 --path-threshold 75 --n-sims 500 --seed 0
```

```bash
# Method 4: WIP as a wetting threshold (add --dem ... --downstream-only as above)
python -m wetconn threshold --wip proctor_wip.tif --streams proctor_flowlines.gpkg \
    -o proctor_threshold.tif
```

Or from Python:

```python
from wetconn import least_cost_connectivity, flowpath_connectivity
r = least_cost_connectivity(wip, streams, dx=3, dy=3, ref_length=3)
r["p_joint"]
```

Both write a float32 GeoTIFF with named bands:

| Band | Meaning |
| --- | --- |
| `p_joint` | P(pixel is wetland **and** connected) = `wip × p_conn` |
| `p_conn` | P(connected \| pixel is wetland) |
| `cost` | `−ln(p_conn)`, the minimised path cost |
| `path_length` | Length of the chosen path (m) |
| `reach_id` | `REACH_ID` of the flowline reached (use `--id-field` for another attribute) |
| `status` | (flowpath only) 1 reaches stream, 2 leaves domain, 3 ends in a pit |
| `p_conn_se` | (all-paths only) Monte Carlo standard error of `p_conn` |

All-paths writes only `p_joint`, `p_conn` and `p_conn_se`. Threshold writes
`threshold`, `route_threshold` and `reach_id` (see method 4 below).

The CLI also prints the expected wetland area (Σ wip × pixel area) and the
expected *connected* wetland area (Σ p_joint × pixel area).

## How the cost is defined

### Length-weighted −ln(wip)

A path from pixel *i* to the network multiplies the probabilities of the
pixels it crosses, so its log-probability is a sum of `−ln(wip)`. To make
that sum independent of pixel size and step direction, each pixel's term is
weighted by the **length of path inside it** relative to a reference length
`L₀`:

```
cost(i) = Σ_k  −ln(wip_k) · ℓ_k / L₀          (k = pixels between i and the stream)
p_conn  = exp(−cost)
p_joint = wip_i · p_conn
```

- `ℓ_k` is half the step into pixel *k* plus half the step out of it. A step
  is `dx` or `dy` for orthogonal moves and `√(dx² + dy²)` for diagonals, so
  rectangular pixels work too.
- `L₀` (`--ref-length`) is the distance over which a pixel's WIP applies
  exactly once. A stretch of length `ℓ` through ground with probability `p`
  contributes `p^(ℓ/L₀)`. Think of `L₀` as the length over which wetland
  presence is independent from one place to the next.
- With `L₀` equal to the pixel size, an orthogonal path reduces to the plain
  product of `wip` over its pixels. That is the default.
- **To compare rasters of different resolution, fix `L₀`** (for example
  `--ref-length 3` for a 3 m native WIP). Then 1 m and 3 m grids charge the
  same cost for the same physical distance.

### Path threshold

`--path-threshold T` (for example the wetland mapping threshold, 75%)
rescales the probability used **for travel only** to `min(wip / T, 1)`.
Pixels at or above `T` cost nothing to cross, and pixels below it are
charged by how far they fall short. The at-site term keeps the raw WIP, so
`p_joint = wip_i · p_conn` is unchanged for pixels next to a stream and rises
for pixels further out. `--min-wip` is applied before the rescaling.

In the implementation, the edges between pixel centres are weighted
`½(λ_a + λ_b)·d` with `λ = −ln(wip)/L₀` (the trapezoid rule, the same as
ArcGIS Cost Distance and `skimage.graph.MCP_Geometric`). Afterwards the
source pixel's own half-step is removed, because the source is already
covered by the at-site `wip_i` term.

### Boundary conventions

- **Stream pixels** are every pixel a flowline touches (`all_touched`;
  `--centre-only` turns that off). They have `p_conn = 1` and
  `p_joint = wip`. A pixel next to a stream only needs to be wetland itself.
- **`wip = 0`** pixels are barriers. They still get a `p_conn` (what their
  connection would be if they were wetland) but `p_joint = 0`. `--min-wip`
  instead makes them passable at high cost.
- **Nodata** in WIP is impassable and NaN in the output. Stream pixels
  connect even where WIP is nodata.

### Downstream-only travel (method 1)

`--downstream-only` (needs `--dem`) forbids any step toward the stream that
climbs more than `--max-uphill-slope` × step length (default 0.01 = 1%).
Expressing the tolerance as a slope keeps it the same at any resolution
(3 cm per 3 m step, 1 cm per 1 m step). Level and downhill steps are always
allowed. Without the flag, paths may go in any direction.

- `--depressions fill` (default) uses the same filled DEM as method 2, so
  closed depressions drain through their spill point. The ε tilt from the
  fill (~10⁻¹⁴ m per cell) is far below the tolerance, so movement across a
  filled pond is unrestricted. `--depressions sink` keeps pits; a pixel in
  one is connected only if it can climb out within the slope tolerance, so
  shallow noise pits stay connected and steep-walled depressions do not.
- The tolerance applies per step, not in total: in valleys flatter than the
  tolerance a path can travel up the valley floor indefinitely.
- Pixels with no allowed path get `p_conn = 0`.
- Costs are ordered: any-direction ≤ downstream-only ≤ method 2 (the D8
  path is downhill on the same filled DEM). The tests check this.

### Method 3: all paths (Monte Carlo)

Methods 1–2 score a single route. Method 3 asks whether *any* wetland route
exists, so several routes, or a wide corridor, raise the probability: two
separate routes that each exist half the time give 0.75.

- Each of `--n-sims` random maps draws one number per pixel. A pixel is
  passable if the number is below `min(wip / T, 1)` (with `--min-wip` applied
  first); stream pixels always are.
- A pixel is connected in a map if it is a stream pixel or one of its 8
  neighbours is joined to a stream by passable pixels (with the same
  per-step slope rule under `--downstream-only`). That is what would happen
  if the pixel were wetland, so `p_conn` is the fraction of maps where it
  holds, and `p_joint = wip × p_conn` exactly.
- The standard error is at most `0.5/√n` (±0.022 for 500 maps). `--seed`
  makes runs repeatable. A map's random numbers are drawn only where the
  search from the streams reaches, so a run takes ~20 s on Proctor.
- Pixels are independent, which matches method 1 with `ref_length` equal to
  the pixel size; `--ref-length` does not apply. A diagonal step crosses one
  pixel, like an orthogonal one.
- `p_conn` is never below method 1's (tests check this, within the noise).

**Percolation caveat.** With independent pixels, connection has a tipping
point: once more than ~41% of pixels are passable (8-neighbour site
percolation), passable clusters span arbitrarily far. With `T = 0.75` that is
WIP ≈ 0.30. On a uniform field, WIP 0.30 connects with probability ≈ 0.6 at
90 m and WIP 0.35 with ≈ 0.97 at 300 m, where method 1 gives ~10⁻¹⁰ and
below. Real WIP is spatially correlated (a 40% patch tends to be wet or dry
as a whole), so independent maps likely overstate connection through broad
moderate-WIP areas. Correlated random maps would address this.

### Method 4: wetting threshold

WIP is read as the condition needed to wet a pixel: at threshold `t` every
pixel with `wip ≥ t` is wet. Stepping `t` down wets more of the landscape,
and a pixel connects once it and some route of wet pixels to a stream are
all at or above `t`.

- `threshold` is the highest `t` at which the pixel is wet and connected:
  the lowest WIP on its best route, including the pixel itself. The
  connected wetland map at any cutoff is simply `threshold ≥ t`, e.g.
  `threshold ≥ 0.75` for the 75% mapping threshold.
- `route_threshold` leaves the pixel itself out: the cutoff at which it
  would connect if it were wet (1 next to a stream). Where `threshold` is
  below the pixel's own WIP, the route is the limit.
- `reach_id` is the stream the best route reaches; among equally good
  routes, the shortest.
- One priority flood from the streams gives the exact answer for every
  cutoff at once; the tests check it against building a binary map at each
  WIP level. Values are 0–1 like the other outputs.
- `--downstream-only` uses the same per-step slope rule and filled DEM as
  methods 1 and 3. Pixels with no allowed route at any cutoff get `-1`.
- `--ref-length`, `--path-threshold` and `--min-wip` don't apply.

### Method 2 specifics

- Steepest descent uses slope = drop / true step length, so diagonals are
  compared fairly and rectangular pixels are handled.
- `--depressions fill` (default) fills closed depressions to their spill
  point with Priority-Flood + ε (Barnes et al., 2014), so the flow path
  crosses the depression (fill-and-spill). `--depressions sink` keeps pits;
  pixels draining into one get `p_conn = 0`, `status = 3`.
- Flow paths that leave the raster or enter WIP/DEM nodata before reaching a
  stream get `p_conn = NaN`, `status = 2`, because the outcome is unknown.
- `--burn-depth` lowers stream pixels before routing, for when flowlines and
  DEM valleys don't line up.
- Method 1's cost is never higher than method 2's, because the D8 path is
  one of the paths method 1 minimises over. The tests check this.

## Caveats

- **Resolution.** A fixed `L₀` removes the effect of pixel count, but results
  still depend somewhat on resolution. Averaging WIP to coarser pixels
  smooths out dry gaps a fine grid would resolve. The source and stream
  pixels are also wider, so more ground counts as "already there". On a
  synthetic field averaged from 1 m to 6 m (with `L₀ = 3 m`), expected
  connected area roughly doubled, even ≥30 m from streams.
- **8-connectivity** overestimates path length by up to 8% for directions
  between grid axes and diagonals (22.5°).

## Performance

Tested on a 5.9 × 7.4 km extent around the Proctor flowlines, using
synthetic WIP and DEM:

| Grid | Pixels | least-cost | flowpath | Peak RAM |
| --- | --- | --- | --- | --- |
| 3 m | 4.9 M | ~9 s | ~9 s | < 1 GB |
| 1 m | 44 M | ~2 min | ~1.5 min | ~6 GB |
