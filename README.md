# wetconn: wetland–drainage connectivity

Estimates, for every pixel, the probability that it is a wetland **and**
connects to the drainage network through wetland, by multiplying wetland
probabilities (WIP) along a path to the nearest flowline.

| Method | Path | Status |
| --- | --- | --- |
| 1. `least-cost` | Most probable 8-connected path, ignoring topography | done |
| 2. `flowpath` | D8 steepest-descent flow path | done |
| 3. hybrid | Topography-aware but flexible | to discuss |
| 4. circuit | All paths (Circuitscape-style) | to discuss |

## Setup

```bash
pip install -e ".[test]"
pytest
```

## Usage

```bash
# Method 1
python -m wetconn least-cost --wip proctor_wip.tif --streams proctor_flowlines.gpkg \
    -o proctor_least_cost.tif --ref-length 3 --min-wip 0.005

# Method 2 (the DEM is resampled onto the WIP grid if it differs)
python -m wetconn flowpath --wip proctor_wip.tif --dem proctor_dem.tif \
    --streams proctor_flowlines.gpkg -o proctor_flowpath.tif --ref-length 3 --min-wip 0.005
```

WIP stored as percent (0–100, as in `proctor_wip.tif`) is detected and divided
by 100; `--wip-scale` sets the divisor explicitly. `--min-wip 0.005` is half
the 1% rounding step: a stored 0 means "under 0.5%", not impossible.

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
