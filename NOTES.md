# Project notes and handoff

Working notes for the wetland–drainage connectivity work: what was decided,
why, what the Proctor runs showed, and what is still open. `README.md`
documents how each method works; this file records the reasoning and where
things stand. A new session should read both before changing anything.

## Working agreements

- **Ask before building when inputs are missing or ambiguous.** Inspect the
  data first and check assumptions with Jeff before writing code.
- **Summarise planned changes before making them** when the design isn't
  already settled, and keep things streamlined; don't add options nobody
  asked for.
- Test assumptions about the environment (network access, secrets, tools)
  directly instead of giving setup instructions from memory.

## Data

- **Proctor test basin** (Skykomish, WA), EPSG:6350, 3 m grid, 2366 × 1959.
  - `proctor_wip.tif`, `proctor_dem.tif` on `main` (pushed by Jeff).
  - `proctor_flowlines.gpkg` (layer `reach_sky100a`, 893 NetMap-style
    reaches with `REACH_ID`, `WIDTH_M`, …) is committed on the feature branch
    (it was a chat upload; `*.gpkg` is gitignored, so it was added with -f).
  - WIP is uint8 **percent** (0–100, nodata 255). 37% of valid pixels are 0;
    only 1.9% are ≥ 75. Valid data covers 64% of the rectangle (one
    watershed). Flowline Z matches the DEM (median −0.14 m).
- **Larger basins** are being uploaded to Cloudflare R2:
  - Endpoint `79aea752d97fec8cbfe73e3448211c0d.r2.cloudflarestorage.com`,
    bucket `skykomish` (e.g. `dem_3m.tif`, a ~3 GB COG of HUC8 17110009).
  - Credentials: environment variables `R2_ACCESS_KEY_ID` and
    `R2_SECRET_ACCESS_KEY` (object read **and write**, scoped to the bucket).
    Use path-style addressing and region `auto`.
  - **Only ever write under `outputs/`**; never overwrite or delete inputs.
    R2 keeps no versions.
  - The sandbox's own `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are
    placeholders; don't use them. The proxy's AWS SigV4 secret type only
    signs `*.amazonaws.com` hosts, so it cannot be used for R2.
  - First check in a new session: list the bucket, read a window from a COG
    with rasterio (`/vsis3/` with `AWS_S3_ENDPOINT`, `AWS_VIRTUAL_HOSTING=FALSE`,
    `AWS_REGION=auto`, keys from the `R2_` variables), and write a small test
    file to `outputs/`.
- Inputs should be Cloud-Optimized GeoTIFFs so a basin's window can be read
  without downloading the whole mosaic. Check with
  `rio_cogeo.cogeo.cog_validate`; missing overviews are only a warning and
  don't matter here.

## Compute plan

- Run in this sandbox, **basin by basin** (e.g. HUC10/HUC12 windows read from
  the COGs): 4 cores, ~15 GB RAM, ~28 GB disk.
- Least-cost peaked at ~6 GB for 44 M pixels (~120–140 bytes/pixel), so one
  run fits ~100 M pixels (~900–1,000 km² at 3 m). Switching internal arrays
  to 32-bit (int32 heap indices, float32 costs) would roughly double that;
  not done yet.
- Write results to R2 `outputs/`, since the sandbox disk is wiped between
  sessions.
- Jeff doesn't want to run locally. A DigitalOcean droplet (created per run
  and **destroyed**, not just powered off) is the fallback if a basin is too
  big.

## Decisions and why

### Cost formulation (methods 1–3)
- Cost through a pixel = `−ln(wip) × ℓ / L₀`, where `ℓ` is the path length
  inside the pixel (half the step in plus half out; diagonal = √(dx²+dy²)).
  This was Jeff's request so results adapt to resolution and diagonal travel.
- `p_joint = wip_i × p_conn`: the pixel itself counts once (at-site), and
  the source and stream pixels' own half-steps are excluded from the path.
- Stream pixels = every pixel a flowline touches (`all_touched`).
- `--ref-length` (L₀) defaults to the pixel size; fix it (e.g. 3) to compare
  resolutions. Residual resolution effects remain: averaging WIP to coarser
  pixels smooths dry gaps, so coarse grids look more connected.
- `--min-wip 0.005`: a stored 0 means "under 0.5%", half the rounding step.

### Path threshold (Jeff's choice)
- `--path-threshold 75`: travel uses `min(wip / 0.75, 1)`, so mapped wetland
  (≥ 75) costs nothing to cross, while the **pixel's own term keeps the raw
  WIP**. Goal: de-emphasise stream-adjacent pixels and lift connection further
  out. It worked: the near-stream share of connected area fell 55% → 39%.

### Downstream-only travel
- `--downstream-only` (needs `--dem`): no step toward the stream may climb
  more than `--max-uphill-slope` × step length (default 1%), a slope so it
  holds at any resolution. Level steps are allowed.
- Depressions: `fill` (default) reuses the Priority-Flood + ε fill. The ε tilt
  (~1e-14 m) is far below the tolerance, so filled ponds are freely
  traversable. `sink` keeps pits; shallow noise pits are still escapable
  within the tolerance.
- Caveat: the tolerance is per step, so valleys flatter than 1% allow
  indefinite upstream travel.

### All-paths (method 3, Monte Carlo)
- Jeff wanted the extra probability from multiple routes or broad corridors.
  It is percolation-style: in each random map a pixel is passable with
  probability `min(wip/T, 1)`, and `p_conn` is the fraction of maps where a
  neighbour links to a stream.
- **Big caveat:** independent pixels have a percolation tipping point (~41%
  passable, i.e. WIP ≈ 0.30 with T = 0.75). Above it, broad areas connect at
  any distance. Correlated random maps would fix this; not done.

### Threshold method (method 4) and multi-scale
- WIP is read as a wetting threshold. Each pixel gets the highest cutoff t at
  which it and some route to a stream are all ≥ t (a max-min flood, exact for
  all cutoffs). `route_threshold` leaves the pixel itself out. Ties for
  `reach_id` go to the shortest route.
- Multi-scale (Jeff's spec): for windows 1, 3, …, 17 (3–51 m), judge routes
  on the focal mean (nodata excluded) while the **pixel keeps its raw WIP**:
  `threshold_k = min(wip, route on the k×k mean)`. The final value is the max
  over windows, and `window` is the smallest window achieving it.

### Not pursued (yet)
- Width buffer from `WIDTH_M` for stream pixels: designed (buffer by
  WIDTH_M/2 and rasterize with all_touched; never combine with
  `--centre-only`), +45% stream pixels and +5% connected area on Proctor.
  Jeff said skip for now and maybe revisit.
- A separate "hybrid" method: covered by downstream-only least-cost.

## Proctor results (3 m, `--min-wip 0.005`)

Expected wetland area (Σ wip × area) = 256.1 ha.

| Run | Connected ha |
| --- | --- |
| Least-cost, ÷100, any direction | 66.7 |
| Least-cost, T75, any direction | 93.3 |
| Least-cost, T75, downstream-only | 90.2 |
| All-paths, T75, any direction (500 maps) | 146.0 |
| All-paths, T75, downstream-only | 114.1 |
| Flow path (method 2), ÷100 | 41.7 |

Threshold method, connected area at each cutoff:

| Cutoff | Mapped | Any direction | Downstream | Multi-scale best (any) |
| --- | --- | --- | --- | --- |
| 75 | 50.7 | 46.1 | 45.0 | 46.1 |
| 50 | 153.5 | 118.3 | 108.2 | 118.5 |
| 25 | 361.7 | 269.8 | 237.9 | 271.2 |

- **Multi-scale with focal means barely helps.** Each larger window alone
  connects less than the raw scale (e.g. 35.2 ha at 75 for window 17), because
  averaging dilutes narrow wet strips along channels as well as bridging gaps.
- **Reach 57655** (headwater; the wet bench above its channel head was
  Jeff's test area):
  - About 411 of 640 pixels within 60 m above the head drain by D8 to reach
    57659 (~400 m north). The bench slopes away from 57655's head.
  - The 70 mapped (≥ 75) pixels above the head connect at 65 (threshold
    method). Bridging gaps of ≤ 2 pixels would connect all 70 at 75; focal
    means make them worse (49 at window 17).

## Skykomish HUC8 run (17110009)

- `scripts/huc12_threshold.py`: threshold method, downstream-only (1% uphill
  tolerance, depressions filled), WIP = `wip_gw.tif` (÷100), DEM =
  `dem_3m.tif`, streams = `reaches_100.parquet` (`reach_id`, rasterized
  all_touched), per HUC12 with a 50 m buffer; each pixel is taken from the
  HUC12 containing its centre.
- HUC12s from the USGS WBD (`prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GPKG/WBD_17_HU2_GPKG.zip`,
  layer `WBDHU12`); 21 units, 52–151 km². The TNM API and
  hydro.nationalmap.gov are blocked here; the S3 file server is not.
- Output: R2 `skykomish/outputs/skykomish_17110009_threshold_downstream_wipgw.tif`
  (COG, 23482 × 21950, bands threshold / route_threshold / reach_id, 0.94 GB).
  Took 27 min, peak ~2–4 GB per HUC12.
- 215,648 ha valid; 0.25% of pixels have no allowed route. Connected vs
  mapped wetland: 9,758 of 10,540 ha at 75 (93%), 17,137 of 18,429 ha at 50
  (93%), 28,278 of 30,650 ha at 25 (92%).
- Same run with `wip_hand.tif` (÷100, no other rescaling):
  `outputs/skykomish_17110009_threshold_downstream_wiphand.tif`, 23.5 min.
  Connected vs mapped: 9,793 of 10,610 ha at 75 (92%), 18,729 of 19,641 ha
  at 50 (95%), 30,385 of 32,006 ha at 25 (95%). Versus `wip_gw`: mean
  threshold difference +0.2 pts, but 9.1% of pixels differ by > 10 pts, and
  only 66% of the area connected at 75 in either run is connected in both
  (gw-only 1,955 ha, hand-only 1,990 ha).
- Whole-basin comparisons need row-strip processing; full float arrays
  exhausted the 15 GB of RAM.
- `wip_smoothed.tif` is not tiled (row strips), so windowed reads are ~10×
  slower; convert to COG before using it.

## Open questions / next steps

1. **Focal closing instead of focal mean** for the multi-scale threshold (max
   then min over the window): fills dry gaps narrower than the window without
   diluting narrow wet strips. Proposed, not built; awaiting Jeff's go-ahead.
2. **Batch runner** for larger basins: basin polygons → read COG windows from
   R2 → run the chosen methods → write to R2 `outputs/`.
3. **32-bit internals** to raise the per-run pixel limit.
4. **Correlated random maps** for all-paths, if that method stays in use.
5. Possibly rerun all-paths without `--path-threshold` to see how much the
   rescale drives it.
