"""Run the threshold method HUC12 by HUC12 and mosaic the results.

Each HUC12 is processed on its own window, buffered so routes near its edge
see the ground just beyond it. Pixels outside the buffered polygon are
treated as outside the domain. Results are kept only for pixels whose centre
lies in the unbuffered HUC12, so neighbouring units never overlap, and are
written into one mosaic that is converted to a COG at the end.

Example (R2 inputs, credentials from R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY):

    python scripts/huc12_threshold.py --huc8 17110009 \\
        --hucs WBD_17_HU2_GPKG.gpkg --wip /vsis3/skykomish/wip_gw.tif \\
        --dem /vsis3/skykomish/dem_3m.tif --reaches reaches_100.parquet \\
        --s3-endpoint <account>.r2.cloudflarestorage.com --out skykomish.tif
"""

import argparse
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window, from_bounds

from wetconn.grid import pixel_size
from wetconn.threshold import NEVER, threshold_connectivity

BANDS = ("threshold", "route_threshold", "reach_id")


def configure_r2(endpoint):
    """Point GDAL's /vsis3/ at R2 using the R2_ credential variables."""
    os.environ.update(
        AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"],
        AWS_S3_ENDPOINT=endpoint,
        AWS_REGION="auto",
        AWS_VIRTUAL_HOSTING="FALSE",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    )


def grid_window(bounds, transform, shape):
    """Whole-pixel window covering ``bounds``, clipped to the raster."""
    w = from_bounds(*bounds, transform=transform)
    w = w.round_offsets(op="floor").round_lengths(op="ceil")
    return w.intersection(Window(0, 0, shape[1], shape[0]))


def read_window(src, window):
    arr = src.read(1, window=window, masked=True)
    data = np.ma.filled(arr.astype(np.float64), np.nan)
    return data, ~np.ma.getmaskarray(arr) & np.isfinite(data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--huc8", required=True, help="HUC8 code; its HUC12s are processed.")
    p.add_argument("--hucs", required=True, help="WBD file with a WBDHU12 layer.")
    p.add_argument("--wip", required=True)
    p.add_argument("--dem", required=True)
    p.add_argument("--reaches", required=True, help="Flowlines (GeoParquet or any vector file).")
    p.add_argument("--id-field", default="reach_id")
    p.add_argument("--buffer", type=float, default=50.0, help="Buffer around each HUC12 (m).")
    p.add_argument("--wip-scale", type=float, default=100.0)
    p.add_argument("--max-uphill-slope", type=float, default=0.01)
    p.add_argument("--any-direction", action="store_true", help="Drop the downstream-only rule.")
    p.add_argument("--s3-endpoint", default=None, help="R2 endpoint host for /vsis3/ paths.")
    p.add_argument("--out", required=True, help="Output COG.")
    args = p.parse_args(argv)

    if args.s3_endpoint:
        configure_r2(args.s3_endpoint)
    out = Path(args.out)
    work = out.with_suffix(".work.tif")
    t_all = time.perf_counter()

    wip_src = rasterio.open(args.wip)
    dem_src = rasterio.open(args.dem)
    if (wip_src.crs, wip_src.transform, wip_src.shape) != (dem_src.crs, dem_src.transform, dem_src.shape):
        raise SystemExit("wip and dem must share a grid.")
    crs, transform, shape = wip_src.crs, wip_src.transform, wip_src.shape
    dx, dy = pixel_size(transform)

    hucs = gpd.read_file(args.hucs, layer="WBDHU12", columns=["huc12", "name"])
    hucs = hucs[hucs["huc12"].str.startswith(args.huc8)].to_crs(crs)
    hucs["geometry"] = shapely.force_2d(hucs.geometry.make_valid())
    hucs = hucs.sort_values("huc12").reset_index(drop=True)
    print(f"{len(hucs)} HUC12s in HUC8 {args.huc8}")

    reaches = (
        gpd.read_parquet(args.reaches)
        if args.reaches.endswith(".parquet")
        else gpd.read_file(args.reaches)
    ).to_crs(crs)
    reaches["geometry"] = shapely.force_2d(reaches.geometry)
    sindex = reaches.sindex

    buffered = hucs.geometry.buffer(args.buffer)
    full = grid_window(buffered.total_bounds, transform, shape)
    print(f"HUC8 mosaic {full.width} x {full.height} pixels")

    profile = dict(
        driver="GTiff", width=full.width, height=full.height, count=len(BANDS),
        dtype="float32", crs=crs, transform=rasterio.windows.transform(full, transform),
        nodata=np.nan, tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES",
    )
    with rasterio.open(work, "w", **profile) as dst:
        for i, name in enumerate(BANDS, 1):
            dst.set_band_description(i, name)

    totals = dict(pixels=0, never=0, at75=0)
    with rasterio.open(work, "r+") as dst:
        for k, huc in hucs.iterrows():
            t0 = time.perf_counter()
            w = grid_window(buffered.iloc[k].bounds, transform, shape)
            wt = rasterio.windows.transform(w, transform)
            kw = dict(out_shape=(w.height, w.width), transform=wt)

            domain = rasterize([(buffered.iloc[k], 1)], fill=0, all_touched=True, dtype="uint8", **kw).astype(bool)
            core = rasterize([(huc.geometry, 1)], fill=0, all_touched=False, dtype="uint8", **kw).astype(bool)

            wip, wip_valid = read_window(wip_src, w)
            dem, dem_valid = read_window(dem_src, w)
            wip_valid &= domain
            dem_valid &= domain

            idx = sindex.query(buffered.iloc[k], predicate="intersects")
            sub = reaches.iloc[idx]
            shapes = list(zip(sub.geometry, sub[args.id_field].astype(np.int64)))
            labels = (
                rasterize(shapes, fill=-1, all_touched=True, dtype="int32", **kw).astype(np.int64)
                if shapes else np.full((w.height, w.width), -1, np.int64)
            )
            streams = (labels >= 0) & domain

            r = threshold_connectivity(
                wip / args.wip_scale, streams, dx, dy, wip_valid, labels, dem, dem_valid,
                downstream_only=not args.any_direction,
                max_uphill_slope=args.max_uphill_slope, depressions="fill",
            )

            # Merge this HUC12's own pixels into the mosaic.
            mw = Window(w.col_off - full.col_off, w.row_off - full.row_off, w.width, w.height)
            block = dst.read(window=mw)
            for i, name in enumerate(BANDS):
                block[i][core] = r[name][core]
            dst.write(block, window=mw)

            t = r["threshold"][core & wip_valid]
            n, never, at75 = t.size, int(np.sum(t == NEVER)), int(np.sum(t >= 0.75))
            totals["pixels"] += n
            totals["never"] += never
            totals["at75"] += at75
            print(
                f"[{k + 1:2d}/{len(hucs)}] {huc.huc12} {huc['name'][:28]:28s} "
                f"{w.width}x{w.height}, {streams.sum():7d} stream px, "
                f"connected at 75: {at75 * dx * dy / 1e4:7.1f} ha, "
                f"no route: {never / max(n, 1):5.1%}, {time.perf_counter() - t0:5.1f} s",
                flush=True,
            )

    with rasterio.open(work, "r+") as dst:
        dst.update_tags(
            method="threshold", downstream_only=str(not args.any_direction),
            max_uphill_slope=args.max_uphill_slope, depressions="fill",
            huc8=args.huc8, buffer_m=args.buffer, wip=args.wip, dem=args.dem,
            reaches=Path(args.reaches).name, id_field=args.id_field,
        )
    rio_copy(work, out, driver="COG", compress="DEFLATE", predictor=3, BIGTIFF="IF_SAFER",
             num_threads="ALL_CPUS")
    work.unlink()
    area = dx * dy / 1e4
    print(
        f"done in {(time.perf_counter() - t_all) / 60:.1f} min: {totals['pixels'] * area:,.0f} ha valid, "
        f"{totals['at75'] * area:,.1f} ha connected at 75, "
        f"{totals['never'] / max(totals['pixels'], 1):.1%} with no allowed route -> {out}"
    )


if __name__ == "__main__":
    main()
