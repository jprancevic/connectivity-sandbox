"""Command-line interface: ``python -m wetconn {least-cost,flowpath} ...``."""

import argparse
import time

import numpy as np

from . import io
from .flowpath import flowpath_connectivity
from .grid import pixel_size
from .least_cost import least_cost_connectivity


def _common(p):
    p.add_argument("--wip", required=True, help="Wetland probability raster (0-1).")
    p.add_argument("--streams", required=True, help="Flowline vector file.")
    p.add_argument("-o", "--out", required=True, help="Output GeoTIFF.")
    p.add_argument(
        "--ref-length",
        type=float,
        default=None,
        help="Length (map units) over which one pixel's wip applies once. "
        "Default: the pixel size. Fix it (e.g. 3) to compare resolutions.",
    )
    p.add_argument("--layer", default=None, help="Layer name in the streams file.")
    p.add_argument(
        "--id-field",
        default=None,
        help="Flowline id attribute for the reach_id band (default REACH_ID).",
    )
    p.add_argument(
        "--centre-only",
        action="store_true",
        help="Rasterize flowlines through pixel centres only, not every touched pixel.",
    )
    p.add_argument(
        "--min-wip",
        type=float,
        default=0.0,
        help="Floor on wip so zero-probability pixels are passable at high cost "
        "instead of being barriers.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wetconn", description=__doc__)
    sub = parser.add_subparsers(dest="method", required=True)

    lc = sub.add_parser("least-cost", help="Method 1: most-probable path, no topography.")
    _common(lc)

    fp = sub.add_parser("flowpath", help="Method 2: D8 steepest-descent flow path.")
    _common(fp)
    fp.add_argument("--dem", required=True, help="DEM; resampled onto the wip grid if needed.")
    fp.add_argument(
        "--depressions",
        choices=("fill", "sink"),
        default="fill",
        help="fill: route through spill points; sink: pits disconnect (default fill).",
    )
    fp.add_argument(
        "--burn-depth",
        type=float,
        default=0.0,
        help="Lower stream pixels by this many DEM units before routing.",
    )

    args = parser.parse_args(argv)
    t0 = time.perf_counter()

    wip, valid, profile = io.read_raster(args.wip)
    if profile["crs"] is None or profile["crs"].is_geographic:
        parser.error("the wip raster must be in a projected CRS with metre-like units.")
    dx, dy = pixel_size(profile["transform"])
    ref_length = float(np.sqrt(dx * dy)) if args.ref_length is None else args.ref_length
    streams, labels = io.rasterize_streams(
        args.streams,
        profile,
        id_field=args.id_field,
        layer=args.layer,
        all_touched=not args.centre_only,
    )
    print(
        f"grid {wip.shape[0]} x {wip.shape[1]}, pixel {dx:g} x {dy:g}, "
        f"ref_length {ref_length:g}, {streams.sum()} stream pixels"
    )

    tags = {"method": args.method, "ref_length": ref_length, "min_wip": args.min_wip}
    if args.method == "least-cost":
        result = least_cost_connectivity(
            wip, streams, dx, dy, ref_length, valid, labels, args.min_wip
        )
    else:
        dem, dem_valid = io.read_aligned(args.dem, profile)
        result = flowpath_connectivity(
            wip,
            dem,
            streams,
            dx,
            dy,
            ref_length,
            valid,
            dem_valid,
            labels,
            depressions=args.depressions,
            burn_depth=args.burn_depth,
            min_wip=args.min_wip,
        )
        tags.update(depressions=args.depressions, burn_depth=args.burn_depth)
        status = result["status"]
        for code, name in [(1, "reach stream"), (2, "leave domain"), (3, "end in pit")]:
            print(f"  {np.mean(status[valid] == code):6.1%} of pixels {name}")

    io.write_bands(args.out, profile, result, tags)

    area = dx * dy / 1e4
    print(
        f"expected wetland area {np.nansum(wip[valid]) * area:.1f} ha, "
        f"connected {np.nansum(result['p_joint']) * area:.1f} ha"
    )
    print(f"wrote {args.out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
