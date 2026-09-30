"""Command-line interface: ``python -m wetconn {least-cost,flowpath} ...``."""

import argparse
import time

import numpy as np

from . import io
from .all_paths import all_paths_connectivity
from .flowpath import flowpath_connectivity
from .grid import pixel_size
from .least_cost import least_cost_connectivity


def _common(p, ref_length=True):
    p.add_argument("--wip", required=True, help="Wetland probability raster.")
    p.add_argument(
        "--wip-scale",
        type=float,
        default=None,
        help="Divide wip by this to get a 0-1 probability. Default: 100 if the "
        "raster's values run past 1 but not past 100 (percent), else 1.",
    )
    p.add_argument("--streams", required=True, help="Flowline vector file.")
    p.add_argument("-o", "--out", required=True, help="Output GeoTIFF.")
    if ref_length:
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
        "--path-threshold",
        type=float,
        default=None,
        help="WIP (in the raster's units, e.g. 75) at or above which a pixel costs "
        "nothing to cross. Travel uses min(wip / threshold, 1); p_joint still uses "
        "the raw wip. Default: no rescaling.",
    )
    p.add_argument(
        "--min-wip",
        type=float,
        default=0.0,
        help="Floor on wip (after scaling) when crossing a pixel, so zero-probability "
        "pixels are passable at high cost instead of being barriers. A pixel's own "
        "p_joint still uses its unfloored wip.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wetconn", description=__doc__)
    sub = parser.add_subparsers(dest="method", required=True)

    dem_help = "DEM; resampled onto the wip grid if needed."
    depressions_help = "fill: drain through spill points; sink: pits disconnect (default fill)."

    def downstream_args(p):
        p.add_argument("--dem", help=dem_help + " Required with --downstream-only.")
        p.add_argument(
            "--downstream-only",
            action="store_true",
            help="Only allow paths that run downhill or level toward the stream "
            "(water-borne transport). Default: any direction (e.g. habitat).",
        )
        p.add_argument(
            "--max-uphill-slope",
            type=float,
            default=0.01,
            help="With --downstream-only, the steepest climb one step may make "
            "(rise / step length; default 0.01 = 1%%) to tolerate DEM noise.",
        )
        p.add_argument(
            "--depressions", choices=("fill", "sink"), default="fill", help=depressions_help
        )

    lc = sub.add_parser("least-cost", help="Method 1: most-probable path.")
    _common(lc)
    downstream_args(lc)

    fp = sub.add_parser("flowpath", help="Method 2: D8 steepest-descent flow path.")
    _common(fp)
    fp.add_argument("--dem", required=True, help=dem_help)
    fp.add_argument(
        "--depressions", choices=("fill", "sink"), default="fill", help=depressions_help
    )
    fp.add_argument(
        "--burn-depth",
        type=float,
        default=0.0,
        help="Lower stream pixels by this many DEM units before routing.",
    )

    ap = sub.add_parser(
        "all-paths", help="Method 3: probability that any wetland route exists (Monte Carlo)."
    )
    _common(ap, ref_length=False)
    downstream_args(ap)
    ap.add_argument(
        "--n-sims",
        type=int,
        default=500,
        help="Number of random wetland maps (default 500; standard error <= 0.5/sqrt(n)).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")

    args = parser.parse_args(argv)
    if args.method != "flowpath" and args.downstream_only and not args.dem:
        parser.error("--downstream-only needs --dem.")
    t0 = time.perf_counter()

    wip, valid, profile = io.read_raster(args.wip)
    if profile["crs"] is None or profile["crs"].is_geographic:
        parser.error("the wip raster must be in a projected CRS with metre-like units.")
    scale = args.wip_scale
    if scale is None:
        top = np.max(wip[valid]) if valid.any() else 1.0
        scale = 100.0 if 1 < top <= 100 else 1.0
        if scale != 1:
            print(f"wip values run to {top:g}; treating them as percent (--wip-scale 100)")
    wip = wip / scale
    path_threshold = 1.0 if args.path_threshold is None else args.path_threshold / scale
    if not 0 < path_threshold <= 1:
        parser.error(
            f"--path-threshold {args.path_threshold:g} is outside the wip range "
            f"(0, {scale:g}]."
        )
    dx, dy = pixel_size(profile["transform"])
    if args.method == "all-paths":
        ref_length = float(np.sqrt(dx * dy))  # pixels are independent
    elif args.ref_length is None:
        ref_length = float(np.sqrt(dx * dy))
    else:
        ref_length = args.ref_length
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

    tags = {
        "method": args.method,
        "ref_length": ref_length,
        "wip_scale": scale,
        "min_wip": args.min_wip,
        "path_threshold": path_threshold,
    }
    if args.method == "least-cost":
        dem, dem_valid = io.read_aligned(args.dem, profile) if args.dem else (None, None)
        result = least_cost_connectivity(
            wip,
            streams,
            dx,
            dy,
            ref_length,
            valid,
            labels,
            args.min_wip,
            path_threshold,
            dem,
            dem_valid,
            downstream_only=args.downstream_only,
            max_uphill_slope=args.max_uphill_slope,
            depressions=args.depressions,
        )
    elif args.method == "all-paths":
        dem, dem_valid = io.read_aligned(args.dem, profile) if args.dem else (None, None)
        result = all_paths_connectivity(
            wip,
            streams,
            dx,
            dy,
            valid,
            args.min_wip,
            path_threshold,
            dem,
            dem_valid,
            downstream_only=args.downstream_only,
            max_uphill_slope=args.max_uphill_slope,
            depressions=args.depressions,
            n_sims=args.n_sims,
            seed=args.seed,
        )
        tags.update(n_sims=args.n_sims, seed=args.seed)
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
            path_threshold=path_threshold,
        )
        tags.update(depressions=args.depressions, burn_depth=args.burn_depth)
        status = result["status"]
        for code, name in [(1, "reach stream"), (2, "leave domain"), (3, "end in pit")]:
            print(f"  {np.mean(status[valid] == code):6.1%} of pixels {name}")

    if args.method != "flowpath":
        tags.update(downstream_only=args.downstream_only)
        if args.downstream_only:
            tags.update(max_uphill_slope=args.max_uphill_slope, depressions=args.depressions)
    io.write_bands(args.out, profile, result, tags)

    area = dx * dy / 1e4
    print(
        f"expected wetland area {np.nansum(wip[valid]) * area:.1f} ha, "
        f"connected {np.nansum(result['p_joint']) * area:.1f} ha"
    )
    print(f"wrote {args.out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
