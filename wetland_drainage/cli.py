"""Command line for run().

    python -m wetland_drainage.cli DEM WETLANDS OUT_DIR PREFIX [--streams STREAMS] [--polygons] ...
"""
import argparse

from .run import run


def main(argv=None):
    p = argparse.ArgumentParser(description="Label every DEM cell with the first wetland its D8 path reaches.")
    p.add_argument("dem")
    p.add_argument("wetlands")
    p.add_argument("out_dir")
    p.add_argument("prefix")
    p.add_argument("--layer", default=None, help="Layer name inside the wetlands file.")
    p.add_argument("--id-field", default="Wetlands ID", help="Non-negative integer wetland IDs.")
    p.add_argument("--streams", default=None, help="Flowlines; their cells become 0, bypassing wetlands.")
    p.add_argument("--all-touched", action="store_true", help="Burn every cell a wetland polygon touches.")
    p.add_argument("--polygons", action="store_true", help="Also write dissolved drainage-area polygons (slow).")
    a = p.parse_args(argv)
    for k, v in run(a.dem, a.wetlands, a.out_dir, a.prefix, id_field=a.id_field, streams_path=a.streams,
                    all_touched=a.all_touched, write_polygons=a.polygons, wetlands_layer=a.layer).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
