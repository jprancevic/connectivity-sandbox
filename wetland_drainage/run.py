"""Label every DEM cell with the first wetland its D8 flow path reaches.

Pass 1 routes every cell to a wetland, a stream, the DEM edge or a pit; pass 2 drains the pits from the
lowest outlet up. See HANDOFF.md for the algorithm and the value scheme.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import shape

from .routing import EXTERNAL, NODATA, UNSET, pass1, pass2


def run(dem_path, wetlands_path, out_dir, prefix, id_field="Wetlands ID", streams_path=None,
        all_touched=False, write_polygons=False, wetlands_layer=None):
    """Write <prefix>_pass1.tif, _labels.tif, _pits.csv, _summary.csv (and optionally
    _drainage_areas.gpkg) to out_dir. Returns a dict of output paths.

    id_field must hold non-negative integers. all_touched burns every cell a wetland polygon touches.
    Stream cells (if streams_path is set) become 0, so flow reaching them bypasses wetlands.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"{prefix}_{k}{ext}" for k, ext in
             (("pass1", ".tif"), ("labels", ".tif"), ("pits", ".csv"), ("summary", ".csv"))}

    # --- pass 1 ---
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float64 if src.dtypes[0] == "float64" else np.float32, copy=False)
        if src.nodata is not None and not np.isnan(src.nodata):
            z[z == src.nodata] = np.nan
        transform, crs, (dx, dy) = src.transform, src.crs, src.res
        profile = dict(driver="GTiff", height=src.height, width=src.width, count=1, dtype="int32", crs=crs,
                       transform=transform, nodata=NODATA, tiled=True, compress="deflate", BIGTIFF="IF_SAFER")
    cell_area = dx * dy

    wl = gpd.read_file(wetlands_path, layer=wetlands_layer)
    wl = wl[wl.geometry.notna() & ~wl.geometry.is_empty & wl[id_field].notna()].to_crs(crs)
    wid = wl[id_field].to_numpy()
    if wid.min() < 0 or np.any(wid != np.floor(wid)):
        raise ValueError(f"'{id_field}' must hold non-negative integers")
    wl = wl.loc[wl.area.sort_values(ascending=False).index]      # smaller polygons burn last, so they win overlaps

    lab = np.full(z.shape, UNSET, dtype=np.int32)
    features.rasterize(zip(wl.geometry, wl[id_field].astype(np.int64) + 1), out=lab, transform=transform,
                       all_touched=all_touched)
    if streams_path:
        streams = gpd.read_file(streams_path).to_crs(crs)
        features.rasterize(((g, EXTERNAL) for g in streams.geometry if g is not None), out=lab,
                           transform=transform, all_touched=True)
    lab[np.isnan(z)] = NODATA
    n_bins = int(wid.max()) + 2
    wetland_cells = np.bincount(lab[lab > 0], minlength=n_bins)

    n_pits, pit_cell, pit_floor = pass1(z, lab, dx, dy)

    with rasterio.open(paths["pass1"], "w", **profile) as dst:
        dst.write(lab, 1)

    pass1_cells = np.zeros(n_bins, np.int64)
    pit_cells = np.zeros(n_pits + 1, np.int64)
    for r0 in range(0, lab.shape[0], 1024):
        a = lab[r0:r0 + 1024]
        pass1_cells += np.bincount(a[a >= 0], minlength=n_bins)
        pit_cells += np.bincount(-a[(a < 0) & (a != NODATA)], minlength=n_pits + 1)
    n_valid = pass1_cells.sum() + pit_cells.sum()
    print(f"{n_valid:,} valid cells: {pass1_cells[1:].sum() / n_valid:.1%} reach a wetland, "
          f"{pass1_cells[0] / n_valid:.1%} leave the DEM, {pit_cells.sum() / n_valid:.1%} end in one of "
          f"{n_pits:,} pits")

    # --- pass 2: drain the pits ---
    spill, outlet, drains_to = pass2(z, lab, n_pits, dx, dy)

    with rasterio.open(paths["labels"], "w", **profile) as dst:
        dst.write(lab, 1)

    final_cells = np.zeros(n_bins, np.int64)
    for r0 in range(0, lab.shape[0], 1024):
        a = lab[r0:r0 + 1024]
        final_cells += np.bincount(a[a >= 0], minlength=n_bins)
    print(f"After draining pits: {final_cells[1:].sum() / n_valid:.1%} reach a wetland, "
          f"{final_cells[0] / n_valid:.1%} leave the DEM, {(lab < 0).sum() - (lab == NODATA).sum():,} cells unresolved")

    pit_r, pit_c = np.divmod(pit_cell, lab.shape[1])
    out_r, out_c = np.divmod(outlet, lab.shape[1])
    pits = pd.DataFrame({
        "pit_id": np.arange(1, n_pits + 1), "floor_row": pit_r, "floor_col": pit_c,
        "floor_elevation": z[pit_r, pit_c], "floor_cells": pit_floor, "catchment_cells": pit_cells[1:],
        "outlet_row": out_r, "outlet_col": out_c, "spill_elevation": spill,
        "drains_to": drains_to,            # label value: wetland ID + 1, or 0
    })
    pits.to_csv(paths["pits"], index=False)

    ids = np.unique(wid).astype(np.int64)
    summary = pd.DataFrame({
        id_field: ids, "label_value": ids + 1, "wetland_cells": wetland_cells[ids + 1],
        "drainage_cells_pass1": pass1_cells[ids + 1], "drainage_cells": final_cells[ids + 1],
    })
    summary["drainage_area_m2"] = summary["drainage_cells"] * cell_area
    summary.to_csv(paths["summary"], index=False)

    # --- optional: drainage-area polygons ---
    if write_polygons:
        pieces = []
        for r0 in range(0, lab.shape[0], 4096):
            a = lab[r0:r0 + 4096]
            pieces += [(int(v), shape(g)) for g, v in features.shapes(
                a, mask=a > 0, transform=transform * rasterio.Affine.translation(0, r0))]
        polys = gpd.GeoDataFrame({"label_value": [v for v, _ in pieces]},
                                 geometry=[g for _, g in pieces], crs=crs).dissolve("label_value").reset_index()
        polys = polys.merge(summary, on="label_value")
        paths["drainage_areas"] = out / f"{prefix}_drainage_areas.gpkg"
        polys.to_file(paths["drainage_areas"], driver="GPKG")
    return paths
