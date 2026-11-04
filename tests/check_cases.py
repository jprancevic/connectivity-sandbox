"""Spot checks on the planted flat cases and the value scheme.

    check_cases(DATA_DIR, OUT_PREFIX) -> bool
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def check_cases(data, pre):
    """Returns True if every spot check passes."""
    NODATA = -2147483648
    cases = np.load(f"{data}/cases.npz")
    P1 = rasterio.open(f"{pre}_pass1.tif").read(1)
    L = rasterio.open(f"{pre}_labels.tif").read(1)
    summary = pd.read_csv(f"{pre}_summary.csv")
    pits = pd.read_csv(f"{pre}_pits.csv")
    ids = gpd.read_file(f"{data}/wetlands.gpkg")["Wetlands ID"]
    fails = []

    def check(ok, msg):
        print(("ok    " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    lake, bowl, around = cases["lake"], cases["bowl"], cases["around"]
    check(not (P1[lake] < 0).any(), "open flat lake contains no pit cells")
    bowl_pits = np.unique(-P1[bowl][P1[bowl] < 0])
    check(len(bowl_pits) == 1 and pits.set_index("pit_id").loc[bowl_pits[0], "floor_cells"] == bowl.sum(),
          "closed flat bowl is exactly one pit covering the whole bowl")
    check((P1[around] == int(cases["around_wetland_id"]) + 1).all(), "flat bowl around a wetland drains into it")
    valid = L != NODATA
    check(L[valid].min() >= 0 and L[valid].max() <= ids.max() + 1, "final labels are 0 or wetland ID + 1")
    check((L == 1).sum() == summary.loc[summary["Wetlands ID"] == 0, "drainage_cells"].item(),
          "wetland ID 0 is stored as 1 and its cell count matches the summary")
    check(summary.drainage_cells.sum() + (L == 0).sum() == valid.sum(), "summary + label 0 account for every cell")
    return not fails
