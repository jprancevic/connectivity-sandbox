import geopandas as gpd
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from wetconn.cli import main

CRS = "EPSG:6350"
X0, Y0, RES = -1915000.0, 3019000.0, 3.0


def _write(path, arr, res=RES, nodata=None, dtype="float32"):
    profile = dict(
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=CRS,
        transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def _inputs(tmp_path):
    n = 40
    wip = np.full((n, n), 0.6)
    wip[0, 0] = -9999  # nodata
    _write(tmp_path / "wip.tif", wip, nodata=-9999)
    # DEM at a coarser resolution, sloping west toward the stream.
    dem = np.tile(np.linspace(20, 0, n // 2)[::-1], (n // 2, 1))
    _write(tmp_path / "dem.tif", dem, res=2 * RES)
    x = X0 + 1.5 * RES  # through column 1
    line = LineString([(x, Y0), (x, Y0 - n * RES)])
    gpd.GeoDataFrame({"REACH_ID": [42]}, geometry=[line], crs=CRS).to_file(
        tmp_path / "streams.gpkg"
    )
    return tmp_path


def _read(path):
    with rasterio.open(path) as src:
        return {d: src.read(i + 1) for i, d in enumerate(src.descriptions)}, src.tags()


def test_least_cost_cli(tmp_path):
    d = _inputs(tmp_path)
    out = d / "lc.tif"
    main(
        ["least-cost", "--wip", str(d / "wip.tif"), "--streams", str(d / "streams.gpkg"),
         "-o", str(out), "--ref-length", "3", "--centre-only"]
    )
    bands, tags = _read(out)
    assert list(bands) == ["p_joint", "p_conn", "cost", "path_length", "reach_id"]
    assert tags["ref_length"] == "3.0"
    assert np.isnan(bands["p_joint"][0, 0])
    # Stream is column 1; the pixel next to it only needs to be wetland.
    np.testing.assert_allclose(bands["p_joint"][5, 1:7], 0.6 ** np.r_[1, np.arange(1, 6)], rtol=1e-6)
    assert np.all(bands["reach_id"][np.isfinite(bands["reach_id"])] == 42)


def test_flowpath_cli(tmp_path):
    d = _inputs(tmp_path)
    out = d / "fp.tif"
    main(
        ["flowpath", "--wip", str(d / "wip.tif"), "--dem", str(d / "dem.tif"),
         "--streams", str(d / "streams.gpkg"), "-o", str(out), "--centre-only"]
    )
    bands, tags = _read(out)
    assert "status" in bands and tags["depressions"] == "fill"
    assert np.nanmax(bands["p_conn"]) == 1.0
    assert np.all(bands["status"][2:-2, 2:-2] == 1)


def test_percent_wip_is_autoscaled_and_zeros_floored(tmp_path):
    # uint8 percent, nodata 255, like the Proctor WIP. A column of zeros sits
    # between the stream (column 1) and the rest of the grid.
    d = _inputs(tmp_path)
    wip = np.full((40, 40), 60, np.uint8)
    wip[:, 4] = 0
    wip[0, 0] = 255
    _write(d / "wip_pct.tif", wip, nodata=255, dtype="uint8")
    out = d / "pct.tif"
    main(
        ["least-cost", "--wip", str(d / "wip_pct.tif"), "--streams", str(d / "streams.gpkg"),
         "-o", str(out), "--centre-only", "--min-wip", "0.005"]
    )
    bands, tags = _read(out)
    assert tags["wip_scale"] == "100.0"
    np.testing.assert_allclose(bands["p_joint"][5, 1:4], [0.6, 0.6, 0.36], rtol=1e-6)
    assert bands["p_joint"][5, 4] == 0.0  # zero pixel keeps its own p_joint = 0
    # Beyond the zero column, paths cross it at the floored probability.
    np.testing.assert_allclose(bands["p_conn"][5, 5], 0.6**2 * 0.005, rtol=1e-5)


def test_least_cost_downstream_cli(tmp_path):
    d = _inputs(tmp_path)
    out = d / "lcd.tif"
    base = ["least-cost", "--wip", str(d / "wip.tif"), "--streams", str(d / "streams.gpkg"),
            "-o", str(out), "--centre-only", "--downstream-only"]
    with pytest.raises(SystemExit):
        main(base)  # no --dem
    main(base + ["--dem", str(d / "dem.tif"), "--path-threshold", "0.75"])
    bands, tags = _read(out)
    assert tags["downstream_only"] == "True" and tags["max_uphill_slope"] == "0.01"
    assert tags["path_threshold"] == "0.75"
    # DEM slopes down to the stream, so every pixel connects as before, with
    # travel through 0.6 pixels at 0.6 / 0.75 = 0.8 each.
    np.testing.assert_allclose(bands["p_conn"][5, 2:6], 0.8 ** np.arange(4), rtol=1e-6)
