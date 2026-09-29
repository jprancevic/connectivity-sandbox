import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from wetconn.cli import main

CRS = "EPSG:6350"
X0, Y0, RES = -1915000.0, 3019000.0, 3.0


def _write(path, arr, res=RES, nodata=None):
    profile = dict(
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


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
