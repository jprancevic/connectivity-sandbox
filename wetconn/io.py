"""Raster/vector input and output."""

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject


def read_raster(path):
    """Read band 1 as float64 with NaN for nodata.

    Returns ``(data, valid, profile)``.
    """
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        profile = src.profile.copy()
    data = np.ma.filled(arr.astype(np.float64), np.nan)
    valid = ~np.ma.getmaskarray(arr) & np.isfinite(data)
    return data, valid, profile


def read_aligned(path, profile, resampling="bilinear"):
    """Read band 1 of ``path`` resampled onto the grid described by ``profile``.

    Returns ``(data, valid)``. Rasters already on that grid are read as-is.
    """
    with rasterio.open(path) as src:
        same_grid = (
            src.crs == profile["crs"]
            and src.transform == profile["transform"]
            and src.shape == (profile["height"], profile["width"])
        )
    if same_grid:
        data, valid, _ = read_raster(path)
        return data, valid
    dst = np.full((profile["height"], profile["width"]), np.nan, np.float64)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling[resampling],
        )
    return dst, np.isfinite(dst)


def rasterize_streams(path, profile, id_field=None, layer=None, all_touched=True):
    """Burn flowlines onto the raster grid.

    ``all_touched=True`` marks every pixel a line passes through, which keeps
    the network 8-connected and avoids flow paths slipping past it
    diagonally.

    Returns ``(streams, labels)``: a bool mask and an int64 array holding
    ``id_field`` (default ``REACH_ID`` if present, else 1-based row number)
    on stream pixels and -1 elsewhere.
    """
    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS.")
    gdf = gdf.to_crs(profile["crs"])
    if id_field is None and "REACH_ID" in gdf.columns:
        id_field = "REACH_ID"
    ids = (
        gdf[id_field].to_numpy(np.int64)
        if id_field
        else np.arange(1, len(gdf) + 1, dtype=np.int64)
    )
    if ids.size and (ids.min() < 0 or ids.max() >= 2**31):
        raise ValueError(f"{id_field} values must fit in a non-negative int32.")
    geoms = shapely.force_2d(gdf.geometry.to_numpy())
    kw = dict(
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        all_touched=all_touched,
    )
    labels = rasterize(zip(geoms, ids.astype(np.int32)), fill=-1, dtype="int32", **kw)
    streams = rasterize(((g, 1) for g in geoms), fill=0, dtype="uint8", **kw).astype(bool)
    return streams, labels.astype(np.int64)


def write_bands(path, profile, bands, tags=None):
    """Write a dict of 2-D arrays as a float32 multi-band GeoTIFF.

    Band descriptions are set to the dict keys so they show up in QGIS/GDAL.
    """
    out = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": len(bands),
        "dtype": "float32",
        "crs": profile["crs"],
        "transform": profile["transform"],
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **out) as dst:
        for i, (name, arr) in enumerate(bands.items(), start=1):
            dst.write(np.asarray(arr, dtype=np.float32), i)
            dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
