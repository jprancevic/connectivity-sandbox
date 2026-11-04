"""Wetland drainage areas: run() on a synthetic landscape, checked against independent implementations.

Three cases: no streams, streams, and a DEM rounded to 0.05 m (many exact ties) with streams. Each is
checked cell by cell against an independent pass 1 (sorted-elevation walk) and a literal pit-by-pit
pass 2. The planted flat cases are spot-checked on the no-streams run. About a minute in total.
"""
import geopandas as gpd
import pytest

from check_cases import check_cases
from check_pass1 import check_pass1
from check_pass2 import check_pass2
from make_test_data import make_test_data
from wetland_drainage import run

CASES = {                    # dem file, use streams, write polygons
    "nostreams": ("dem.tif", False, True),
    "streams": ("dem.tif", True, True),
    "rounded": ("dem_rounded.tif", True, False),
}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    make_test_data(d)
    return d


@pytest.fixture(scope="module", params=list(CASES))
def case(request, data, tmp_path_factory):
    dem, use_streams, polygons = CASES[request.param]
    streams = data / "streams.shp" if use_streams else None
    out = tmp_path_factory.mktemp(request.param)
    paths = run(data / dem, data / "wetlands.gpkg", out, "test", streams_path=streams, write_polygons=polygons)
    return dict(name=request.param, dem=data / dem, streams=streams, prefix=out / "test", paths=paths,
                polygons=polygons)


def test_outputs_written(case):
    expected = {"pass1", "labels", "pits", "summary"} | ({"drainage_areas"} if case["polygons"] else set())
    assert set(case["paths"]) == expected
    assert all(p.exists() for p in case["paths"].values())
    if case["polygons"]:
        polys = gpd.read_file(case["paths"]["drainage_areas"])
        assert polys["label_value"].min() >= 1 and polys["label_value"].is_unique


def test_pass1_matches_independent(case, data):
    assert check_pass1(case["dem"], data / "wetlands.gpkg", case["prefix"], streams=case["streams"])


def test_pass2_matches_literal(case):
    assert check_pass2(case["dem"], case["prefix"])


@pytest.mark.parametrize("case", ["nostreams"], indirect=True)
def test_planted_cases(case, data):
    assert check_cases(data, case["prefix"])
