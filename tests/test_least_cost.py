import numpy as np
import pytest
from skimage.graph import MCP_Geometric

from wetconn import least_cost_connectivity
from wetconn.grid import OFFSETS, step_lengths
from wetconn.least_cost import _dijkstra


def _column_stream(shape):
    s = np.zeros(shape, bool)
    s[:, 0] = True
    return s


def test_orthogonal_path_is_plain_product():
    # With ref_length = pixel size, a straight path multiplies wip once per
    # pixel: the source pixel (at-site) and each pixel between it and the stream.
    p = 0.6
    r = least_cost_connectivity(np.full((3, 8), p), _column_stream((3, 8)), 2.0, 2.0)
    k = np.arange(8)
    np.testing.assert_allclose(r["p_conn"][1], np.r_[1.0, p ** np.maximum(k[1:] - 1, 0)])
    np.testing.assert_allclose(r["p_joint"][1], p * r["p_conn"][1])
    np.testing.assert_allclose(r["path_length"][1], 2.0 * k)


def test_diagonal_steps_cost_sqrt2():
    p, n = 0.7, 9
    streams = np.zeros((n, n), bool)
    streams[0, 0] = True
    r = least_cost_connectivity(np.full((n, n), p), streams, 1.0, 1.0)
    c = -np.log(p)
    k = np.arange(1, n)
    np.testing.assert_allclose(r["cost"][k, k], c * np.sqrt(2) * (k - 1))
    np.testing.assert_allclose(r["cost"][0, k], c * (k - 1))
    np.testing.assert_allclose(r["path_length"][k, k], np.sqrt(2) * k)


@pytest.mark.parametrize("dx", [4.0, 2.0, 1.0, 0.5])
def test_fixed_ref_length_is_resolution_independent(dx):
    # Uniform field, stream along x = 0. The path between the source and
    # stream pixel centres, minus the half pixel at each end, has length
    # D - dx, so cost = lambda * (D - dx) at every resolution.
    p, ref, D = 0.8, 3.0, 24.0
    n = int(D / dx) + 1
    r = least_cost_connectivity(
        np.full((3, n), p), _column_stream((3, n)), dx, dx, ref_length=ref
    )
    lam = -np.log(p) / ref
    np.testing.assert_allclose(r["cost"][1, -1], lam * (D - dx))


def test_rectangular_pixels():
    p, dx, dy = 0.5, 3.0, 1.0
    streams = np.zeros((7, 7), bool)
    streams[0, :] = True
    r = least_cost_connectivity(np.full((7, 7), p), streams, dx, dy, ref_length=1.0)
    lam = -np.log(p)
    # Moving north: dy per pixel.
    np.testing.assert_allclose(r["cost"][:, 3], lam * dy * np.maximum(np.arange(7) - 1, 0))


def test_dijkstra_matches_skimage_mcp_geometric():
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.05, 3.0, (40, 55))
    targets = np.zeros(lam.shape, bool)
    targets[rng.integers(0, 40, 6), rng.integers(0, 55, 6)] = True
    lam[targets] = 0.0
    dx, dy = 2.0, 1.5
    z = np.zeros(lam.shape)
    cost, _, _ = _dijkstra(
        lam, targets, np.zeros(lam.shape, np.int64), z, 0.0, OFFSETS, step_lengths(dx, dy)
    )
    mcp = MCP_Geometric(lam, sampling=(dy, dx), fully_connected=True)
    expected, _ = mcp.find_costs(list(zip(*np.nonzero(targets))))
    np.testing.assert_allclose(cost, expected, rtol=1e-12)


def test_prefers_longer_wetter_route():
    # A dry band blocks the direct route; a wet corridor goes around it.
    wip = np.full((9, 12), 0.95)
    wip[1:8, 5] = 0.01
    streams = _column_stream(wip.shape)
    r = least_cost_connectivity(wip, streams, 1.0, 1.0)
    direct = wip[4, 5] * wip[4, 1:5].prod() * wip[4, 6:10].prod()
    assert r["p_conn"][4, 10] > 10 * direct
    assert r["path_length"][4, 10] > 10.0


def test_zero_wip_is_barrier_but_still_scored():
    wip = np.full((5, 6), 0.9)
    wip[:, 3] = 0.0  # full-height wall
    r = least_cost_connectivity(wip, _column_stream(wip.shape), 1.0, 1.0)
    assert np.all(r["p_conn"][:, 4:] == 0.0)
    # Pixels on the wall can't be wetland, but would connect if they were.
    np.testing.assert_allclose(r["p_conn"][:, 3], 0.9**2)
    assert np.all(r["p_joint"][:, 3] == 0.0)


def test_nodata_and_stream_pixels():
    wip = np.full((4, 5), 0.5)
    wip[2, 3] = np.nan
    wip[0, 0] = np.nan  # a stream pixel without wip still connects
    r = least_cost_connectivity(wip, _column_stream(wip.shape), 1.0, 1.0)
    assert np.isnan(r["p_joint"][2, 3]) and np.isnan(r["p_conn"][2, 3])
    assert r["p_conn"][0, 0] == 1.0 and np.isnan(r["p_joint"][0, 0])
    assert r["p_conn"][0, 1] == 1.0
    np.testing.assert_allclose(r["p_joint"][:, 0][1:], 0.5)


def test_reach_id_is_nearest_labelled_stream():
    wip = np.full((3, 11), 0.5)
    streams = np.zeros(wip.shape, bool)
    streams[:, 0] = streams[:, -1] = True
    labels = np.full(wip.shape, -1)
    labels[:, 0], labels[:, -1] = 7, 9
    r = least_cost_connectivity(wip, streams, 1.0, 1.0, labels=labels)
    assert np.all(r["reach_id"][:, :5] == 7) and np.all(r["reach_id"][:, 6:] == 9)


def test_rejects_non_probabilities():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        least_cost_connectivity(np.full((3, 3), 50.0), _column_stream((3, 3)), 1, 1)


def test_path_threshold_rescales_travel_only():
    p, t = 0.6, 0.75
    r = least_cost_connectivity(
        np.full((3, 8), p), _column_stream((3, 8)), 1.0, 1.0, path_threshold=t
    )
    k = np.arange(2, 8)
    np.testing.assert_allclose(r["p_conn"][1, 2:], (p / t) ** (k - 1))
    np.testing.assert_allclose(r["p_joint"][1], p * r["p_conn"][1])  # raw wip at-site


def test_pixels_above_path_threshold_cost_nothing():
    wip = np.full((5, 30), 0.1)
    wip[2, :] = 0.8  # a long corridor of mapped wetland
    r = least_cost_connectivity(wip, _column_stream(wip.shape), 1.0, 1.0, path_threshold=0.75)
    np.testing.assert_allclose(r["p_conn"][2], 1.0)
    np.testing.assert_allclose(r["p_joint"][2, 1:], 0.8)


def _downstream(wip, dem, dx, **kw):
    return least_cost_connectivity(
        wip, _column_stream(wip.shape), dx, dx, ref_length=3.0, dem=dem,
        downstream_only=True, **kw
    )


def test_downhill_plane_matches_unconstrained():
    wip = np.full((5, 12), 0.7)
    dem = np.tile(0.2 * np.arange(12.0), (5, 1))
    free = least_cost_connectivity(wip, _column_stream(wip.shape), 1.0, 1.0, ref_length=3.0)
    np.testing.assert_allclose(_downstream(wip, dem, 1.0)["cost"], free["cost"])


@pytest.mark.parametrize("dx", [1.0, 3.0])
def test_uphill_tolerance_is_a_slope(dx):
    # Ground rises toward the stream. Below the tolerance every pixel still
    # connects; above it none does. Same outcome at both resolutions.
    n = 10
    x = dx * np.arange(n)
    wip = np.full((3, n), 0.7)
    gentle = np.tile(-0.005 * x, (3, 1))  # 0.5% climb toward the stream
    steep = np.tile(-0.02 * x, (3, 1))  # 2% climb
    assert np.all(_downstream(wip, gentle, dx)["p_conn"][:, 1:] > 0)
    assert np.all(_downstream(wip, steep, dx)["p_conn"][:, 1:] == 0)
    assert np.all(_downstream(wip, steep, dx, max_uphill_slope=0.03)["p_conn"][:, 1:] > 0)


def test_downstream_only_detours_around_a_ridge():
    wip = np.full((9, 10), 0.9)
    dem = np.tile(0.1 * np.arange(10.0), (9, 1))
    dem[1:, 3] += 5.0  # ridge with a gap in row 0
    streams = _column_stream(wip.shape)
    free = least_cost_connectivity(wip, streams, 1.0, 1.0)
    down = least_cost_connectivity(wip, streams, 1.0, 1.0, dem=dem, downstream_only=True)
    assert down["p_conn"][8, 6] > 0
    assert down["path_length"][8, 6] > free["path_length"][8, 6] + 5
    assert down["p_conn"][8, 6] < free["p_conn"][8, 6]


def test_depressions_fill_or_sink():
    wip = np.full((9, 12), 0.9)
    dem = np.tile(0.2 * np.arange(12.0), (9, 1))
    dem[3:6, 5:8] = -5.0  # deep pit with steep walls
    filled = _downstream(wip, dem, 1.0)
    sink = _downstream(wip, dem, 1.0, depressions="sink")
    assert np.all(filled["p_conn"][3:6, 5:8] > 0)
    assert np.all(sink["p_conn"][3:6, 5:8] == 0)
    # A 5 mm dip is climbed out of within the 1% tolerance even when kept.
    dem[3:6, 5:8] = dem[3:6, 4:5] - 0.005
    assert np.all(_downstream(wip, dem, 1.0, depressions="sink")["p_conn"][3:6, 5:8] > 0)


def test_cost_ordering_all_directions_downstream_flowpath():
    # Any-direction paths include all downstream-only paths, which include
    # the D8 steepest-descent path on the same filled DEM.
    from wetconn import flowpath_connectivity

    rng = np.random.default_rng(2)
    shape = (60, 70)
    wip = rng.uniform(0.05, 1.0, shape)
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    dem = 0.2 * cc + np.abs(rr - 30) * 0.1 + rng.normal(0, 0.3, shape)
    streams = _column_stream(shape)
    kw = dict(ref_length=2.0, path_threshold=0.75)
    free = least_cost_connectivity(wip, streams, 2.0, 2.0, **kw)
    down = least_cost_connectivity(
        wip, streams, 2.0, 2.0, dem=dem, downstream_only=True, **kw
    )
    fp = flowpath_connectivity(wip, dem, streams, 2.0, 2.0, **kw)
    ok = fp["status"] == 1
    assert np.all(free["cost"] <= down["cost"] + 1e-9)
    assert np.all(down["cost"][ok] <= fp["cost"][ok] + 1e-9)
    assert np.mean(down["cost"] > free["cost"] + 1e-9) > 0.1  # the rule bites
