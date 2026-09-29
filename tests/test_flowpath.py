import numpy as np

from wetconn import flowpath_connectivity, least_cost_connectivity
from wetconn.flowpath import CLOSED_DEPRESSION, LEAVES_DOMAIN, REACHES_STREAM


def _column_stream(shape):
    s = np.zeros(shape, bool)
    s[:, 0] = True
    return s


def test_plane_draining_to_stream():
    p = 0.5
    shape = (5, 8)
    dem = np.tile(np.arange(8.0), (5, 1))
    r = flowpath_connectivity(np.full(shape, p), dem, _column_stream(shape), 3.0, 3.0)
    assert np.all(r["status"] == REACHES_STREAM)
    k = np.arange(8)
    np.testing.assert_allclose(r["p_conn"][2], p ** np.maximum(k - 1, 0))
    np.testing.assert_allclose(r["path_length"][2], 3.0 * k)


def test_steepest_descent_uses_true_step_length():
    # z = row + col: a diagonal step drops 2 over sqrt(2), steeper than the
    # orthogonal drop of 1 over 1, so flow runs diagonally to the corner.
    n, p = 8, 0.7
    rr, cc = np.mgrid[:n, :n]
    streams = np.zeros((n, n), bool)
    streams[0, 0] = True
    r = flowpath_connectivity(np.full((n, n), p), (rr + cc).astype(float), streams, 1.0, 1.0)
    k = np.arange(1, n)
    np.testing.assert_allclose(r["path_length"][k, k], np.sqrt(2) * k)
    np.testing.assert_allclose(r["cost"][k, k], -np.log(p) * np.sqrt(2) * (k - 1))


def test_rectangular_pixels_change_steepest_direction():
    # Same z = row + col, but rows are 10 m apart: the east-west step (1 m)
    # is now steepest, so flow goes west along the row to the stream column.
    rr, cc = np.mgrid[:6, :6]
    shape = (6, 6)
    r = flowpath_connectivity(
        np.full(shape, 0.5), (rr + cc).astype(float), _column_stream(shape), 1.0, 10.0
    )
    np.testing.assert_allclose(r["path_length"][3], np.arange(6.0))


def _plane_with_pit():
    dem = np.tile(np.arange(12.0), (9, 1))
    dem[3:6, 5:8] = -5.0
    return dem


def test_depressions_sink_disconnects():
    dem = _plane_with_pit()
    shape = dem.shape
    r = flowpath_connectivity(
        np.full(shape, 0.9), dem, _column_stream(shape), 1.0, 1.0, depressions="sink"
    )
    assert np.all(r["status"][3:6, 5:8] == CLOSED_DEPRESSION)
    assert np.all(r["p_conn"][3:6, 5:8] == 0.0)
    assert np.all(r["p_conn"][4, 8:] == 0.0)  # upslope pixels drain into the pit


def test_depressions_fill_spills_to_stream():
    dem = _plane_with_pit()
    shape = dem.shape
    r = flowpath_connectivity(np.full(shape, 0.9), dem, _column_stream(shape), 1.0, 1.0)
    interior = r["status"][1:-1, 1:-1]
    assert np.all(interior == REACHES_STREAM)
    assert np.all(r["p_conn"][3:6, 5:8] > 0)


def test_flow_leaving_domain_is_unknown():
    shape = (4, 6)
    dem = np.tile(np.arange(6.0)[::-1], (4, 1))  # slopes away from the stream
    r = flowpath_connectivity(np.full(shape, 0.5), dem, _column_stream(shape), 1.0, 1.0)
    assert np.all(r["status"][:, 1:] == LEAVES_DOMAIN)
    assert np.all(np.isnan(r["p_conn"][:, 1:]))


def test_flowpath_never_beats_least_cost():
    # The D8 path is one of the 8-connected paths method 1 minimises over.
    rng = np.random.default_rng(1)
    shape = (60, 70)
    wip = rng.uniform(0.05, 1.0, shape)
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    dem = 0.2 * cc + np.abs(rr - 30) * 0.1 + rng.normal(0, 0.3, shape)
    streams = _column_stream(shape)
    fp = flowpath_connectivity(wip, dem, streams, 2.0, 2.0)
    lc = least_cost_connectivity(wip, streams, 2.0, 2.0)
    ok = fp["status"] == REACHES_STREAM
    assert ok.mean() > 0.9
    assert np.all(lc["cost"][ok] <= fp["cost"][ok] + 1e-9)


def test_burn_depth_captures_offset_stream():
    # The valley floor is at col 3 but the mapped stream is at col 4 and
    # stops short of the lower edge; burning pulls flow into it.
    rr, cc = np.mgrid[:10, :9]
    dem = np.abs(cc - 3) * 1.0 + (9 - rr) * 0.5
    streams = np.zeros(dem.shape, bool)
    streams[:8, 4] = True
    wip = np.full(dem.shape, 0.9)
    plain = flowpath_connectivity(wip, dem, streams, 1.0, 1.0)
    burned = flowpath_connectivity(wip, dem, streams, 1.0, 1.0, burn_depth=5.0)
    assert np.any(plain["status"][:, :3] == LEAVES_DOMAIN)
    assert np.sum(burned["status"] == REACHES_STREAM) > np.sum(plain["status"] == REACHES_STREAM)
