from collections import deque

import numpy as np
import pytest

from wetconn.grid import OFFSETS, step_lengths
from wetconn.least_cost import downstream_elevations
from wetconn.threshold import NEVER, threshold_connectivity


def _column_stream(shape):
    s = np.zeros(shape, bool)
    s[:, 0] = True
    return s


def _brute_force(wip, streams, z, max_rise, d=1.0):
    """The procedure as stated: a binary map per WIP level, then a search."""
    lengths = step_lengths(d, d)
    nr, nc = wip.shape
    best = np.full(wip.shape, NEVER)
    for t in sorted(set(wip.ravel()), reverse=True):
        wet = wip >= t
        reached = streams.copy()
        queue = deque(zip(*np.nonzero(streams)))
        while queue:
            r, c = queue.popleft()
            for k, (dr, dc) in enumerate(OFFSETS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < nr and 0 <= cc < nc) or reached[rr, cc] or not wet[rr, cc]:
                    continue
                if z[r, c] - z[rr, cc] > max_rise * lengths[k]:
                    continue
                reached[rr, cc] = True
                queue.append((rr, cc))
        hit = reached & wet & (best == NEVER)
        best[hit] = t
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_binary_maps_any_direction(seed):
    rng = np.random.default_rng(seed)
    wip = rng.integers(0, 21, (15, 18)) / 20  # few distinct levels, many ties
    streams = _column_stream(wip.shape)
    r = threshold_connectivity(wip, streams, 1, 1)
    np.testing.assert_array_equal(r["threshold"], _brute_force(wip, streams, np.zeros(wip.shape), 0.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_binary_maps_downstream(seed):
    rng = np.random.default_rng(seed)
    shape = (15, 18)
    wip = rng.integers(0, 21, shape) / 20
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    # Ground falls toward the stream up to a ridge at col 12, then away.
    dem = 0.05 * np.minimum(cc, 24 - cc) + 0.03 * np.abs(rr - 7) + rng.normal(0, 0.05, shape)
    streams = _column_stream(shape)
    r = threshold_connectivity(wip, streams, 1, 1, dem=dem, downstream_only=True)
    z, _ = downstream_elevations(dem, None, streams, "fill")
    expected = _brute_force(wip, streams, z, 0.01)
    np.testing.assert_array_equal(r["threshold"], expected)
    assert np.any(expected == NEVER) and np.any(expected > 0)  # the rule bites


def test_corridor_running_minimum():
    wip = np.zeros((3, 7))
    wip[1, 1:] = [0.9, 0.6, 0.8, 0.3, 0.7, 0.95]
    r = threshold_connectivity(wip, _column_stream(wip.shape), 1, 1)
    np.testing.assert_allclose(r["threshold"][1, 1:], [0.9, 0.6, 0.6, 0.3, 0.3, 0.3])
    np.testing.assert_allclose(r["route_threshold"][1, 1:], [1.0, 0.9, 0.6, 0.6, 0.3, 0.3])


def test_best_of_two_routes():
    wip = np.zeros((7, 8))
    wip[1, 1:5] = 0.4  # weak route
    wip[5, 1:5] = 0.7  # strong route
    wip[2, 5] = wip[4, 5] = 1.0
    wip[3, 6] = 0.9
    r = threshold_connectivity(wip, _column_stream(wip.shape), 1, 1)
    assert r["threshold"][3, 6] == pytest.approx(0.7)


def test_stream_pixels_and_neighbours():
    wip = np.full((4, 5), 0.3)
    wip[0, 0] = np.nan  # stream pixel without wip
    r = threshold_connectivity(wip, _column_stream(wip.shape), 1, 1)
    np.testing.assert_allclose(r["threshold"][1:, 0], 0.3)  # streams: own wip
    assert np.isnan(r["threshold"][0, 0])
    np.testing.assert_allclose(r["route_threshold"][:, :2], 1.0)
    np.testing.assert_allclose(r["threshold"][:, 1], 0.3)


def test_downstream_uphill_never_connects():
    x = np.arange(8.0)
    wip = np.full((3, 8), 0.9)
    steep = np.tile(-0.02 * x, (3, 1))  # 2% climb toward the stream
    gentle = np.tile(-0.005 * x, (3, 1))
    s = _column_stream(wip.shape)
    r = threshold_connectivity(wip, s, 1, 1, dem=steep, downstream_only=True)
    assert np.all(r["threshold"][:, 1:] == NEVER)
    assert np.all(np.isnan(r["reach_id"][:, 1:]))
    r = threshold_connectivity(wip, s, 1, 1, dem=gentle, downstream_only=True)
    np.testing.assert_allclose(r["threshold"][:, 1:], 0.9)


def test_reach_id_follows_best_route():
    wip = np.full((3, 11), 0.5)
    wip[:, 7:] = 0.9
    streams = np.zeros(wip.shape, bool)
    streams[:, 0] = streams[:, -1] = True
    labels = np.full(wip.shape, -1)
    labels[:, 0], labels[:, -1] = 7, 9
    r = threshold_connectivity(wip, streams, 1, 1, labels=labels)
    assert np.all(r["reach_id"][:, 7:] == 9)
    assert np.all(r["reach_id"][:, 1:3] == 7)


def _broken_band():
    # A 5-row band of 0.8 (rows 2-6, stream column included) cut by one dry
    # column at col 6; zeros elsewhere.
    wip = np.zeros((9, 13))
    wip[2:7, :] = 0.8
    wip[2:7, 6] = 0.0
    return wip


def test_focal_mean_ignores_nodata_and_edges():
    from wetconn.threshold import focal_mean

    rng = np.random.default_rng(0)
    wip = rng.uniform(0, 1, (9, 11))
    valid = rng.uniform(size=wip.shape) > 0.15
    wip[~valid] = np.nan
    for k in (3, 5):
        h = k // 2
        expected = np.full(wip.shape, np.nan)
        for i, j in zip(*np.nonzero(valid)):
            expected[i, j] = np.nanmean(wip[max(0, i - h) : i + h + 1, max(0, j - h) : j + h + 1])
        np.testing.assert_allclose(focal_mean(wip, valid, k), expected, atol=1e-12)


def test_multiscale_window_one_is_the_plain_method():
    from wetconn.threshold import multiscale_threshold

    rng = np.random.default_rng(3)
    wip = rng.integers(0, 21, (15, 18)) / 20
    s = _column_stream(wip.shape)
    plain = threshold_connectivity(wip, s, 1, 1)
    multi = multiscale_threshold(wip, s, 1, 1, windows=[1])
    np.testing.assert_array_equal(multi["threshold"], plain["threshold"])
    assert np.all(multi["window"] == 1)


def test_multiscale_bridges_a_one_pixel_break():
    from wetconn.threshold import multiscale_threshold

    wip = _broken_band()
    seen = {}
    r = multiscale_threshold(
        wip, _column_stream(wip.shape), 1, 1, windows=[1, 3, 5, 7],
        on_window=lambda k, res: seen.__setitem__(k, res["threshold"][4, 10]),
    )
    assert list(seen) == [1, 3, 5, 7]
    np.testing.assert_allclose(
        [seen[1], seen[3], seen[5], seen[7]], [0.0, 0.8 * 6 / 9, 0.8 * 20 / 25, 0.8 * 30 / 49]
    )
    assert r["threshold"][4, 10] == pytest.approx(0.64) and r["window"][4, 10] == 5


def test_multiscale_never_exceeds_raw_wip_and_ties_keep_smallest_window():
    from wetconn.threshold import multiscale_threshold

    wip = _broken_band()
    r = multiscale_threshold(wip, _column_stream(wip.shape), 1, 1, windows=[1, 3, 5, 7])
    assert np.all(r["threshold"] <= wip + 1e-12)
    assert np.all(r["threshold"][:, 6] == 0.0)  # the dry pixels themselves stay at 0
    assert np.all(r["window"][2:7, 1:6] == 1)  # already at raw wip before averaging
