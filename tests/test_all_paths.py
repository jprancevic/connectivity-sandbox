import numpy as np
import pytest

from wetconn import all_paths_connectivity, least_cost_connectivity

N = 20000  # enough maps for tight comparisons on tiny grids


def _column_stream(shape):
    s = np.zeros(shape, bool)
    s[:, 0] = True
    return s


def _close(r, idx, expected, k=4):
    """Monte Carlo estimate within k standard errors of the exact value."""
    se = np.sqrt(expected * (1 - expected) / N)
    assert abs(r["p_conn"][idx] - expected) <= k * se + 1e-12


def test_single_corridor_is_product_of_pixels_between():
    wip = np.zeros((5, 8))
    wip[2, 1:6] = 0.8
    r = all_paths_connectivity(wip, _column_stream(wip.shape), 1, 1, n_sims=N, seed=1)
    for col in range(2, 7):
        _close(r, (2, col), 0.8 ** (col - 1))
    np.testing.assert_allclose(r["p_joint"], np.where(np.isfinite(wip), wip, np.nan) * r["p_conn"])


def _two_corridors(a, b):
    # Source pixel (3, 6) reaches the stream via row 1 (wip a) or row 5 (wip
    # b); the two routes share no pixels and don't touch.
    wip = np.zeros((7, 8))
    wip[1, 1:5] = a
    wip[5, 1:5] = b
    wip[2, 5] = wip[4, 5] = 1.0
    return wip


def test_two_separate_routes_combine():
    a, b = 0.8, 0.7
    wip = _two_corridors(a, b)
    r = all_paths_connectivity(wip, _column_stream(wip.shape), 1, 1, n_sims=N, seed=2)
    _close(r, (3, 6), 1 - (1 - a**4) * (1 - b**4))
    lc = least_cost_connectivity(wip, _column_stream(wip.shape), 1, 1, min_wip=0.0)
    assert r["p_conn"][3, 6] > lc["p_conn"][3, 6] + 0.1  # beats the best single route


def test_wide_corridor_beats_narrow():
    narrow = np.zeros((9, 14))
    narrow[4, 1:] = 0.8
    wide = np.zeros((9, 14))
    wide[3:6, 1:] = 0.8
    s = _column_stream(narrow.shape)
    rn = all_paths_connectivity(narrow, s, 1, 1, n_sims=4000, seed=3)
    rw = all_paths_connectivity(wide, s, 1, 1, n_sims=4000, seed=3)
    assert rw["p_conn"][4, 12] > 3 * rn["p_conn"][4, 12]


def test_path_threshold_makes_mapped_wetland_certain():
    wip = np.zeros((5, 12))
    wip[2, 1:] = 0.8
    r = all_paths_connectivity(
        wip, _column_stream(wip.shape), 1, 1, path_threshold=0.75, n_sims=200
    )
    assert np.all(r["p_conn"][2] == 1.0)
    np.testing.assert_allclose(r["p_joint"][2, 1:], 0.8)


def test_zero_wip_barrier_and_floor():
    wip = np.full((5, 8), 0.9)
    wip[:, 3] = 0.0
    s = _column_stream(wip.shape)
    hard = all_paths_connectivity(wip, s, 1, 1, n_sims=500)
    assert np.all(hard["p_conn"][:, 5:] == 0)
    assert np.all(hard["p_conn"][:, 4] == 0)
    assert np.all(hard["p_conn"][:, 3] > 0.5)  # wall pixels would connect if wet
    assert np.all(hard["p_joint"][:, 3] == 0)
    floored = all_paths_connectivity(wip, s, 1, 1, min_wip=0.2, n_sims=500)
    assert np.all(floored["p_conn"][:, 5] > 0)


def test_seed_reproducible():
    rng = np.random.default_rng(0)
    wip = rng.uniform(0, 1, (30, 30))
    s = _column_stream(wip.shape)
    a = all_paths_connectivity(wip, s, 1, 1, n_sims=50, seed=7)
    b = all_paths_connectivity(wip, s, 1, 1, n_sims=50, seed=7)
    c = all_paths_connectivity(wip, s, 1, 1, n_sims=50, seed=8)
    np.testing.assert_array_equal(a["p_conn"], b["p_conn"])
    assert not np.array_equal(a["p_conn"], c["p_conn"])


@pytest.mark.parametrize("dx", [1.0, 3.0])
def test_downstream_only_slope_tolerance(dx):
    n = 8
    x = dx * np.arange(n)
    wip = np.full((3, n), 0.9)
    s = _column_stream(wip.shape)
    kw = dict(downstream_only=True, n_sims=300)
    gentle = all_paths_connectivity(wip, s, dx, dx, dem=np.tile(-0.005 * x, (3, 1)), **kw)
    steep = all_paths_connectivity(wip, s, dx, dx, dem=np.tile(-0.02 * x, (3, 1)), **kw)
    assert np.all(gentle["p_conn"][:, 1:3] > 0)
    assert np.all(steep["p_conn"][:, 1:] == 0)


def test_downstream_only_matches_any_direction_on_downhill_plane():
    wip = np.full((6, 8), 0.7)
    s = _column_stream(wip.shape)
    dem = np.tile(0.3 * np.arange(8.0), (6, 1))
    free = all_paths_connectivity(wip, s, 1, 1, n_sims=N, seed=4)
    down = all_paths_connectivity(wip, s, 1, 1, dem=dem, downstream_only=True, n_sims=N, seed=5)
    # Steps toward the stream and level steps across the slope are allowed;
    # steps away from it are not, so downstream-only can only be lower.
    assert np.all(down["p_conn"] <= free["p_conn"] + 4 * free["p_conn_se"] + 1e-3)
    assert np.mean(free["p_conn"][:, 2:] - down["p_conn"][:, 2:]) > 0.005
    # A single downhill corridor loses nothing.
    corridor = np.zeros_like(wip)
    corridor[3, 1:] = 0.8
    one = all_paths_connectivity(
        corridor, s, 1, 1, dem=dem, downstream_only=True, n_sims=N, seed=6
    )
    for col in range(2, 8):
        _close(one, (3, col), 0.8 ** (col - 1))


@pytest.mark.parametrize("downstream", [False, True])
def test_never_below_least_cost(downstream):
    # With independent pixels and ref_length = pixel size, method 1 scores one
    # route (charging diagonals extra); all-paths counts every route.
    rng = np.random.default_rng(6)
    shape = (40, 50)
    wip = rng.uniform(0.3, 1.0, shape)
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    dem = 0.2 * cc + np.abs(rr - 20) * 0.1 + rng.normal(0, 0.2, shape)
    s = _column_stream(shape)
    kw = dict(dem=dem, downstream_only=downstream)
    ap = all_paths_connectivity(wip, s, 1, 1, n_sims=2000, seed=9, **kw)
    lc = least_cost_connectivity(wip, s, 1, 1, ref_length=1.0, **kw)
    assert np.all(ap["p_conn"] >= lc["p_conn"] - 4 * ap["p_conn_se"] - 0.01)
    assert np.mean(ap["p_conn"] > lc["p_conn"] + 0.05) > 0.2  # redundancy shows up
