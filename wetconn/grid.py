"""Grid geometry shared by all connectivity methods.

Every method walks an 8-connected raster graph. The physical length of a step
depends on the pixel size and on whether the step is orthogonal or diagonal;
that length is what makes the path costs independent of raster resolution
(see ``README.md`` for the derivation).
"""

import numpy as np

# (row, col) offsets of the 8 neighbours, clockwise from east. Row index
# increases southward (north-up rasters).
OFFSETS = np.array(
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)],
    dtype=np.int64,
)


def step_lengths(dx, dy):
    """Physical length (map units) of a step to each neighbour in ``OFFSETS``.

    ``dx`` is the column spacing and ``dy`` the row spacing, both positive.
    Orthogonal steps are ``dx`` or ``dy``; diagonals are ``hypot(dx, dy)``.
    """
    dx = abs(float(dx))
    dy = abs(float(dy))
    return np.array([np.hypot(dr * dy, dc * dx) for dr, dc in OFFSETS], dtype=np.float64)


def pixel_size(transform):
    """Return ``(dx, dy)`` from an affine transform, rejecting rotated grids."""
    if transform.b != 0 or transform.d != 0:
        raise ValueError("Rotated/sheared rasters are not supported.")
    return abs(transform.a), abs(transform.e)


def attenuation(wip, valid, ref_length, min_wip=0.0, threshold=1.0):
    """Per-unit-length cost ``-ln(wip) / ref_length``.

    ``ref_length`` is the distance over which a path through a pixel of
    wetland probability ``p`` has connection probability exactly ``p``. A path
    of length ``l`` through that pixel then carries probability
    ``p ** (l / ref_length)``, which is the same whatever the pixel size.

    ``threshold`` rescales the probability used for travel to
    ``min(wip / threshold, 1)``: pixels at or above it (e.g. the wetland
    mapping threshold) cost nothing to cross. ``min_wip`` is applied first.

    Invalid pixels and pixels with ``wip == 0`` (below ``min_wip`` if given)
    get ``inf``, which makes them barriers.
    """
    if ref_length <= 0:
        raise ValueError("ref_length must be positive.")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    p = np.where(valid, np.minimum(wip, 1.0), np.nan).astype(np.float64)
    if min_wip > 0:
        p = np.where(valid, np.maximum(p, min_wip), np.nan)
    p = np.minimum(p / threshold, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -np.log(p) / ref_length
    lam[~valid] = np.inf
    return lam


def check_wip(wip, valid):
    """Raise if valid ``wip`` values are not probabilities."""
    vals = wip[valid]
    if vals.size and (vals.min() < 0 or vals.max() > 1 + 1e-6):
        raise ValueError(
            f"wip must be a probability in [0, 1]; got range "
            f"[{vals.min():.3g}, {vals.max():.3g}]."
        )
