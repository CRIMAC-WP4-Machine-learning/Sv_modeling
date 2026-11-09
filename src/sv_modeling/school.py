"""Fish school geometry: random positions and orientations.

Every random function takes a numpy Generator ``rng`` so runs are reproducible:
    rng = np.random.default_rng(seed)

Orientation angles (spherical coordinates, radians):
  theta : azimuth in the XY plane, measured from +X toward +Y.
  phi   : polar angle from +Z; phi = 90 deg means a horizontal fish.
"""

import numpy as np

DEFAULT_MAX_ATTEMPTS = 1_000_000


def _rejection_sample(n, min_dist, draw, max_attempts):
    """Draw candidate points with ``draw()`` until n are accepted.

    ``draw`` returns a point (3,) or None if the candidate falls outside the
    volume. Accepted points are at least ``min_dist`` from all previous ones.
    """
    points = np.empty((n, 3))
    n_accepted = 0
    for _ in range(max_attempts):
        if n_accepted == n:
            break
        p = draw()
        if p is None:
            continue
        if n_accepted == 0 or np.min(np.linalg.norm(points[:n_accepted] - p, axis=1)) >= min_dist:
            points[n_accepted] = p
            n_accepted += 1
    if n_accepted < n:
        raise RuntimeError(
            f"Placed only {n_accepted} of {n} fish after {max_attempts} attempts; "
            "reduce n or min_dist, or enlarge the volume."
        )
    return points


def place_fish_spheroid(n, a, b, depth, min_dist, rng, max_attempts=DEFAULT_MAX_ATTEMPTS):
    """Place n fish uniformly in a spheroid centred at (0, 0, -depth).

    Horizontal semi-axes are both ``a`` and the vertical semi-axis is ``b``
    (a > b: oblate, a < b: prolate). Fish are at least ``min_dist`` apart.
    z is negative below the transducer (z = -depth at the centre).

    Returns
    -------
    ndarray, shape (n, 3)
    """

    def draw():
        x = rng.uniform(-a, a)
        y = rng.uniform(-a, a)
        dz = rng.uniform(-b, b)
        if x**2 / a**2 + y**2 / a**2 + dz**2 / b**2 <= 1:
            return np.array([x, y, dz - depth])
        return None

    return _rejection_sample(n, min_dist, draw, max_attempts)


def place_fish_beam_shell(
    n, mean_range, thickness, beam_half_angle_deg, min_dist, rng, max_attempts=DEFAULT_MAX_ATTEMPTS
):
    """Place n fish uniformly in a circular conical shell along +Z (the beam).

    The shell spans slant ranges [mean_range - thickness/2, mean_range + thickness/2]
    and polar angles up to ``beam_half_angle_deg`` from +Z. Sampling is uniform
    in volume. Fish are at least ``min_dist`` apart. z is positive downward.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    r_min = mean_range - thickness / 2
    r_max = mean_range + thickness / 2
    cos_alpha = np.cos(np.deg2rad(beam_half_angle_deg))

    def draw():
        r = np.cbrt(rng.uniform(r_min**3, r_max**3))
        gamma = np.arccos(rng.uniform(cos_alpha, 1.0))  # angle from beam axis
        azimuth = rng.uniform(0, 2 * np.pi)
        return np.array(
            [r * np.sin(gamma) * np.cos(azimuth), r * np.sin(gamma) * np.sin(azimuth), r * np.cos(gamma)]
        )

    return _rejection_sample(n, min_dist, draw, max_attempts)


def angles_to_unit_vectors(theta, phi):
    """Convert azimuth theta and polar angle phi (radians) to unit vectors, shape (N, 3)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    v = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=1)
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]


def _truncated_normal(lo, hi, n, rng):
    """Normal(mean=(lo+hi)/2, std=(hi-lo)/4) restricted to [lo, hi] by redrawing."""
    mu, sigma = (lo + hi) / 2, 0.25 * (hi - lo)
    x = rng.normal(mu, sigma, n)
    bad = (x < lo) | (x > hi)
    while bad.any():
        x[bad] = rng.normal(mu, sigma, bad.sum())
        bad = (x < lo) | (x > hi)
    return x


def orient_fish(n, theta_range_deg, phi_range_deg, rng, distribution="normal"):
    """Draw random fish orientations.

    Parameters
    ----------
    n : int
        Number of fish.
    theta_range_deg, phi_range_deg : (low, high)
        Angle ranges in degrees.
    rng : numpy.random.Generator
    distribution : {"normal", "uniform"}
        "uniform": uniform within each range.
        "normal": mean at the range centre, std = 1/4 of the range width,
        truncated to the range (out-of-range draws are redrawn).

    Returns
    -------
    orientations : ndarray, shape (n, 3)
        Unit vectors.
    theta, phi : ndarray, shape (n,)
        Azimuth and polar angle (radians).
    """
    ranges = np.deg2rad([theta_range_deg, phi_range_deg])
    if distribution == "uniform":
        theta, phi = (rng.uniform(lo, hi, n) for lo, hi in ranges)
    elif distribution == "normal":
        theta, phi = (_truncated_normal(lo, hi, n, rng) for lo, hi in ranges)
    else:
        raise ValueError(f"Unknown distribution {distribution!r}; use 'normal' or 'uniform'")
    return angles_to_unit_vectors(theta, phi), theta, phi
