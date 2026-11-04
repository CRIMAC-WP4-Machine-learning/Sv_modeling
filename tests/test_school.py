import numpy as np
import pytest
from scipy.spatial.distance import pdist

from sv_modeling.school import (
    angles_to_unit_vectors,
    orient_fish,
    place_fish_beam_shell,
    place_fish_spheroid,
)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_spheroid_points_inside_and_separated():
    a, b, depth, min_dist = 2.0, 0.6, 50.0, 0.1
    pts = place_fish_spheroid(200, a, b, depth, min_dist, rng())
    assert pts.shape == (200, 3)
    dz = pts[:, 2] + depth
    assert np.all(pts[:, 0] ** 2 / a**2 + pts[:, 1] ** 2 / a**2 + dz**2 / b**2 <= 1)
    assert pdist(pts).min() >= min_dist
    assert np.all(pts[:, 2] < 0)  # below the transducer, z negative


def test_beam_shell_points_inside_and_separated():
    R, DR, half_angle, min_dist = 50.0, 1.0, 3.5, 0.1
    pts = place_fish_beam_shell(200, R, DR, half_angle, min_dist, rng())
    r = np.linalg.norm(pts, axis=1)
    off_axis = np.degrees(np.arccos(pts[:, 2] / r))
    assert pts.shape == (200, 3)
    assert np.all((r >= R - DR / 2) & (r <= R + DR / 2))
    assert off_axis.max() <= half_angle + 1e-12
    assert pdist(pts).min() >= min_dist


@pytest.mark.parametrize(
    "place",
    [
        lambda g: place_fish_spheroid(50, 2.0, 0.6, 50.0, 0.1, g),
        lambda g: place_fish_beam_shell(50, 50.0, 1.0, 3.5, 0.1, g),
        lambda g: orient_fish(50, (-10, 10), (80, 100), g)[0],
    ],
    ids=["spheroid", "beam_shell", "orientation"],
)
def test_same_seed_reproduces(place):
    np.testing.assert_array_equal(place(rng(42)), place(rng(42)))
    assert not np.array_equal(place(rng(42)), place(rng(43)))


def test_impossible_packing_raises():
    # 1000 fish 1 m apart do not fit in a 0.5 m spheroid
    with pytest.raises(RuntimeError, match="Placed only"):
        place_fish_spheroid(1000, 0.5, 0.5, 50.0, 1.0, rng(), max_attempts=10_000)


@pytest.mark.parametrize(
    "theta_deg, phi_deg, expected",
    [
        (0, 90, [1, 0, 0]),   # horizontal along +X
        (90, 90, [0, 1, 0]),  # horizontal along +Y
        (0, 0, [0, 0, 1]),    # vertical
    ],
)
def test_angles_to_unit_vectors(theta_deg, phi_deg, expected):
    v = angles_to_unit_vectors(np.radians(theta_deg), np.radians(phi_deg))
    np.testing.assert_allclose(v, [expected], atol=1e-15)


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_orient_fish_statistics(distribution):
    n = 20_000
    o, theta, phi = orient_fish(n, (-10, 10), (80, 100), rng(), distribution=distribution)
    assert o.shape == (n, 3)
    np.testing.assert_allclose(np.linalg.norm(o, axis=1), 1.0)
    np.testing.assert_allclose(angles_to_unit_vectors(theta, phi), o)
    assert np.degrees(theta).mean() == pytest.approx(0, abs=0.2)
    assert np.degrees(phi).mean() == pytest.approx(90, abs=0.2)
    if distribution == "uniform":
        assert np.degrees(theta).min() >= -10 and np.degrees(theta).max() <= 10
    else:
        assert np.degrees(theta).std() == pytest.approx(5.0, rel=0.03)  # 1/4 of the 20 deg width


def test_orient_fish_unknown_distribution():
    with pytest.raises(ValueError, match="distribution"):
        orient_fish(10, (0, 1), (89, 91), rng(), distribution="gaussian")
