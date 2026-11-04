import numpy as np
import pytest
from scipy.integrate import tplquad

from sv_modeling.density import band_average, beam_shell_volume, estimate_density, sv_from_pfar
from sv_modeling.scattering import BackscatterResult

FREQ = np.arange(10e3, 100e3 + 1, 1e3)


def make_result(distances, sigma_bs, p_far):
    distances = np.asarray(distances, dtype=float)
    n = len(distances)
    return BackscatterResult(
        freq_hz=FREQ,
        p_far=np.broadcast_to(p_far, FREQ.shape).astype(complex),
        sigma_bs=np.broadcast_to(sigma_bs, (n, len(FREQ))).astype(float),
        distances=distances,
        incident_angles=np.full(n, 90.0),
        filenames=[""] * n,
    )


def test_beam_shell_volume_matches_numerical_integral():
    R, DR, half_angle = 50.0, 1.0, 3.5
    alpha = np.deg2rad(half_angle)
    numeric, _ = tplquad(
        lambda r, th, ph: r**2 * np.sin(th),
        0, 2 * np.pi, 0, alpha, R - DR / 2, R + DR / 2,
    )
    assert beam_shell_volume(R, DR, half_angle) == pytest.approx(numeric, rel=1e-10)


def test_beam_shell_volume_approximation_for_narrow_beam():
    exact = beam_shell_volume(50.0, 0.1, 1.0)
    approx = beam_shell_volume(50.0, 0.1, 1.0, approximate=True)
    assert approx == pytest.approx(exact, rel=1e-3)


def test_sv_from_pfar():
    assert sv_from_pfar(2.0 + 0j, ref_range=10.0, volume=5.0) == pytest.approx(4 * 1e4 / 5)


def test_band_average_constant_and_linear():
    assert band_average(np.full(FREQ.shape, 3.0), FREQ, 38e3, 70e3) == pytest.approx(3.0)
    # linear in f: average is the value at the band centre
    assert band_average(FREQ, FREQ, 38e3, 70e3) == pytest.approx(54e3)


def test_band_average_along_axis():
    values = np.vstack([np.ones_like(FREQ), 2 * np.ones_like(FREQ)])
    np.testing.assert_allclose(band_average(values, FREQ, 38e3, 70e3, axis=1), [1.0, 2.0])


def test_band_average_needs_two_points():
    with pytest.raises(ValueError, match="Fewer than two"):
        band_average(FREQ, FREQ, 38.2e3, 38.8e3)


def test_single_fish_on_reference_range_gives_true_density():
    R, V, sigma = 50.0, 2.0, 1e-4
    # one fish at R: |p_far| = |f_bs| / R^2, |f_bs|^2 = sigma
    res = make_result([R], sigma, np.sqrt(sigma) / R**2)
    est = estimate_density(res, R, V, (38e3, 70e3))
    assert est.true == pytest.approx(1 / V)
    assert est.coherent == pytest.approx(1 / V)
    assert est.incoherent == pytest.approx(1 / V)
    assert est.sigma_bs_mean == pytest.approx(sigma)


def test_incoherent_density_exact_for_fish_at_reference_range():
    R, V, n = 50.0, 3.0, 7
    sigma = np.linspace(1e-5, 1e-4, n)[:, None]  # different fish sizes
    res = make_result([R] * n, sigma, 0.0)
    est = estimate_density(res, R, V, (38e3, 70e3))
    assert est.incoherent == pytest.approx(n / V)
    assert est.true == pytest.approx(n / V)
    assert est.coherent == 0.0  # fully destructive field


def test_incoherent_range_compensation():
    # a fish at 2R contributes 1/16 of one at R
    R, V, sigma = 50.0, 1.0, 1e-4
    est = estimate_density(make_result([R, 2 * R], sigma, 0.0), R, V, (38e3, 70e3))
    assert est.sv_incoherent == pytest.approx(R**4 / V * sigma * (1 / R**4 + 1 / (2 * R) ** 4))
