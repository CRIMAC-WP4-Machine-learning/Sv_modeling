"""Sampled volume, volume backscattering s_v and fish number density estimates."""

from typing import NamedTuple

import numpy as np


def beam_shell_volume(mean_range, thickness, beam_half_angle_deg, approximate=False):
    """Volume (m^3) of a circular conical shell along the beam axis.

    Integrates dV = r^2 sin(theta) dr dtheta dphi over r in
    [R - DR/2, R + DR/2], theta in [0, alpha], phi in [0, 2*pi]:

        V = (2*pi/3) * (1 - cos(alpha)) * (r_max^3 - r_min^3)

    With ``approximate=True`` the narrow-beam, thin-shell form is returned:

        V ~= pi * R^2 * alpha^2 * DR

    This is the beam-sampled volume, not necessarily the volume of the whole
    school if it extends beyond the beam.
    """
    alpha = np.deg2rad(beam_half_angle_deg)
    if approximate:
        return np.pi * mean_range**2 * alpha**2 * thickness
    r_min = mean_range - thickness / 2
    r_max = mean_range + thickness / 2
    return (2 * np.pi / 3) * (1 - np.cos(alpha)) * (r_max**3 - r_min**3)


def sv_from_pfar(p_far, ref_range, volume):
    """Volume backscattering coefficient s_v(f) (m^-1) of the coherent field.

    s_v = |p_far|^2 * R^4 / V, with R the reference range (m) and V the
    sampled volume (m^3).
    """
    return np.abs(np.asarray(p_far)) ** 2 * ref_range**4 / volume


def band_average(values, freq_hz, f1, f2, axis=-1):
    """Average ``values`` over the band [f1, f2] (Hz) with a flat spectral weight.

    Trapezoidal integral over the grid points inside the band, divided by the
    width spanned by those points (freq_band[-1] - freq_band[0]).
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    mask = (freq_hz >= f1) & (freq_hz <= f2)
    if mask.sum() < 2:
        raise ValueError(f"Fewer than two frequency points in band [{f1}, {f2}] Hz")
    freq_band = freq_hz[mask]
    width = freq_band[-1] - freq_band[0]
    return np.trapezoid(np.compress(mask, values, axis=axis), freq_band, axis=axis) / width


class DensityEstimate(NamedTuple):
    """Output of estimate_density. Densities in fish/m^3, s_v in m^-1."""

    coherent: float
    incoherent: float
    true: float
    sv_coherent: float
    sv_incoherent: float
    sigma_bs_mean: float
    svf: np.ndarray


def estimate_density(result, ref_range, volume, band_hz):
    """Estimate fish number density from a coherent_backscatter result.

    Over the band [f1, f2] (flat weighting):

      sigma_bs_i  = band average of |f_bs_i|^2 for each fish
      sigma_bs    = mean_i sigma_bs_i
      sv_coherent = band average of |p_far|^2 R^4 / V
      sv_incoh    = R^4 / V * sum_i sigma_bs_i / R_i^4
      density     = sv / sigma_bs  (coherent and incoherent)
      true        = N / V

    Parameters
    ----------
    result : BackscatterResult
        Output of sv_modeling.scattering.coherent_backscatter.
    ref_range : float
        Reference range R (m), e.g. the mean school range.
    volume : float
        Sampled volume V (m^3), e.g. from beam_shell_volume.
    band_hz : (f1, f2)
        Frequency band (Hz).

    Returns
    -------
    DensityEstimate
        Includes ``svf``, the coherent s_v(f) over the full frequency grid.
    """
    f1, f2 = band_hz
    svf = sv_from_pfar(result.p_far, ref_range, volume)
    sv_coherent = band_average(svf, result.freq_hz, f1, f2)

    sigma_bs_fish = band_average(result.sigma_bs, result.freq_hz, f1, f2, axis=1)
    sigma_bs_mean = np.mean(sigma_bs_fish)

    sv_incoherent = ref_range**4 / volume * np.sum(sigma_bs_fish / result.distances**4)

    return DensityEstimate(
        coherent=sv_coherent / sigma_bs_mean,
        incoherent=sv_incoherent / sigma_bs_mean,
        true=len(result.distances) / volume,
        sv_coherent=sv_coherent,
        sv_incoherent=sv_incoherent,
        sigma_bs_mean=sigma_bs_mean,
        svf=svf,
    )

def estimate_density_sphericalbubbles(
    freq_hz, p_far, sigma_bs, distances, ref_range, volume, band_hz
):
    """Estimate bubble number density from coherent and incoherent backscattering."""

    f1, f2 = band_hz
    svf = sv_from_pfar(p_far, ref_range, volume)
    sv_coherent = band_average(svf, freq_hz, f1, f2)

    sigma_bs_bubbles = band_average(sigma_bs, freq_hz, f1, f2, axis=1)
    sigma_bs_mean = np.mean(sigma_bs_bubbles)

    sv_incoherent = ref_range**4 / volume * np.sum(sigma_bs_bubbles / distances**4)

    return DensityEstimate(
        coherent=sv_coherent / sigma_bs_mean,
        incoherent=sv_incoherent / sigma_bs_mean,
        true=len(distances) / volume,
        sv_coherent=sv_coherent,
        sv_incoherent=sv_incoherent,
        sigma_bs_mean=sigma_bs_mean,
        svf=svf,
    )