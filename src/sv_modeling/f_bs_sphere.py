
import numpy as np
from scipy.special import spherical_jn, spherical_yn


def func_formfunc_FluidFiledSphere_Rinf(
    R, ro_w, ro_fluid, c_w, c_fluid, freq,
    R_inf=None, use_exact_farfield=False, extra_orders=20
):
    """
    Calculate the complex far-field backscattering amplitude f_bs(f)
    of a fluid-filled sphere.

    Parameters
    ----------
    R : float
        Sphere radius (m).
    ro_w, ro_fluid : float
        Water and sphere-fluid densities (kg/m^3).
    c_w, c_fluid : float
        Sound speeds in water and sphere fluid (m/s).
    freq : float or array_like
        Frequencies (Hz).
    R_inf : float, optional
        Observation distance (m), used only for finite-distance evaluation.
    use_exact_farfield : bool
        False: analytical far-field limit (default).
        True: original finite-distance Hankel expression.
    extra_orders : int
        Additional spherical harmonic terms (default 20).

    Returns
    -------
    f_bs : ndarray of complex
        Complex backscattering amplitude (m), one value per frequency.

    Notes
    -----
    Each frequency uses its own truncation order:
        M(f) = ceil(max(k_w*R, k_g*R)) + extra_orders

    The original complex conjugation convention is preserved.
    """

    # 1. Frequency input and validation
    freq = np.atleast_1d(np.asarray(freq, dtype=float))

    if not np.all(np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("Frequencies must be finite and positive.")
    if not np.all(np.isfinite([R, ro_w, ro_fluid, c_w, c_fluid])):
        raise ValueError("Physical parameters must be finite.")
    if R <= 0 or min(ro_w, ro_fluid, c_w, c_fluid) <= 0:
        raise ValueError("Radius, densities and sound speeds must be positive.")
    if not isinstance(extra_orders, (int, np.integer)) or extra_orders < 1:
        raise ValueError("extra_orders must be a positive integer.")
    if use_exact_farfield and (R_inf is None or not np.isfinite(R_inf) or R_inf <= 0):
        raise ValueError("A positive finite R_inf is required.")

    # 2. Acoustic wavenumbers
    kw = 2 * np.pi * freq / c_w
    kg = 2 * np.pi * freq / c_fluid

    xw = kw * R
    xg = kg * R

    # 3. Frequency-dependent truncation
    x = np.maximum(xw, xg)
    M_vec = np.ceil(x).astype(int) + extra_orders
    M_max = int(np.max(M_vec))

    f_sum = np.zeros(freq.shape, dtype=np.complex128)

    # 4. Spherical harmonic expansion
    for m in range(M_max):

        # Select only frequencies requiring order m
        mask = m < M_vec

        if not np.any(mask):
            break

        kw_m = kw[mask]
        kg_m = kg[mask]
        xw_m = xw[mask]
        xg_m = xg[mask]

        # Bessel functions in water
        jw = spherical_jn(m, xw_m)
        djw = spherical_jn(m, xw_m, derivative=True)

        yw = spherical_yn(m, xw_m)
        dyw = spherical_yn(m, xw_m, derivative=True)

        hw = jw + 1j * yw
        dhw = djw + 1j * dyw

        # Bessel functions inside sphere
        jg = spherical_jn(m, xg_m)
        djg = spherical_jn(m, xg_m, derivative=True)

        # Boundary-condition matrix
        a = -kw_m * dhw
        b = kg_m * djg
        c = -ro_w * hw
        d = ro_fluid * jg

        # Right-hand side without common i**m factor
        factor = 2 * m + 1

        F0 = factor * kw_m * djw
        F1 = factor * ro_w * jw

        # Scattering coefficient
        det = a * d - b * c
        A_reduced = (F0 * d - b * F1) / det

        # 5. Backscattering contribution
        if use_exact_farfield:

            # Original finite-distance expression
            xinf_m = kw_m * R_inf
            h_inf = spherical_jn(m, xinf_m) + 1j * spherical_yn(m, xinf_m)
            Am = (1j ** m) * A_reduced

            f_sum[mask] += Am * (-1)**m * h_inf

        else:

            # Analytical far-field limit
            f_sum[mask] += (-1)**m * A_reduced

    # 6. Final backscattering amplitude
    if use_exact_farfield:
        f_bs = R_inf * f_sum * np.exp(-1j * kw * R_inf)
    else:
        f_bs = (-1j / kw) * f_sum

    # Preserve original phase convention
    f_bs = np.conj(f_bs)

    return f_bs
