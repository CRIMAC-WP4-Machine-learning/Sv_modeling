"""Transmit signal generation."""

import numpy as np


def func_Gen_SweepSig_Explicit(T_whole, T_center, T_effectiveDuration, Fs, F_start, F_end, Mexpo):
    """Linear frequency sweep with a super-Gaussian envelope.

    Parameters
    ----------
    T_whole : float
        Total length of the time signal, including zero padding (s).
    T_center : float
        Time at which the centre of the pulse is located (s).
    T_effectiveDuration : float
        Effective duration of the pulse (s).
    Fs : float
        Sampling frequency (Hz).
    F_start, F_end : float
        Start and end frequency of the sweep (Hz).
    Mexpo : float
        Envelope exponent; larger values give faster ramping.

    Returns
    -------
    t_vec : ndarray
        Time vector (s), starting at 0.
    y : ndarray
        Signal.
    """
    T = T_effectiveDuration
    beta = np.pi * (F_start + F_end)
    alpha = np.pi * (F_end - F_start) / T
    dt = 1 / Fs

    t = np.arange(-T_center, (T_whole - T_center), dt)
    t_vec = t + T_center
    y = np.exp(-((2 * t / T) ** (2 * Mexpo))) * np.sin(alpha * t * t + beta * t)
    return t_vec, y
