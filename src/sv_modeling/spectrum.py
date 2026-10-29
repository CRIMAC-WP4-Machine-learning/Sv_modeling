"""Energy spectral density of transient signals."""

import numpy as np
import scipy.fftpack


def ESD_BKH(SigX, dt):
    """Energy spectral density (ESD) estimator for transient signals.

    The ESD describes how the energy of a signal is distributed with frequency.

    Parameters
    ----------
    SigX : array_like, shape (N,) or (1, N)
        Time signal.
    dt : float
        Sampling interval (1/Fs) in seconds.

    Returns
    -------
    ndarray, shape (M, 2)
        Column 0: frequency (Hz) from 0 to Fs/2 in steps of 1/(N*dt).
        Column 1: one-sided ESD.
    """
    SigX = np.asarray(SigX).reshape(1, -1)
    N = np.max(SigX.shape)

    # Rectangular window (no windowing)
    Win = np.ones((1, N))
    WSig = (Win * SigX).reshape(1, -1)

    Fs = 1 / dt
    DF = 1 / (N * dt)
    F_vec = np.arange(0, Fs / 2, DF).reshape(1, -1)
    Nfvec = np.max(F_vec.shape)

    DFT_Wsig = scipy.fftpack.fft(WSig[0, :])

    ESD = 2 * dt * dt * (np.absolute(DFT_Wsig[0:Nfvec]) ** 2)
    ESD[0] = 0.5 * ESD[0]

    return np.column_stack((F_vec.reshape(-1, 1), ESD.reshape(-1, 1)))
