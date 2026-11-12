"""Coherent backscatter from a set of fish using the modelled TS database."""

import os
from typing import NamedTuple

import numpy as np
import pandas as pd

from sv_modeling.database import Find_closestfile_in_database
from sv_modeling.geometry import Get_Angle
from sv_modeling.database import Get_bubble_fbs, Get_bubble_ith_spectrum

SOUND_SPEED = 1500.0  # m/s


class BackscatterResult(NamedTuple):
    """Output of coherent_backscatter.

    freq_hz : ndarray, shape (F,)
        Frequency grid of the database (Hz).
    p_far : ndarray, shape (F,), complex
        Coherent sum of f_bs_i / R_i**2 * exp(-2j k R_i) over all fish.
    sigma_bs : ndarray, shape (N, F)
        Backscattering cross-section |f_bs|**2 of each fish (m^2).
    distances : ndarray, shape (N,)
        Distance from the observation point to each fish (m).
    incident_angles : ndarray, shape (N,)
        Incident angle of each fish (deg, 0 = end-on, 90 = broadside).
    filenames : list of str
        Database file selected for each fish.
    """

    freq_hz: np.ndarray
    p_far: np.ndarray
    sigma_bs: np.ndarray
    distances: np.ndarray
    incident_angles: np.ndarray
    filenames: list


def _load_entry(database_dir, filename):
    """Return (freq_hz, f_bs) for one database CSV file."""
    df_csv = pd.read_csv(os.path.join(database_dir, filename))
    freq_hz = df_csv["Freq_kHz"].to_numpy(dtype=float) * 1000.0
    f_bs = df_csv["f_bs"].to_numpy()
    if not np.iscomplexobj(f_bs):
        f_bs = f_bs.astype(complex)  # stored as strings such as "1+2j"
    return freq_hz, f_bs


def coherent_backscatter(
    points,
    orientations,
    df_database,
    database_dir,
    observation_point=(0.0, 0.0, 0.0),
    fish_length=0.0,
    sound_speed=SOUND_SPEED,
    verbose=False,
):
    """Coherently sum the backscattered far field from N fish.

    For each fish the closest database entry (depth = |z|, length, incident
    angle) is selected and its complex f_bs(f) is propagated to the
    observation point with two-way spherical spreading and phase:

        p_far(f) = sum_i f_bs_i(f) / R_i**2 * exp(-1j * 2 * k * R_i),  k = 2*pi*f/c

    Parameters
    ----------
    points : array_like, shape (N, 3)
        Fish positions (m).
    orientations : array_like, shape (N, 3)
        Fish orientation vectors.
    df_database : DataFrame
        Output of Extract_database_metadata.
    database_dir : path
        Directory with the database CSV files.
    observation_point : array_like, shape (3,)
        Echosounder position (m).
    fish_length : float
        Fish length (m) used for the database lookup.
    sound_speed : float
        Sound speed c (m/s).
    verbose : bool
        Print the database selection for each fish.

    Returns
    -------
    BackscatterResult
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    orientations = np.atleast_2d(np.asarray(orientations, dtype=float))
    observation_point = np.asarray(observation_point, dtype=float)
    if points.shape != orientations.shape or points.shape[1] != 3:
        raise ValueError(
            f"points and orientations must both have shape (N, 3); got {points.shape} and {orientations.shape}"
        )

    n_fish = len(points)
    cache = {}  # filename -> (freq_hz, f_bs); many fish share the same entry
    freq_hz = None
    p_far = None
    sigma_bs = None
    distances = np.empty(n_fish)
    incident_angles = np.empty(n_fish)
    filenames = []

    for ii in range(n_fish):
        incident_angles[ii] = Get_Angle(observation_point, points[ii], orientations[ii])

        entry = Find_closestfile_in_database(
            df_database, np.abs(points[ii, 2]), fish_length, incident_angles[ii], verbose=verbose
        )
        filename = entry["filename"]
        filenames.append(filename)
        if filename not in cache:
            cache[filename] = _load_entry(database_dir, filename)
        freq_ii, f_bs_ii = cache[filename]

        if freq_hz is None:
            freq_hz = freq_ii
            p_far = np.zeros(len(freq_hz), dtype=complex)
            sigma_bs = np.zeros((n_fish, len(freq_hz)))
        elif not np.allclose(freq_ii, freq_hz, rtol=0, atol=1e-6):
            raise ValueError(f"Different frequency grids in scattering database ({filename})")

        distances[ii] = np.linalg.norm(observation_point - points[ii])
        k = 2 * np.pi * freq_hz / sound_speed
        p_far += f_bs_ii / distances[ii] ** 2 * np.exp(-1j * 2 * k * distances[ii])
        sigma_bs[ii] = np.abs(f_bs_ii) ** 2

    return BackscatterResult(freq_hz, p_far, sigma_bs, distances, incident_angles, filenames)


def total_ts(p_far, ref_distance):
    """Target strength (dB) of the coherent field, referenced to ref_distance (m).

    TS = 20 log10 |ref_distance**2 * p_far|
    """
    return 20 * np.log10(np.abs(ref_distance**2 * np.asarray(p_far)))


def coherent_backscatter_sphericalbubbles(
    points, bubble_radii, database_dir, gas,
    observation_point=(0.0, 0.0, 0.0),
    sound_speed=SOUND_SPEED, transducer_depth=0.0
):
    """Coherently sum backscattering from N spherical bubbles.

    Returns
    -------
    freq_hz : ndarray, shape (F,)
    p_far : ndarray, shape (F,), complex
    sigma_bs : ndarray, shape (N, F)
    distances : ndarray, shape (N,)
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    bubble_radii = np.asarray(bubble_radii, dtype=float)
    observation_point = np.asarray(observation_point, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n_bubbles = len(points)
    if bubble_radii.shape != (n_bubbles,):
        raise ValueError("bubble_radii must have shape (N,)")
    if n_bubbles == 0:
        raise ValueError("At least one bubble is required")

    distances = np.linalg.norm(points - observation_point, axis=1)
    if np.any(distances == 0):
        raise ValueError("Bubble cannot be at the observation point")

    depths = transducer_depth + points[:, 2]

    freq_hz = None
    p_far = None
    sigma_bs = None

    df = Get_bubble_fbs(database_dir, gas, depths[0])
    
    for ii in range(n_bubbles):
        freq_ii, f_bs_ii = Get_bubble_ith_spectrum(df, depths[ii], bubble_radii[ii])

        if freq_hz is None:
            freq_hz = freq_ii
            p_far = np.zeros(len(freq_hz), dtype=complex)
            sigma_bs = np.zeros((n_bubbles, len(freq_hz)))
            k = 2 * np.pi * freq_hz / sound_speed

        elif not np.allclose(freq_ii, freq_hz, rtol=0, atol=1e-6):
            raise ValueError("Different frequency grids in bubble databases")

        p_far += f_bs_ii / distances[ii]**2 * np.exp(-2j * k * distances[ii])
        sigma_bs[ii] = np.abs(f_bs_ii)**2

    return freq_hz, p_far, sigma_bs, distances