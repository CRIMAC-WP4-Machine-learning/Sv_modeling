"""Lookup in the database of modelled TS(f) and f_bs(f) CSV files."""

import glob
import os
import re

import numpy as np
import pandas as pd
from functools import lru_cache

# Parameters encoded in database filenames, e.g.
# ts_vs_freq_loop_herring_a_0.01000_b_0.00400_..._IncAngle_90_depth_50_length_0.0_iterRef_LU.csv
_FILENAME_PATTERN = (
    r"a_(?P<a>[0-9.]+)_b_(?P<b>[0-9.]+).*?_IncAngle_(?P<IncAngle>[0-9.]+)"
    r"_depth_(?P<depth>[0-9.]+)_length_(?P<length>[0-9.]+)"
)


def Extract_database_metadata(_Dir):
    """Parse metadata from the filenames of all .csv files in _Dir.

    Returns a DataFrame with columns: filename, a, b, IncAngle, depth, length
    (a, b are the prolate-spheroid swimbladder semi-axes).
    """
    csv_files = glob.glob(os.path.join(_Dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found in {_Dir}. The TS database is not stored in git; "
            "see the 'Data' section in README.md."
        )

    records = []
    for file in csv_files:
        match = re.search(_FILENAME_PATTERN, file)
        if match:
            records.append(
                {
                    "filename": os.path.basename(file),
                    "a": float(match.group("a")),
                    "b": float(match.group("b")),
                    "IncAngle": float(match.group("IncAngle")),
                    "depth": float(match.group("depth")),
                    "length": float(match.group("length")),
                }
            )

    return pd.DataFrame(records)


def Find_closestfile_in_database(_df_database, _Depth, _Length, _IncAngl, verbose=False):
    """Select the database entry closest to a fish's depth, length and incident angle.

    Selection order:
      1. closest depth;
      2. smallest length >= _Length (fallback: closest length);
      3. closest incident angle.
    The first row is taken if several match.

    Returns
    -------
    dict
        The selected row (filename, a, b, IncAngle, depth, length, ...).
    """
    df_db = _df_database.copy()

    # Step 1: closest depth
    df_db["depth_diff"] = np.abs(df_db["depth"] - _Depth)
    min_depth_diff = df_db["depth_diff"].min()
    depth_filtered = df_db[df_db["depth_diff"] == min_depth_diff].copy()

    # Step 2: closest length >= _Length
    larger_or_equal = depth_filtered[depth_filtered["length"] >= _Length]
    if not larger_or_equal.empty:
        selected_length = larger_or_equal["length"].min()
        length_filtered = larger_or_equal[larger_or_equal["length"] == selected_length].copy()
    else:
        depth_filtered["length_diff"] = np.abs(depth_filtered["length"] - _Length)
        min_length_diff = depth_filtered["length_diff"].min()
        length_filtered = depth_filtered[depth_filtered["length_diff"] == min_length_diff].copy()

    # Step 3: closest incident angle
    length_filtered["angle_diff"] = np.abs(length_filtered["IncAngle"] - _IncAngl)
    min_angle_diff = length_filtered["angle_diff"].min()
    final_selection = length_filtered[length_filtered["angle_diff"] == min_angle_diff]

    if verbose:
        print("final_selection: ", final_selection)

    return final_selection.iloc[0].to_dict()


def func_get_frq_TS_fbs_Dict(_Dir, _Inp_Dict, _Lfish):
    """Load frequency, TS and complex f_bs for a database entry.

    Parameters
    ----------
    _Dir : path
        Directory with the modelled CSV files.
    _Inp_Dict : dict
        Output of Find_closestfile_in_database.
    _Lfish : float
        Fish length (m). Currently unused: no length scaling is applied.

    Returns
    -------
    freq_vec0 : Series
        Frequency (kHz).
    TS_vec0 : Series
        Target strength (dB).
    Fbs_vec0 : Series
        Complex backscattering amplitude f_bs.
    """
    df_csv = pd.read_csv(os.path.join(_Dir, _Inp_Dict["filename"]))

    freq_vec0 = df_csv["Freq_kHz"]
    TS_vec0 = df_csv["TS"]
    Fbs_vec0 = df_csv["f_bs"]

    # f_bs may be stored as strings such as "1+2j"
    if not np.iscomplexobj(Fbs_vec0):
        Fbs_vec0 = Fbs_vec0.astype(complex)

    return freq_vec0, TS_vec0, Fbs_vec0




def Get_bubble_fbs(_Dir, Gas, depth):
    """
    Load the gas-bubble scattering database covering the requested depth.

    Selects the CSV file with the narrowest depth interval containing
    the requested depth.

    Example
    -------
    Gas = "CO2", depth = 53
    -> CO2_depth50-60m_R0.5-2mm.csv

    Parameters
    ----------
    _Dir : str or Path
        Directory containing the scattering database CSV files.
    Gas : str
        Gas type, e.g. "O2", "CO2", "C2H6".
    depth : float
        Requested bubble depth (m).

    Returns
    -------
    df : pandas.DataFrame
        Scattering database containing depth_m, radius_m,
        frequency_Hz, and complex f_bs.
    """

    from pathlib import Path

    database_dir = Path(_Dir)

    # Find CSV covering the requested depth
    pattern = re.compile(
        rf"{re.escape(Gas)}_depth(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)m_R"
        rf"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)mm",
        re.IGNORECASE
    )

    matches = []

    for filepath in database_dir.glob("*.csv"):
        match = pattern.fullmatch(filepath.stem)

        if match:
            d_min, d_max = map(float, match.groups()[:2])

            if d_min <= depth <= d_max:
                matches.append((d_max - d_min, filepath))

    if not matches:
        raise FileNotFoundError(
            f"No {Gas} database covering depth {depth} m"
        )

    # Select narrowest matching interval
    filepath = min(matches, key=lambda x: x[0])[1]

    # Read CSV
    df = _read_bubble_database(str(filepath))

    return df

def Get_bubble_ith_spectrum(
    df, depth, radius,
    depth_tolerance=0.5,
    radius_tolerance=0.00005
):
    """
    Extract the nearest available bubble spectrum.

    Raises ValueError if the requested depth or radius is
    outside the database range or exceeds the allowed tolerance.
    """

    # Find nearest available depth
    depth_values = np.sort(df["depth_m"].unique())

    # To enforce the documented behavior
    if depth < depth_values[0] or depth > depth_values[-1]:
        raise ValueError(
            f"Requested depth {depth:g} m outside database range "
            f"[{depth_values[0]:g}, {depth_values[-1]:g}] m"
        )
    
    depth_selected = depth_values[np.abs(depth_values - depth).argmin()]

    # Check depth tolerance
    if abs(depth_selected - depth) > depth_tolerance + 1e-9:
        raise ValueError(
            f"Requested depth {depth:g} m: nearest available depth "
            f"{depth_selected:g} m exceeds tolerance {depth_tolerance:g} m"
        )

    df = df[df["depth_m"] == depth_selected]

    # Find nearest available radius
    radius_values = np.sort(df["radius_m"].unique())
    # To enforce the documented behavior
    if radius < radius_values[0] or radius > radius_values[-1]:
        raise ValueError(
            f"Requested radius {radius*1e3:g} mm outside database range "
            f"[{radius_values[0]*1e3:g}, {radius_values[-1]*1e3:g}] mm"
        )

    radius_selected = radius_values[np.abs(radius_values - radius).argmin()]

    # Check radius tolerance
    if abs(radius_selected - radius) > radius_tolerance + 1e-12:
        raise ValueError(
            f"Requested radius {radius*1e3:g} mm: nearest available radius "
            f"{radius_selected*1e3:g} mm exceeds tolerance "
            f"{radius_tolerance*1e3:g} mm"
        )

    df = df[df["radius_m"] == radius_selected]

    # Sort by frequency
    df = df.sort_values("frequency_Hz")

    freq_vec = df["frequency_Hz"].to_numpy(dtype=float)
    f_bs = df["f_bs"].to_numpy(dtype=complex)

    return freq_vec, f_bs

@lru_cache(maxsize=16)
def _read_bubble_database(filepath):
    df = pd.read_csv(filepath)
    df["f_bs"] = df["f_bs"].astype(complex)
    return df