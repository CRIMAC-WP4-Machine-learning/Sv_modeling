"""Lookup in the database of modelled TS(f) and f_bs(f) CSV files."""

import glob
import os
import re

import numpy as np
import pandas as pd

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
