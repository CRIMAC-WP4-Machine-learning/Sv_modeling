"""Reader for FEM/BEM (COMSOL) target-strength exports."""

import os

import numpy as np


def func_FEM_BEM(_Dir, _FileName):
    """Read frequency and TS from a FEM/BEM text export.

    The first 5 lines are header. Column 2 is frequency (Hz) and
    column 3 is TS (TS_Global_var_Point, dB).

    Returns
    -------
    freq_bem : ndarray
        Frequency (Hz).
    TS_bem : ndarray
        Target strength (dB).
    """
    File = os.path.join(_Dir, _FileName)
    with open(File, mode="r") as file:
        lines = file.readlines()[5:]

    freq_bem = []
    TS_bem = []
    for row in lines:
        split_row = row.split()
        freq_bem.append(float(split_row[1]))
        TS_bem.append(float(split_row[2]))

    return np.array(freq_bem), np.array(TS_bem)
