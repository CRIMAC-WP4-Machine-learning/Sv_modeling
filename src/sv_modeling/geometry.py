"""Geometry helpers for fish position and orientation."""

import numpy as np


def Get_Angle(_p0, _p1, _u_p1):
    """Incident angle between the fish axis and the direction to the echosounder.

    Computes the angle between the vector from p1 to p0 and the orientation
    vector u_p1 at p1, folded into [0, 90] degrees (0 = end-on, 90 = broadside).

    Parameters
    ----------
    _p0 : array_like, shape (3,)
        Observation point (echosounder position).
    _p1 : array_like, shape (3,)
        Fish position.
    _u_p1 : array_like, shape (3,)
        Fish orientation (need not be normalised).

    Example
    -------
    >>> Get_Angle(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    90.0
    """
    _u_p1_norm = np.linalg.norm(_u_p1)

    v = _p0 - _p1
    v_norm = np.linalg.norm(v)

    if v_norm > 0:
        cos_theta = np.dot(_u_p1, v) / (v_norm * _u_p1_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
        theta_deg = np.degrees(np.arccos(cos_theta))
    else:
        print("Warning: p0 and p1 are the same point.")

    Incident_Angle_deg = 90 - np.abs(90 - theta_deg)
    return Incident_Angle_deg
