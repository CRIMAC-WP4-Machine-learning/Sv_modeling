"""Plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np


def plot_fish_school(
    _points,
    _orientations,
    ax=None,
    length=0.2,
    margin=1.0,
    view=None,
    invert_z=False,
):
    """3-D quiver plot of fish positions and orientations with equal aspect ratio.

    Parameters
    ----------
    _points, _orientations : ndarray, shape (N, 3)
    ax : 3-D Axes, optional
        Axes to draw in. A new figure is created if None.
    length : float
        Arrow length.
    margin : float
        Scale factor on the half-range of the axis limits.
    view : (elev, azim), optional
        Passed to ax.view_init.
    invert_z : bool
        Invert the z axis (use when +Z points downward).
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.quiver(
        _points[:, 0], _points[:, 1], _points[:, 2],
        _orientations[:, 0], _orientations[:, 1], _orientations[:, 2],
        length=length, normalize=True,
    )

    max_range = np.array([
        _points[:, 0].max() - _points[:, 0].min(),
        _points[:, 1].max() - _points[:, 1].min(),
        _points[:, 2].max() - _points[:, 2].min(),
    ]).max() / 2.0

    mid_x = (_points[:, 0].max() + _points[:, 0].min()) * 0.5
    mid_y = (_points[:, 1].max() + _points[:, 1].min()) * 0.5
    mid_z = (_points[:, 2].max() + _points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - margin * max_range, mid_x + margin * max_range)
    ax.set_ylim(mid_y - margin * max_range, mid_y + margin * max_range)
    ax.set_zlim(mid_z - margin * max_range, mid_z + margin * max_range)

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    if invert_z:
        ax.invert_zaxis()
    return ax
