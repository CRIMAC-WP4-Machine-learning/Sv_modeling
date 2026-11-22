"""Plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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
    # Equal limits alone are not enough: mplot3d's default box aspect is 4:4:3,
    # which squashes z and makes tilted fish look flatter than they are.
    ax.set_box_aspect((1, 1, 1))

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    if invert_z:
        ax.invert_zaxis()
    return ax


def _set_equal_limits(ax, pts, margin=1.0):
    """Equal x/y/z limits around ``pts`` (N, 3) and a cubic box, so angles are not distorted."""
    pts = np.atleast_2d(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = (lo + hi) / 2
    half = margin * max((hi - lo).max() / 2, 1e-9)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _cone_surface(apex, axis, half_angle_deg, h_min, h_max, n_theta=48, n_h=2):
    """X, Y, Z grids of a circular cone surface between distances h_min..h_max along the axis."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = np.linspace(0, 2 * np.pi, n_theta)
    h = np.linspace(h_min, h_max, n_h)[:, None]
    r = h * np.tan(np.deg2rad(half_angle_deg))
    pts = (apex[:, None, None] + axis[:, None, None] * h
           + (u[:, None, None] * np.cos(t) + v[:, None, None] * np.sin(t)) * r)
    return pts[0], pts[1], pts[2]


def plot_school_and_source(
    _points,
    _orientations,
    source,
    source_direction=(0.0, 0.0, -1.0),
    beam_half_angle_deg=None,
    length=0.2,
    ax=None,
    margin=1.05,
):
    """Single 3-D plot zoomed on the fish school, with the echosounder beam.

    The axes are centred on the middle of the school (bounding-box centre), so
    mouse rotation pivots around the school. Axes are equal and cubic, so angles
    are not distorted. The beam axis and beam cone are drawn where they pass
    through the plotted volume. The echosounder itself is normally outside the
    view; its position and range are given in the title. A red arrowhead at the
    end of the beam axis shows the direction of the transmitted sound.

    Parameters
    ----------
    _points, _orientations : ndarray, shape (N, 3)
    source : array_like, shape (3,)
        Echosounder position.
    source_direction : array_like, shape (3,)
        Beam axis. Default straight down (-Z), matching ``place_fish_spheroid``.
    beam_half_angle_deg : float, optional
        Beam half-angle; draws a translucent cone section through the school.
    length : float
        Fish length (m). Each fish is a line of this length centred on its
        position, with a red dot at the head (the orientation direction).
    ax : 3-D Axes, optional
    margin : float
        Scale factor on the axis half-range.

    Returns
    -------
    ax
    """
    source = np.asarray(source, dtype=float)
    d = np.asarray(source_direction, dtype=float)
    d = d / np.linalg.norm(d)
    centre = (_points.min(axis=0) + _points.max(axis=0)) / 2  # middle of the school
    reach = np.dot(centre - source, d)  # distance along the beam to the school centre

    # Half-range: the school, plus the beam cross-section at the school if a cone is drawn
    half = np.abs(_points - centre).max()
    if beam_half_angle_deg is not None and reach > 0:
        half = max(half, reach * np.tan(np.deg2rad(beam_half_angle_deg)))
    half *= margin

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Fish: a line centred on the fish position along its orientation, red dot at the head
    o = _orientations / np.linalg.norm(_orientations, axis=1)[:, None]
    heads = _points + 0.5 * length * o
    tails = _points - 0.5 * length * o
    ax.add_collection3d(Line3DCollection(np.stack([tails, heads], axis=1), colors="C0", linewidths=1.5))
    ax.scatter(heads[:, 0], heads[:, 1], heads[:, 2], s=3, color="red", depthshade=False)
    ax.plot([], [], color="C0", marker="o", markerfacecolor="red", markeredgecolor="red",
            markevery=[1], label="Fish (dot = head)")

    # Beam axis and cone, only the section inside the plotted volume
    h_lo, h_hi = max(reach - half, 0.0), reach + half
    ax.plot(*np.stack([source + d * h_lo, source + d * h_hi]).T, "r--", lw=1, label="Beam axis")
    if beam_half_angle_deg is not None:
        X, Y, Z = _cone_surface(source, d, beam_half_angle_deg, h_lo, h_hi)
        ax.plot_surface(X, Y, Z, color="orange", alpha=0.12, linewidth=0)
        for k in (0, -1):
            ax.plot(X[k], Y[k], Z[k], color="orange", lw=1,
                    label=f"Beam ({beam_half_angle_deg:g}\N{DEGREE SIGN} half-angle)" if k == 0 else None)

    # Arrowhead at the far end of the beam axis: direction of the transmitted sound
    head = 0.12 * half
    ax.quiver(*(source + d * (h_hi - head)), *d, length=head, color="r", lw=2,
              arrow_length_ratio=0.7, normalize=True, label="Sound propagation")
    dist = np.linalg.norm(source - centre)
    src_txt = ", ".join(f"{c:g}" for c in source)
    ax.set_title(f"Echosounder at ({src_txt}) m, {dist:.1f} m from school centre", fontsize=10)

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper left", fontsize=8)
    return ax

def plot_bubble_cloud(points, radii, source=(0, 0, 0), source_direction=(0, 0, 1),
                      beam_half_angle_deg=None, ax=None, margin=1.05,
                      view=(15, -55), invert_z=True, size_scale=2):
    """Plot bubble positions and the acoustic beam in 3D.

    Bubble marker areas are proportional to radius squared, but exaggerated
    for visibility. Physical bubble radii are specified in metres.
    """
    points = np.asarray(points, dtype=float)
    radii = np.asarray(radii, dtype=float)
    source = np.asarray(source, dtype=float)
    d = np.asarray(source_direction, dtype=float)
    d /= np.linalg.norm(d)

    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have shape (N, 3) with N > 0")
    if radii.shape != (len(points),) or np.any(radii <= 0):
        raise ValueError("radii must contain one positive radius per bubble")

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    reach = np.dot(centre - source, d)
    half = np.abs(points - centre).max()

    if beam_half_angle_deg is not None and reach > 0:
        half = max(half, reach * np.tan(np.deg2rad(beam_half_angle_deg)))

    half = max(half * margin, 1e-9)

    # Bubble positions; marker areas scale with physical radius squared
    sizes = size_scale * (radii / radii.min())**2
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    s=sizes, c=radii * 1e3, cmap="BuPu",
                    alpha=0.7, edgecolors="none", depthshade=True)

    # Beam axis
    h_lo, h_hi = max(reach - half, 0.0), reach + half
    beam_line = np.stack([source + d * h_lo, source + d * h_hi])
    ax.plot(beam_line[:, 0], beam_line[:, 1], beam_line[:, 2],
            "r--", lw=1, label="Beam axis")

    # Beam cone
    if beam_half_angle_deg is not None:
        X, Y, Z = _cone_surface(source, d, beam_half_angle_deg, h_lo, h_hi)
        ax.plot_surface(X, Y, Z, color="yellow", alpha=0.10, linewidth=0)
        for k in (0, -1):
            ax.plot(X[k], Y[k], Z[k], color="yellow", lw=1)

    # Sound propagation direction
    arrow_length = 0.12 * half
    ax.quiver(*(source + d * (h_hi - arrow_length)), *d, length=arrow_length,
              color="red", lw=2, arrow_length_ratio=0.7, normalize=True)

    # Equal axis scaling
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1))

    ax.view_init(elev=view[0], azim=view[1])
    if invert_z:
        ax.invert_zaxis()

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth Z (m)")
    ax.set_title(f"Bubble cloud: N={len(points)}, mean depth={points[:, 2].mean():.2f} m")
    ax.figure.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="Bubble radius (mm)")

    return ax