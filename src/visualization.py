"""Matplotlib visualization utilities for the planar 3R robotic arm."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .forward_kinamatics import forward_kinematics


def plot_arm(
    joint_angles: Sequence[float],
    link_lengths: Sequence[float],
    *,
    target: Sequence[float] | None = None,
    show: bool = True,
):
    """Draw the robotic arm in the XY plane.

    Args:
        joint_angles: Three relative joint angles in radians.
        link_lengths: Three positive link lengths.
        target: Optional XY target point to show on the plot.
        show: Display the figure when ``True``; set to ``False`` in tests.

    Returns:
        The created Matplotlib figure and axes objects.
    """
    positions, _ = forward_kinematics(joint_angles, link_lengths)
    lengths = np.asarray(link_lengths, dtype=float)
    reach = float(np.sum(lengths))

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        "o-",
        color="#1565c0",
        linewidth=3,
        markersize=8,
        label="Arm links",
    )
    axis.scatter(*positions[0], color="#212121", s=65, zorder=3, label="Base")
    axis.scatter(*positions[-1], color="#d32f2f", s=75, zorder=3, label="End effector")

    if target is not None:
        target_point = np.asarray(target, dtype=float)
        if target_point.shape != (2,):
            raise ValueError("Target must contain exactly two coordinates: (x, y).")
        axis.scatter(*target_point, color="#2e7d32", marker="x", s=90, linewidths=3, label="Target")

    margin = max(0.2, reach * 0.1)
    axis.set_xlim(-reach - margin, reach + margin)
    axis.set_ylim(-reach - margin, reach + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X position")
    axis.set_ylabel("Y position")
    axis.set_title("Planar 3R Robotic Arm")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")

    if show:
        plt.show()
    return figure, axis
