"""Matplotlib visualization utilities for the spatial 3R robotic arm."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .forward_kinematics_3d import forward_kinematics_3d


def plot_arm_3d(
    joint_angles: Sequence[float],
    link_lengths: Sequence[float],
    *,
    target: Sequence[float] | None = None,
    show: bool = True,
):
    """Draw the spatial 3R arm in an XYZ Matplotlib plot.

    Args:
        joint_angles: Base yaw, shoulder pitch, and elbow pitch in radians.
        link_lengths: Positive base-height, upper-arm, and forearm lengths.
        target: Optional XYZ target point to show on the plot.
        show: Display the figure when ``True``; set to ``False`` in tests.

    Returns:
        The created Matplotlib figure and 3D axes objects.
    """
    positions, _ = forward_kinematics_3d(joint_angles, link_lengths)
    target_point = _validate_target(target)

    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(projection="3d")
    axis.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "o-",
        color="#1565c0",
        linewidth=3,
        markersize=8,
        label="Arm links",
    )
    axis.scatter(*positions[0], color="#212121", s=65, label="Base")
    axis.scatter(*positions[-1], color="#d32f2f", s=75, label="End effector")

    plot_points = positions
    if target_point is not None:
        axis.scatter(
            *target_point,
            color="#2e7d32",
            marker="x",
            s=90,
            linewidths=3,
            label="Target",
        )
        plot_points = np.vstack((plot_points, target_point))

    _set_equal_limits(axis, plot_points, link_lengths)
    axis.set_xlabel("X position")
    axis.set_ylabel("Y position")
    axis.set_zlabel("Z position")
    axis.set_title("Spatial 3R Robotic Arm")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")

    if show:
        plt.show()
    return figure, axis


def _validate_target(target: Sequence[float] | None) -> np.ndarray | None:
    """Return a validated XYZ target point when one was supplied."""
    if target is None:
        return None

    target_point = np.asarray(target, dtype=float)
    if target_point.shape != (3,):
        raise ValueError("Target must contain exactly three coordinates: (x, y, z).")
    return target_point


def _set_equal_limits(axis, points: np.ndarray, link_lengths: Sequence[float]) -> None:
    """Set XYZ limits with equal numerical spans around the displayed objects."""
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = (lower + upper) / 2
    maximum_span = float(np.max(upper - lower))
    margin = max(0.2, float(np.sum(link_lengths)) * 0.1)
    half_span = max(maximum_span / 2 + margin, margin)

    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1, 1, 1))
