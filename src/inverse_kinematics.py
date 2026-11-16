"""Inverse kinematics for a planar three-revolute-joint robotic arm."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def inverse_kinematics(
    target_pose: Sequence[float],
    link_lengths: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate elbow-up and elbow-down joint-angle solutions for a planar 3R arm.

    Args:
        target_pose: Desired end-effector pose as ``(x, y, orientation)``, where
            orientation is in radians.
        link_lengths: Three positive link lengths.

    Returns:
        A tuple containing the elbow-up and elbow-down joint-angle solutions.
        Each solution is a NumPy array of shape ``(3,)`` in radians.

    Raises:
        ValueError: If inputs do not describe a 3R arm or the target pose is
            outside the reachable workspace.
    """
    pose = np.asarray(target_pose, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)

    if pose.shape != (3,):
        raise ValueError("Target pose must contain x, y, and orientation.")
    if lengths.shape != (3,):
        raise ValueError("A planar 3R arm requires exactly three link lengths.")
    if np.any(lengths <= 0):
        raise ValueError("All link lengths must be positive.")

    x, y, orientation = pose
    link_1, link_2, link_3 = lengths

    wrist_x = x - link_3 * np.cos(orientation)
    wrist_y = y - link_3 * np.sin(orientation)

    cosine_joint_2 = (
        wrist_x**2 + wrist_y**2 - link_1**2 - link_2**2
    ) / (2 * link_1 * link_2)

    tolerance = 1e-12
    if cosine_joint_2 < -1 - tolerance or cosine_joint_2 > 1 + tolerance:
        raise ValueError("Target pose is outside the reachable workspace.")

    cosine_joint_2 = np.clip(cosine_joint_2, -1.0, 1.0)
    joint_2_magnitude = np.arccos(cosine_joint_2)

    elbow_up = _solve_joint_angles(
        wrist_x,
        wrist_y,
        orientation,
        link_1,
        link_2,
        joint_2_magnitude,
    )
    elbow_down = _solve_joint_angles(
        wrist_x,
        wrist_y,
        orientation,
        link_1,
        link_2,
        -joint_2_magnitude,
    )

    return elbow_up, elbow_down


def _solve_joint_angles(
    wrist_x: float,
    wrist_y: float,
    orientation: float,
    link_1: float,
    link_2: float,
    joint_2: float,
) -> np.ndarray:
    """Calculate one 3R solution for a fixed second-joint angle."""
    joint_1 = np.arctan2(wrist_y, wrist_x) - np.arctan2(
        link_2 * np.sin(joint_2),
        link_1 + link_2 * np.cos(joint_2),
    )
    joint_3 = orientation - joint_1 - joint_2

    return np.array([joint_1, joint_2, joint_3])