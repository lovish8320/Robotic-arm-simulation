"""Forward kinematics for a planar three-revolute-joint robotic arm.

The module name follows the repository's existing spelling. Angles are given
in radians and link lengths use any consistent linear unit (for example,
metres).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def forward_kinematics(
    joint_angles: Sequence[float], link_lengths: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate all joint positions and the end-effector pose.

    Args:
        joint_angles: Three joint angles, in radians.
        link_lengths: Three positive link lengths.

    Returns:
        A tuple containing:
        - a ``(4, 2)`` array of base, joint, and end-effector XY positions;
        - a ``(3, 3)`` homogeneous transform from base to end effector.
    """
    angles = np.asarray(joint_angles, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)

    if angles.shape != (3,) or lengths.shape != (3,):
        raise ValueError("A planar 3R arm requires exactly three angles and three link lengths.")
    if np.any(lengths <= 0):
        raise ValueError("All link lengths must be positive.")

    cumulative_angles = np.cumsum(angles)
    x_steps = lengths * np.cos(cumulative_angles)
    y_steps = lengths * np.sin(cumulative_angles)
    positions = np.column_stack((np.r_[0.0, np.cumsum(x_steps)], np.r_[0.0, np.cumsum(y_steps)]))

    orientation = cumulative_angles[-1]
    end_effector_transform = np.array(
        [
            [np.cos(orientation), -np.sin(orientation), positions[-1, 0]],
            [np.sin(orientation), np.cos(orientation), positions[-1, 1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return positions, end_effector_transform
