"""Forward kinematics for a spatial three-revolute-joint robotic arm.

The arm uses a yaw-pitch-pitch configuration:

* joint 1 rotates the arm about the global Z axis;
* joint 2 rotates the shoulder about its local Y axis;
* joint 3 rotates the elbow about its local Y axis.

``link_lengths`` contains ``(base_height, upper_arm, forearm)``. Angles are
in radians and lengths use any consistent unit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def forward_kinematics_3d(
    joint_angles: Sequence[float], link_lengths: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate spatial joint positions and the end-effector transform.

    Args:
        joint_angles: Base yaw, shoulder pitch, and elbow pitch in radians.
        link_lengths: Positive ``(base_height, upper_arm, forearm)`` lengths.

    Returns:
        A tuple of a ``(4, 3)`` array for base, shoulder, elbow, and end
        effector positions, and the ``(4, 4)`` base-to-end-effector transform.
    """
    angles = np.asarray(joint_angles, dtype=float)
    lengths = np.asarray(link_lengths, dtype=float)

    if angles.shape != (3,) or lengths.shape != (3,):
        raise ValueError("A spatial 3R arm requires exactly three angles and three link lengths.")
    if np.any(lengths <= 0):
        raise ValueError("All link lengths must be positive.")

    base_height, upper_arm, forearm = lengths
    base_yaw, shoulder_pitch, elbow_pitch = angles

    transform = np.eye(4)
    positions = [transform[:3, 3].copy()]

    transform = transform @ _rotation_z(base_yaw) @ _translation(0.0, 0.0, base_height)
    positions.append(transform[:3, 3].copy())

    transform = transform @ _rotation_y(shoulder_pitch) @ _translation(upper_arm, 0.0, 0.0)
    positions.append(transform[:3, 3].copy())

    transform = transform @ _rotation_y(elbow_pitch) @ _translation(forearm, 0.0, 0.0)
    positions.append(transform[:3, 3].copy())

    return np.asarray(positions), transform


def _rotation_z(angle: float) -> np.ndarray:
    """Return a 4x4 homogeneous rotation about Z."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _rotation_y(angle: float) -> np.ndarray:
    """Return a 4x4 homogeneous rotation about Y."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sine, 0.0, cosine, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _translation(x: float, y: float, z: float) -> np.ndarray:
    """Return a 4x4 homogeneous translation."""
    return np.array(
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
