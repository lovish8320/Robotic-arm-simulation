"""Numerical inverse kinematics for the spatial 3R robotic arm."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares

from .forward_kinematics_3d import forward_kinematics_3d

DEFAULT_JOINT_LIMITS = np.array(
    [
        [-np.pi, np.pi],        # Base yaw
        [-np.pi / 2, np.pi / 2],  # Shoulder pitch
        [-np.pi, np.pi],        # Elbow pitch
    ]
)


def inverse_kinematics_3d(
    target: Sequence[float],
    link_lengths: Sequence[float],
    *,
    joint_limits: Sequence[Sequence[float]] = DEFAULT_JOINT_LIMITS,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Solve joint angles for an XYZ end-effector target.

    Args:
        target: Desired end-effector position as ``(x, y, z)``.
        link_lengths: Base height, upper-arm, and forearm lengths.
        joint_limits: Lower and upper limits for each joint in radians.
        tolerance: Maximum accepted Euclidean position error.

    Returns:
        Three joint angles in radians: base yaw, shoulder pitch, and elbow pitch.

    Raises:
        ValueError: If the target or limits are invalid, or if no solution within
            the joint limits can reach the target.
    """
    target_position = np.asarray(target, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)

    if target_position.shape != (3,):
        raise ValueError("Target must contain exactly three coordinates: (x, y, z).")
    if not np.all(np.isfinite(target_position)):
        raise ValueError("Target coordinates must be finite.")
    if limits.shape != (3, 2):
        raise ValueError("Joint limits must contain lower and upper bounds for three joints.")

    lower_bounds = limits[:, 0]
    upper_bounds = limits[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Each joint lower limit must be less than its upper limit.")

    initial_guesses = _initial_guesses(
        target_position,
        link_lengths,
        lower_bounds,
        upper_bounds,
    )

    best_result = None
    for guess in initial_guesses:
        result = least_squares(
            _position_error,
            x0=guess,
            bounds=(lower_bounds, upper_bounds),
            args=(target_position, link_lengths),
        )

        if best_result is None or result.cost < best_result.cost:
            best_result = result

    assert best_result is not None

    solution = best_result.x
    error = np.linalg.norm(
        _position_error(solution, target_position, link_lengths)
    )

    if not best_result.success or error > tolerance:
        raise ValueError(
            "Target is unreachable within the configured joint limits. "
            f"Closest position error: {error:.6f}."
        )

    return solution


def _position_error(
    joint_angles: np.ndarray,
    target: np.ndarray,
    link_lengths: Sequence[float],
) -> np.ndarray:
    """Return XYZ end-effector error for the numerical solver."""
    positions, _ = forward_kinematics_3d(joint_angles, link_lengths)
    return positions[-1] - target


def _initial_guesses(
    target: np.ndarray,
    link_lengths: Sequence[float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> list[np.ndarray]:
    """Create bounded starting points for numerical IK."""
    base_height, upper_arm, forearm = np.asarray(link_lengths, dtype=float)
    x, y, z = target

    yaw = np.arctan2(y, x)
    radial_distance = np.hypot(x, y)
    vertical_distance = z - base_height

    cosine_elbow = (
        radial_distance**2
        + vertical_distance**2
        - upper_arm**2
        - forearm**2
    ) / (2 * upper_arm * forearm)

    guesses = [
        np.array([yaw, 0.0, 0.0]),
        (lower_bounds + upper_bounds) / 2,
    ]

    if -1.0 <= cosine_elbow <= 1.0:
        elbow_magnitude = np.arccos(cosine_elbow)

        for elbow_pitch in (elbow_magnitude, -elbow_magnitude):
            shoulder_pitch = np.arctan2(
                -vertical_distance,
                radial_distance,
            ) - np.arctan2(
                forearm * np.sin(elbow_pitch),
                upper_arm + forearm * np.cos(elbow_pitch),
            )
            guesses.append(
                np.array([yaw, shoulder_pitch, elbow_pitch])
            )

    return [
        _clip_inside_bounds(guess, lower_bounds, upper_bounds)
        for guess in guesses
    ]


def _clip_inside_bounds(
    values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Keep least-squares initial values strictly inside its bounds."""
    margin = 1e-9
    return np.clip(values, lower_bounds + margin, upper_bounds - margin)