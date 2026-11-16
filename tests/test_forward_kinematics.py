import numpy as np
import pytest

from src.forward_kinamatics import forward_kinematics


def test_straight_arm_positions_and_transform():
    positions, transform = forward_kinematics([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        positions,
        [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transform,
        [[1.0, 0.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_joint_angles_are_relative_and_orientation_accumulates():
    positions, transform = forward_kinematics(
        [np.pi / 2, 0.0, -np.pi / 2], [2.0, 1.0, 1.0]
    )

    np.testing.assert_allclose(
        positions,
        [[0.0, 0.0], [0.0, 2.0], [0.0, 3.0], [1.0, 3.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(transform[:2, :2], np.eye(2), atol=1e-12)
    np.testing.assert_allclose(transform[:2, 2], positions[-1], atol=1e-12)
    np.testing.assert_allclose(transform[2], [0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    ("angles", "lengths"),
    [([0.0, 0.0], [1.0, 1.0, 1.0]), ([0.0, 0.0, 0.0], [1.0, 1.0])],
)
def test_requires_three_angles_and_three_link_lengths(angles, lengths):
    with pytest.raises(ValueError):
        forward_kinematics(angles, lengths)


@pytest.mark.parametrize("lengths", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
def test_requires_positive_link_lengths(lengths):
    with pytest.raises(ValueError):
        forward_kinematics([0.0, 0.0, 0.0], lengths)
