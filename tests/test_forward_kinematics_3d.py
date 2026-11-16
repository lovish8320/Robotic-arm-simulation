import numpy as np
import pytest

from src.forward_kinematics_3d import forward_kinematics_3d


def test_home_pose_places_arm_on_x_axis_above_base():
    positions, transform = forward_kinematics_3d(
        joint_angles=[0.0, 0.0, 0.0],
        link_lengths=[0.5, 1.0, 0.8],
    )

    np.testing.assert_allclose(
        positions,
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.8, 0.0, 0.5]],
        atol=1e-12,
    )
    np.testing.assert_allclose(transform[:3, 3], [1.8, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)


def test_base_yaw_rotates_arm_into_y_axis():
    positions, transform = forward_kinematics_3d(
        joint_angles=[np.pi / 2, 0.0, 0.0],
        link_lengths=[0.5, 1.0, 0.8],
    )

    np.testing.assert_allclose(positions[-1], [0.0, 1.8, 0.5], atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], positions[-1], atol=1e-12)


def test_shoulder_pitch_changes_end_effector_height():
    positions, _ = forward_kinematics_3d(
        joint_angles=[0.0, np.pi / 2, 0.0],
        link_lengths=[0.5, 1.0, 0.8],
    )

    np.testing.assert_allclose(positions[-1], [0.0, 0.0, -1.3], atol=1e-12)


@pytest.mark.parametrize(
    ("angles", "lengths"),
    [([0.0, 0.0], [1.0, 1.0, 1.0]), ([0.0, 0.0, 0.0], [1.0, 1.0])],
)
def test_requires_three_angles_and_link_lengths(angles, lengths):
    with pytest.raises(ValueError):
        forward_kinematics_3d(angles, lengths)


@pytest.mark.parametrize("lengths", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0]])
def test_requires_positive_link_lengths(lengths):
    with pytest.raises(ValueError):
        forward_kinematics_3d([0.0, 0.0, 0.0], lengths)
