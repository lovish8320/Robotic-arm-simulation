import numpy as np
import pytest

from src.forward_kinematics_3d import forward_kinematics_3d
from src.inverse_kinematics_3d import inverse_kinematics_3d


def test_solver_reaches_known_3d_target():
    link_lengths = [0.5, 1.0, 0.8]
    expected_angles = [0.5, -0.4, 0.7]

    positions, _ = forward_kinematics_3d(expected_angles, link_lengths)
    target = positions[-1]

    solution = inverse_kinematics_3d(target, link_lengths)
    solved_positions, _ = forward_kinematics_3d(solution, link_lengths)

    np.testing.assert_allclose(
        solved_positions[-1],
        target,
        atol=1e-6,
    )


def test_solver_solution_obeys_joint_limits():
    link_lengths = [0.5, 1.0, 0.8]
    target = [1.2, 0.8, 1.0]
    joint_limits = [
        [-1.0, 1.0],
        [-1.2, 0.2],
        [-1.5, 1.5],
    ]

    solution = inverse_kinematics_3d(
        target,
        link_lengths,
        joint_limits=joint_limits,
    )

    lower_bounds = np.array(joint_limits)[:, 0]
    upper_bounds = np.array(joint_limits)[:, 1]

    assert np.all(solution >= lower_bounds)
    assert np.all(solution <= upper_bounds)


def test_solver_rejects_unreachable_target():
    with pytest.raises(ValueError, match="unreachable"):
        inverse_kinematics_3d(
            target=[10.0, 10.0, 10.0],
            link_lengths=[0.5, 1.0, 0.8],
        )


def test_solver_rejects_target_outside_joint_limits():
    with pytest.raises(ValueError, match="unreachable"):
        inverse_kinematics_3d(
            target=[0.0, 1.5, 0.5],
            link_lengths=[0.5, 1.0, 0.8],
            joint_limits=[
                [-0.1, 0.1],
                [-np.pi / 2, np.pi / 2],
                [-np.pi, np.pi],
            ],
        )


@pytest.mark.parametrize(
    "target",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
    ],
)
def test_solver_requires_xyz_target(target):
    with pytest.raises(ValueError, match="exactly three coordinates"):
        inverse_kinematics_3d(
            target=target,
            link_lengths=[0.5, 1.0, 0.8],
        )