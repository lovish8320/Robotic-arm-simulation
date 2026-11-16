import numpy as np
import pytest

from src.forward_kinamatics import forward_kinematics
from src.inverse_kinematics import inverse_kinematics


def test_inverse_kinematics_returns_solutions_that_reach_target_pose():
    target_pose = np.array([2.0, 1.0, np.pi / 2])
    link_lengths = [1.0, 1.0, 1.0]

    elbow_up, elbow_down = inverse_kinematics(target_pose, link_lengths)

    for solution in (elbow_up, elbow_down):
        positions, transform = forward_kinematics(solution, link_lengths)

        np.testing.assert_allclose(positions[-1], target_pose[:2], atol=1e-12)
        np.testing.assert_allclose(
            np.arctan2(transform[1, 0], transform[0, 0]),
            target_pose[2],
            atol=1e-12,
        )


def test_inverse_kinematics_returns_distinct_elbow_configurations():
    elbow_up, elbow_down = inverse_kinematics(
        target_pose=[2.0, 1.0, np.pi / 2],
        link_lengths=[1.0, 1.0, 1.0],
    )

    assert elbow_up[1] > 0
    assert elbow_down[1] < 0


def test_inverse_kinematics_rejects_unreachable_target():
    with pytest.raises(ValueError, match="outside the reachable workspace"):
        inverse_kinematics(
            target_pose=[10.0, 0.0, 0.0],
            link_lengths=[1.0, 1.0, 1.0],
        )


@pytest.mark.parametrize(
    ("target_pose", "link_lengths"),
    [
        ([1.0, 2.0], [1.0, 1.0, 1.0]),
        ([1.0, 2.0, 0.0], [1.0, 1.0]),
    ],
)
def test_inverse_kinematics_requires_valid_input_shapes(
    target_pose,
    link_lengths,
):
    with pytest.raises(ValueError):
        inverse_kinematics(target_pose, link_lengths)


@pytest.mark.parametrize(
    "link_lengths",
    [
        [1.0, 0.0, 1.0],
        [1.0, -1.0, 1.0],
    ],
)
def test_inverse_kinematics_requires_positive_link_lengths(link_lengths):
    with pytest.raises(ValueError):
        inverse_kinematics([2.0, 1.0, 0.0], link_lengths)