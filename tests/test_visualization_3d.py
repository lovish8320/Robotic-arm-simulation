import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.visualization_3d import plot_arm_3d


def test_plot_arm_3d_draws_labeled_3d_plot_with_target():
    figure, axis = plot_arm_3d(
        joint_angles=[0.0, 0.0, 0.0],
        link_lengths=[0.5, 1.0, 0.8],
        target=[1.0, 0.5, 1.0],
        show=False,
    )

    assert axis.name == "3d"
    assert axis.get_xlabel() == "X position"
    assert axis.get_ylabel() == "Y position"
    assert axis.get_zlabel() == "Z position"
    assert len(axis.lines) == 1
    assert len(axis.collections) == 3

    spans = np.array(
        [
            np.diff(axis.get_xlim())[0],
            np.diff(axis.get_ylim())[0],
            np.diff(axis.get_zlim())[0],
        ]
    )
    np.testing.assert_allclose(spans, spans[0])
    figure.canvas.draw()


def test_plot_arm_3d_requires_three_coordinate_target():
    with pytest.raises(ValueError, match="exactly three coordinates"):
        plot_arm_3d(
            joint_angles=[0.0, 0.0, 0.0],
            link_lengths=[0.5, 1.0, 0.8],
            target=[1.0, 2.0],
            show=False,
        )
