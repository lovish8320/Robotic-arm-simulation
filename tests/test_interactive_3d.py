import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.forward_kinematics_3d import forward_kinematics_3d
from src.interactive_3d import create_interactive_3d


def test_interactive_plot_labels_initial_end_effector_position():
    figure, axis, _ = create_interactive_3d([0.5, 1.0, 0.8])

    assert len(axis.texts) == 1
    assert axis.texts[0].get_text() == "  End effector\n  (1.800, 0.000, 0.500)"
    figure.canvas.draw()


def test_interactive_controls_update_end_effector_position_and_label():
    figure, axis, sliders = create_interactive_3d([0.5, 1.0, 0.8])

    assert set(sliders) == {"yaw", "shoulder", "elbow"}
    sliders["yaw"].set_val(np.pi / 2)

    end_effector = axis.lines[0].get_data_3d()
    np.testing.assert_allclose(
        [end_effector[0][-1], end_effector[1][-1], end_effector[2][-1]],
        [0.0, 1.8, 0.5],
        atol=1e-12,
    )
    assert axis.texts[0].get_text() == "  End effector\n  (0.000, 1.800, 0.500)"
    figure.canvas.draw()


def test_interactive_label_matches_kinematics_after_multiple_slider_changes():
    link_lengths = [0.5, 1.0, 0.8]
    figure, axis, sliders = create_interactive_3d(link_lengths)
    joint_angles = [0.4, -0.3, 0.5]

    sliders["yaw"].set_val(joint_angles[0])
    sliders["shoulder"].set_val(joint_angles[1])
    sliders["elbow"].set_val(joint_angles[2])

    positions, _ = forward_kinematics_3d(joint_angles, link_lengths)
    expected_label = (
        "  End effector\n"
        f"  ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})"
    )

    assert len(axis.texts) == 1
    assert axis.texts[0].get_text() == expected_label
    figure.canvas.draw()
