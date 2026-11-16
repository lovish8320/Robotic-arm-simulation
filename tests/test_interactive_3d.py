import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.interactive_3d import create_interactive_3d


def test_interactive_controls_update_end_effector_position():
    figure, axis, sliders = create_interactive_3d([0.5, 1.0, 0.8])

    assert set(sliders) == {"yaw", "shoulder", "elbow"}
    sliders["yaw"].set_val(np.pi / 2)

    end_effector = axis.lines[0].get_data_3d()
    np.testing.assert_allclose(
        [end_effector[0][-1], end_effector[1][-1], end_effector[2][-1]],
        [0.0, 1.8, 0.5],
        atol=1e-12,
    )
    figure.canvas.draw()
