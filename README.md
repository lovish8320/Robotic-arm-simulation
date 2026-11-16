# Robotic Arm Simulation

A Python project for simulating a planar three-revolute-joint (3R) robotic arm.

## Setup

Use Python 3.10 or newer. Install the project dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Forward kinematics

`src/forward_kinamatics.py` provides `forward_kinematics(joint_angles, link_lengths)`.
Pass exactly three joint angles in radians and three positive link lengths in any
consistent unit. Angles are relative to the previous link, with the first angle
measured from the positive X axis.

The function returns `(positions, transform)`. `positions` is a 4-by-2 NumPy
array containing the base, both intermediate joints, and end-effector XY
coordinates. `transform` is the 3-by-3 homogeneous transform from the base frame
to the end-effector frame.

Example:

```python
from src.forward_kinamatics import forward_kinematics

positions, transform = forward_kinematics(
    joint_angles=[0.0, 0.0, 0.0],
    link_lengths=[1.0, 1.0, 1.0],
)
print(positions[-1])  # [3. 0.]
print(transform)
```

## Tests

Run the test suite from the repository root with:

```bash
python -m pytest
```

Inverse kinematics, visualization, and the interactive entry point are planned
but are not implemented yet.
