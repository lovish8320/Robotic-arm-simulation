# Robotic Arm Simulation

A Python simulator for planar and spatial three-revolute-joint (3R) robotic
arms. It provides forward kinematics, inverse kinematics, joint-limit-aware
3D target solving, and Matplotlib visualizations.

## Features

- **2D planar arm:** analytical forward and inverse kinematics.
- **3D spatial arm:** yaw-pitch-pitch forward kinematics with 4x4 homogeneous
  transformations.
- **3D inverse kinematics:** numerical XYZ target solving with configurable
  joint limits.
- **Visualization:** 2D and 3D arm plots, with optional target markers.
- **Tests:** coverage for kinematics, numerical solving, and 3D plotting.

## Setup

Use Python 3.10 or newer. From the repository root, create an optional virtual
environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Visualize the 3D arm

Run the spatial arm at its home pose:

```powershell
python -m src.main --mode 3d
```

Solve and display a reachable XYZ target:

```powershell
python -m src.main --mode 3d --lengths 0.5 1.0 0.8 --target 1.2 0.8 1.0
```

The program prints the solved joint angles and end-effector transform, then
opens an interactive Matplotlib 3D window. Close that window to finish the
program. Add `--no-show` when running without a display or in automated checks.

## 3D arm model

The spatial 3R arm has three joints:

1. Base yaw around the global Z axis.
2. Shoulder pitch around its local Y axis.
3. Elbow pitch around its local Y axis.

In 3D mode, `--lengths` means:

```text
base_height upper_arm_length forearm_length
```

`src/forward_kinematics_3d.py` returns base, shoulder, elbow, and end-effector
XYZ positions plus the final 4x4 homogeneous transform.

## Use the planar 2D arm

Visualize a direct joint-angle configuration:

```powershell
python -m src.main --mode 2d --angles 0.5 -0.4 0.7
```

Solve a 2D target with a chosen orientation:

```powershell
python -m src.main --mode 2d --target 1.5 1.0 --orientation 1.5708 --elbow up
```

## Run tests

```powershell
python -m pytest
```

## Project layout

```text
src/
  forward_kinamatics.py      # Planar forward kinematics (legacy filename)
  inverse_kinematics.py      # Planar analytical inverse kinematics
  visualization.py           # Planar plot
  forward_kinematics_3d.py   # Spatial forward kinematics
  inverse_kinematics_3d.py   # Spatial numerical inverse kinematics
  visualization_3d.py        # Spatial plot
  main.py                    # Command-line entry point
tests/                       # Automated tests
report/                      # Project report material
```
