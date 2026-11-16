"""Command-line entry point for planar and spatial robotic-arm simulations."""

from __future__ import annotations

import argparse

import numpy as np

from .forward_kinamatics import forward_kinematics
from .forward_kinematics_3d import forward_kinematics_3d
from .inverse_kinematics import inverse_kinematics
from .inverse_kinematics_3d import inverse_kinematics_3d
from .visualization import plot_arm
from .visualization_3d import plot_arm_3d


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Simulate planar and spatial 3R robotic arms."
    )
    parser.add_argument(
        "--mode",
        choices=("2d", "3d"),
        default="2d",
        help="simulation mode (default: 2d)",
    )
    parser.add_argument(
        "--lengths",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("L1", "L2", "L3"),
        help="three positive link lengths (default: 1 1 1)",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("Q1", "Q2", "Q3"),
        help="three joint angles in radians",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs="+",
        metavar="COORDINATE",
        help="target coordinates: X Y for 2D, or X Y Z for 3D",
    )
    parser.add_argument(
        "--orientation",
        type=float,
        default=0.0,
        help="2D target orientation in radians (default: 0)",
    )
    parser.add_argument(
        "--elbow",
        choices=("up", "down"),
        default="up",
        help="2D inverse-kinematics solution to display (default: up)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="create the plot without opening a GUI window",
    )
    return parser


def run_2d(arguments: argparse.Namespace) -> None:
    """Run the existing planar 3R simulation."""
    joint_angles = np.asarray(arguments.angles, dtype=float)
    target = None

    if arguments.target is not None:
        if len(arguments.target) != 2:
            raise ValueError("2D mode requires --target X Y.")

        target_pose = [*arguments.target, arguments.orientation]
        elbow_up, elbow_down = inverse_kinematics(
            target_pose,
            arguments.lengths,
        )
        joint_angles = elbow_up if arguments.elbow == "up" else elbow_down
        target = arguments.target

    positions, transform = forward_kinematics(
        joint_angles,
        arguments.lengths,
    )

    print("Mode: 2D")
    print("Joint angles (radians):", np.array2string(joint_angles, precision=3))
    print("End-effector position:", np.array2string(positions[-1], precision=3))
    print("End-effector transform:\n", np.array2string(transform, precision=3))

    plot_arm(
        joint_angles,
        arguments.lengths,
        target=target,
        show=not arguments.no_show,
    )


def run_3d(arguments: argparse.Namespace) -> None:
    """Run the spatial 3R simulation."""
    joint_angles = np.asarray(arguments.angles, dtype=float)
    target = None

    if arguments.target is not None:
        if len(arguments.target) != 3:
            raise ValueError("3D mode requires --target X Y Z.")

        target = arguments.target
        joint_angles = inverse_kinematics_3d(
            target,
            arguments.lengths,
        )

    positions, transform = forward_kinematics_3d(
        joint_angles,
        arguments.lengths,
    )

    print("Mode: 3D")
    print("Joint angles (radians):", np.array2string(joint_angles, precision=3))
    print("End-effector position:", np.array2string(positions[-1], precision=3))
    print("End-effector transform:\n", np.array2string(transform, precision=3))

    plot_arm_3d(
        joint_angles,
        arguments.lengths,
        target=target,
        show=not arguments.no_show,
    )


def main() -> None:
    """Run the selected simulator."""
    parser = build_parser()
    arguments = parser.parse_args()

    try:
        if arguments.mode == "3d":
            run_3d(arguments)
        else:
            run_2d(arguments)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()