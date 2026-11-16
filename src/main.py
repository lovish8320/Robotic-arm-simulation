"""Command-line entry point for the planar 3R robotic arm simulator."""

from __future__ import annotations

import argparse

import numpy as np

from .forward_kinamatics import forward_kinematics
from .inverse_kinematics import inverse_kinematics
from .visualization import plot_arm


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface parser."""
    parser = argparse.ArgumentParser(description="Simulate a planar 3R robotic arm.")
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
        help="three relative joint angles in radians",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="solve inverse kinematics for this target position",
    )
    parser.add_argument(
        "--orientation",
        type=float,
        default=0.0,
        help="target orientation in radians; used with --target (default: 0)",
    )
    parser.add_argument(
        "--elbow",
        choices=("up", "down"),
        default="up",
        help="inverse-kinematics solution to display (default: up)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="create the plot without opening a GUI window",
    )
    return parser


def main() -> None:
    """Run the simulator from command-line arguments."""
    arguments = build_parser().parse_args()
    joint_angles = np.asarray(arguments.angles, dtype=float)
    target = None

    if arguments.target is not None:
        target_pose = [*arguments.target, arguments.orientation]
        elbow_up, elbow_down = inverse_kinematics(target_pose, arguments.lengths)
        joint_angles = elbow_up if arguments.elbow == "up" else elbow_down
        target = arguments.target

    positions, transform = forward_kinematics(joint_angles, arguments.lengths)
    print("Joint angles (radians):", np.array2string(joint_angles, precision=3))
    print("End-effector position:", np.array2string(positions[-1], precision=3))
    print("End-effector transform:\n", np.array2string(transform, precision=3))

    plot_arm(joint_angles, arguments.lengths, target=target, show=not arguments.no_show)


if __name__ == "__main__":
    main()
