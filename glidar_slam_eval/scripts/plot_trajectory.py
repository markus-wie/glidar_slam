#!/usr/bin/env python3
"""Plot an estimated TUM trajectory against ground truth in the XY plane."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from evaluate_trajectory import align_se2, interpolate, read_tum, transform_se2


def plot_trajectory(
    ground_truth_path: Path,
    estimate_path: Path,
    output_path: Path | None,
    alignment_mode: str,
) -> None:
    ground_truth = read_tum(ground_truth_path)
    raw_estimate = read_tum(estimate_path)

    if alignment_mode == "se2":
        ground_truth = [
            pose
            for pose in ground_truth
            if raw_estimate[0].timestamp <= pose.timestamp <= raw_estimate[-1].timestamp
        ]
        if not ground_truth:
            raise ValueError("The trajectories have no overlapping timestamps")
        estimate = [interpolate(raw_estimate, pose.timestamp) for pose in ground_truth]
        transform = align_se2(ground_truth, estimate)
        estimate = [transform_se2(pose, transform) for pose in estimate]
    else:
        estimate = raw_estimate

    ground_truth_x = [pose.translation[0] for pose in ground_truth]
    ground_truth_y = [pose.translation[1] for pose in ground_truth]
    estimate_x = [pose.translation[0] for pose in estimate]
    estimate_y = [pose.translation[1] for pose in estimate]

    figure, axes = plt.subplots(figsize=(10, 8))
    axes.plot(
        ground_truth_x,
        ground_truth_y,
        color="tab:blue",
        label="Ground truth",
        linewidth=2,
    )
    axes.plot(
        estimate_x,
        estimate_y,
        color="tab:orange",
        label="Estimate",
        linewidth=2,
    )
    axes.scatter(
        ground_truth_x[0],
        ground_truth_y[0],
        color="tab:blue",
        marker=MarkerStyle("o"),
        s=60,
        label="Ground truth start",
    )
    axes.scatter(
        estimate_x[0],
        estimate_y[0],
        color="tab:orange",
        marker=MarkerStyle("o"),
        s=60,
        label="Estimate start",
    )
    axes.set_title("GLiDAR-SLAM trajectory comparison")
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_aspect("equal", adjustable="datalim")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    if output_path is None:
        plt.show()
    else:
        figure.savefig(str(output_path), dpi=150)
        print(f"Saved trajectory plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot two TUM trajectory files in the XY plane."
    )
    parser.add_argument(
        "trajectory", type=Path, help="Estimated trajectory (.tum) path"
    )
    parser.add_argument(
        "ground_truth", type=Path, help="Ground-truth TUM trajectory path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the plot to this image path instead of opening a window",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("none", "se2"),
        default="se2",
        help="Align the estimate to ground truth before plotting (default: se2)",
    )
    args = parser.parse_args()
    plot_trajectory(
        args.ground_truth, args.trajectory, args.output, args.alignment_mode
    )


if __name__ == "__main__":
    main()
