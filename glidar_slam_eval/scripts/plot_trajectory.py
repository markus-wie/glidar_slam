#!/usr/bin/env python3
"""Plot an estimated TUM trajectory against ground truth in the XY plane."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from evaluate_trajectory import align_se2, interpolate, read_tum, transform_se2


def align_and_crop(
    ground_truth: list, raw_estimate: list, alignment_mode: str
) -> tuple[list, list]:
    t_start = max(ground_truth[0].timestamp, raw_estimate[0].timestamp)
    t_end = min(ground_truth[-1].timestamp, raw_estimate[-1].timestamp)

    cropped_gt = [pose for pose in ground_truth if t_start <= pose.timestamp <= t_end]
    cropped_est = [pose for pose in raw_estimate if t_start <= pose.timestamp <= t_end]
    if not cropped_gt or not cropped_est:
        raise ValueError("The trajectories have no overlapping timestamps")

    if alignment_mode == "se2":
        gt_timestamps = [p.timestamp for p in ground_truth]
        matched_gt = []
        matched_est = []
        for pose in cropped_est:
            gt_interp = interpolate(ground_truth, pose.timestamp, gt_timestamps)
            if gt_interp is not None:
                matched_gt.append(gt_interp)
                matched_est.append(pose)
        transform = align_se2(matched_gt, matched_est)
        cropped_est = [transform_se2(pose, transform) for pose in cropped_est]

    return cropped_gt, cropped_est


def plot_trajectory(
    ground_truth_path: Path,
    estimate_path: Path,
    output_path: Path | None,
    alignment_mode: str,
) -> None:
    ground_truth = read_tum(ground_truth_path)
    raw_estimate = read_tum(estimate_path)
    cropped_gt, estimate = align_and_crop(ground_truth, raw_estimate, alignment_mode)

    ground_truth_x = [pose.translation[0] for pose in cropped_gt]
    ground_truth_y = [pose.translation[1] for pose in cropped_gt]
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
        description="Plot TUM trajectory files in the XY plane."
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
        args.ground_truth,
        args.trajectory,
        args.output,
        args.alignment_mode,
    )


if __name__ == "__main__":
    main()
