#!/usr/bin/env python3
"""Evaluate GLiDAR-SLAM TUM trajectories against Bonn LILocBench ground truth."""

import argparse
import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Pose:
    timestamp: float
    translation: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


def normalize_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm == 0.0:
        raise ValueError("Encountered a zero-length quaternion")
    return tuple(component / norm for component in quaternion)  # type: ignore[return-value]


def read_tum(path: Path, allow_empty: bool = False) -> list[Pose]:
    if allow_empty and not path.exists():
        return []
    poses: list[Pose] = []
    with path.open(encoding="ascii") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 TUM fields, found {len(fields)}"
                )
            values = [float(field) for field in fields]
            pose = Pose(
                values[0],
                (values[1], values[2], values[3]),
                normalize_quaternion((values[4], values[5], values[6], values[7])),
            )
            if poses and pose.timestamp <= poses[-1].timestamp:
                raise ValueError(
                    f"{path}:{line_number}: timestamps must be strictly increasing"
                )
            poses.append(pose)
    if not poses and not allow_empty:
        raise ValueError(f"{path}: no poses found")
    return poses


def slerp(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    dot = sum(left * right for left, right in zip(first, second))
    if dot < 0.0:
        second = (-second[0], -second[1], -second[2], -second[3])
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(tuple(left + ratio * (right - left) for left, right in zip(first, second)))  # type: ignore[arg-type]
    angle = math.acos(dot)
    sin_angle = math.sin(angle)
    first_weight = math.sin((1.0 - ratio) * angle) / sin_angle
    second_weight = math.sin(ratio * angle) / sin_angle
    return tuple(first_weight * left + second_weight * right for left, right in zip(first, second))  # type: ignore[return-value]


def interpolate(poses: list[Pose], timestamp: float) -> Pose:
    if timestamp < poses[0].timestamp or timestamp > poses[-1].timestamp:
        raise ValueError("Timestamp outside estimated trajectory range")
    upper_index = bisect.bisect_left([pose.timestamp for pose in poses], timestamp)
    upper = poses[upper_index]
    if upper.timestamp == timestamp or upper_index == 0:
        return Pose(timestamp, upper.translation, upper.quaternion)
    lower = poses[upper_index - 1]
    ratio = (timestamp - lower.timestamp) / (upper.timestamp - lower.timestamp)
    translation = (
        lower.translation[0] + ratio * (upper.translation[0] - lower.translation[0]),
        lower.translation[1] + ratio * (upper.translation[1] - lower.translation[1]),
        lower.translation[2] + ratio * (upper.translation[2] - lower.translation[2]),
    )
    return Pose(
        timestamp, translation, slerp(lower.quaternion, upper.quaternion, ratio)
    )


def yaw(quaternion: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def align_se2(
    ground_truth: list[Pose], estimates: list[Pose]
) -> tuple[float, float, float]:
    gt_x = sum(pose.translation[0] for pose in ground_truth) / len(ground_truth)
    gt_y = sum(pose.translation[1] for pose in ground_truth) / len(ground_truth)
    est_x = sum(pose.translation[0] for pose in estimates) / len(estimates)
    est_y = sum(pose.translation[1] for pose in estimates) / len(estimates)
    cross = 0.0
    dot = 0.0
    for reference, estimate in zip(ground_truth, estimates):
        gx = reference.translation[0] - gt_x
        gy = reference.translation[1] - gt_y
        ex = estimate.translation[0] - est_x
        ey = estimate.translation[1] - est_y
        dot += ex * gx + ey * gy
        cross += ex * gy - ey * gx
    theta = math.atan2(cross, dot)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return (
        theta,
        gt_x - cos_theta * est_x + sin_theta * est_y,
        gt_y - sin_theta * est_x - cos_theta * est_y,
    )


def transform_se2(pose: Pose, transform: tuple[float, float, float]) -> Pose:
    theta, translation_x, translation_y = transform
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x, y, z = pose.translation
    rotated_yaw = yaw(pose.quaternion) + theta
    half_yaw = rotated_yaw / 2.0
    return Pose(
        pose.timestamp,
        (
            cos_theta * x - sin_theta * y + translation_x,
            sin_theta * x + cos_theta * y + translation_y,
            z,
        ),
        (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
    )


def rms(values: Iterable[float]) -> float:
    values = list(values)
    return math.sqrt(sum(value * value for value in values) / len(values))


def calculate_ape(
    ground_truth: list[Pose], estimates: list[Pose]
) -> tuple[list[float], list[float]]:
    translation_errors = []
    rotation_errors = []
    for reference, estimate in zip(ground_truth, estimates):
        dx = reference.translation[0] - estimate.translation[0]
        dy = reference.translation[1] - estimate.translation[1]
        # dz = reference.translation[2] - estimate.translation[2]
        # translation_errors.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        # rotation_errors.append(
        #     abs(wrap_angle(yaw(reference.quaternion) - yaw(estimate.quaternion)))
        # )
        translation_errors.append(math.hypot(dx, dy))
        rotation_errors.append(
            abs(wrap_angle(yaw(reference.quaternion) - yaw(estimate.quaternion)))
        )
    return translation_errors, rotation_errors


def calculate_rpe(
    ground_truth: list[Pose],
    estimates: list[Pose],
    time_step: float,
    distance_step: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    cumulative_distance = [0.0]
    for previous, current in zip(ground_truth, ground_truth[1:]):
        cumulative_distance.append(
            cumulative_distance[-1]
            + math.hypot(
                current.translation[0] - previous.translation[0],
                current.translation[1] - previous.translation[1],
            )
        )

    def errors_for_pairs(
        index_pairs: Iterable[tuple[int, int]],
    ) -> tuple[list[float], list[float]]:
        translation_errors = []
        rotation_errors = []
        for index, later_index in index_pairs:
            reference = ground_truth[index]
            gt_later = ground_truth[later_index]
            estimate = estimates[index]
            estimate_later = estimates[later_index]
            gt_dx = gt_later.translation[0] - reference.translation[0]
            gt_dy = gt_later.translation[1] - reference.translation[1]
            est_dx = estimate_later.translation[0] - estimate.translation[0]
            est_dy = estimate_later.translation[1] - estimate.translation[1]
            translation_errors.append(math.hypot(gt_dx - est_dx, gt_dy - est_dy))
            gt_yaw_delta = wrap_angle(
                yaw(gt_later.quaternion) - yaw(reference.quaternion)
            )
            estimate_yaw_delta = wrap_angle(
                yaw(estimate_later.quaternion) - yaw(estimate.quaternion)
            )
            rotation_errors.append(abs(wrap_angle(gt_yaw_delta - estimate_yaw_delta)))
        return translation_errors, rotation_errors

    time_pairs = []
    distance_pairs = []
    timestamps = [pose.timestamp for pose in ground_truth]
    for index, reference in enumerate(ground_truth):
        later_index = bisect.bisect_left(
            timestamps, reference.timestamp + time_step, lo=index + 1
        )
        if later_index < len(ground_truth):
            time_pairs.append((index, later_index))
        later_index = bisect.bisect_left(
            cumulative_distance,
            cumulative_distance[index] + distance_step,
            lo=index + 1,
        )
        if later_index < len(ground_truth):
            distance_pairs.append((index, later_index))

    time_translation, time_rotation = errors_for_pairs(time_pairs)
    distance_translation, distance_rotation = errors_for_pairs(distance_pairs)
    return time_translation, time_rotation, distance_translation, distance_rotation


def metric_block(values: list[float], unit_scale: float) -> dict[str, float]:
    if not values:
        return {"rmse": float("nan"), "max": float("nan"), "samples": 0}
    return {
        "rmse": rms(values) * unit_scale,
        "max": max(values) * unit_scale,
        "samples": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alignment-mode", choices=("none", "se2"), default="se2")
    parser.add_argument("--rpe-time-step", type=float, default=1.0)
    parser.add_argument("--rpe-distance-step", type=float, default=1.0)
    args = parser.parse_args()

    ground_truth = read_tum(args.ground_truth)
    raw_estimates = read_tum(args.estimate, allow_empty=True)
    if len(raw_estimates) < 2:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "insufficient_data",
            "message": "At least two estimated poses are required for APE/RPE metrics",
            "estimate_samples": len(raw_estimates),
            "ground_truth_overlap_samples": 0,
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="ascii"
        )
        (args.output_dir / "summary.txt").write_text(
            "status: insufficient_data\n"
            "At least two estimated poses are required for APE/RPE metrics.\n",
            encoding="ascii",
        )
        print(
            f"Evaluation incomplete: {args.estimate} contains "
            f"{len(raw_estimates)} estimated poses. Results written to {args.output_dir}"
        )
        return
    matched_ground_truth = [
        pose
        for pose in ground_truth
        if raw_estimates[0].timestamp <= pose.timestamp <= raw_estimates[-1].timestamp
    ]
    if len(matched_ground_truth) < 2:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "insufficient_data",
            "message": "At least two overlapping poses are required for APE/RPE metrics",
            "estimate_samples": len(raw_estimates),
            "ground_truth_overlap_samples": len(matched_ground_truth),
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="ascii"
        )
        (args.output_dir / "summary.txt").write_text(
            "status: insufficient_data\n"
            "At least two overlapping poses are required for APE/RPE metrics.\n",
            encoding="ascii",
        )
        return
    estimates = [
        interpolate(raw_estimates, pose.timestamp) for pose in matched_ground_truth
    ]
    transform = (0.0, 0.0, 0.0)
    if args.alignment_mode == "se2":
        transform = align_se2(matched_ground_truth, estimates)
        estimates = [transform_se2(pose, transform) for pose in estimates]

    ape_translation, ape_rotation = calculate_ape(matched_ground_truth, estimates)
    rpe_translation, rpe_rotation, rpe_distance_translation, rpe_distance_rotation = (
        calculate_rpe(
            matched_ground_truth, estimates, args.rpe_time_step, args.rpe_distance_step
        )
    )
    result = {
        "alignment_mode": args.alignment_mode,
        "alignment_se2": {
            "yaw_rad": transform[0],
            "x_m": transform[1],
            "y_m": transform[2],
        },
        "timestamp_overlap": {
            "start": matched_ground_truth[0].timestamp,
            "end": matched_ground_truth[-1].timestamp,
        },
        "ape": {
            "translation_cm": metric_block(ape_translation, 100.0),
            "rotation_deg": metric_block(ape_rotation, 180.0 / math.pi),
            "translation_below_5cm": sum(error < 0.05 for error in ape_translation)
            / len(ape_translation),
            "translation_below_10cm": sum(error < 0.10 for error in ape_translation)
            / len(ape_translation),
            "rotation_below_1deg": sum(
                error < math.radians(1.0) for error in ape_rotation
            )
            / len(ape_rotation),
            "rotation_below_2deg": sum(
                error < math.radians(2.0) for error in ape_rotation
            )
            / len(ape_rotation),
            "rotation_below_4deg": sum(
                error < math.radians(4.0) for error in ape_rotation
            )
            / len(ape_rotation),
        },
        "rpe_1s": {
            "translation_cm": metric_block(rpe_translation, 100.0),
            "rotation_deg": metric_block(rpe_rotation, 180.0 / math.pi),
        },
        "rpe_1m": {
            "translation_cm": metric_block(rpe_distance_translation, 100.0),
            "rotation_deg": metric_block(rpe_distance_rotation, 180.0 / math.pi),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    summary = [
        "metric                 rmse       max       samples",
        f"APE translation [cm]   {result['ape']['translation_cm']['rmse']:.3f}      {result['ape']['translation_cm']['max']:.3f}      {len(ape_translation)}",
        f"APE rotation [deg]     {result['ape']['rotation_deg']['rmse']:.3f}      {result['ape']['rotation_deg']['max']:.3f}      {len(ape_rotation)}",
        "",
        "Tracking Thresholds (matches Paper Table III):",
        f"Translation < 5cm:     {result['ape']['translation_below_5cm'] * 100:.2f}%",
        f"Translation < 10cm:    {result['ape']['translation_below_10cm'] * 100:.2f}%",
        f"Rotation < 1 deg:      {result['ape']['rotation_below_1deg'] * 100:.2f}%",
        f"Rotation < 2 deg:      {result['ape']['rotation_below_2deg'] * 100:.2f}%",
        f"Rotation < 4 deg:      {result['ape']['rotation_below_4deg'] * 100:.2f}%",
        "",
        f"RPE 1s translation [cm] {result['rpe_1s']['translation_cm']['rmse']:.3f}      {result['rpe_1s']['translation_cm']['max']:.3f}      {len(rpe_translation)}",
        f"RPE 1s rotation [deg]  {result['rpe_1s']['rotation_deg']['rmse']:.3f}      {result['rpe_1s']['rotation_deg']['max']:.3f}      {len(rpe_rotation)}",
        f"RPE 1m translation [cm] {result['rpe_1m']['translation_cm']['rmse']:.3f}      {result['rpe_1m']['translation_cm']['max']:.3f}      {len(rpe_distance_translation)}",
        f"RPE 1m rotation [deg]  {result['rpe_1m']['rotation_deg']['rmse']:.3f}      {result['rpe_1m']['rotation_deg']['max']:.3f}      {len(rpe_distance_rotation)}",
    ]
    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="ascii"
    )
    print("\n".join(summary))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
