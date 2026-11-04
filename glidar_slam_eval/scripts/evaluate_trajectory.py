#!/usr/bin/env python3
"""Evaluate GLiDAR-SLAM TUM trajectories against ground truth."""

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
    quaternion: tuple[float, float, float, float]  # (x, y, z, w)


def normalize_quaternion(
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm == 0.0:
        raise ValueError("Encountered a zero-length quaternion")
    return tuple(component / norm for component in quaternion)  # type: ignore[return-value]


def multiply_quaternions(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def conjugate_quaternion(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def so3_angle_error(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> float:
    """Geodesic angle on SO(3) between two unit quaternions in radians."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


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


def interpolate(
    poses: list[Pose],
    timestamp: float,
    timestamps: list[float] | None = None,
    max_gap: float | None = None,
) -> Pose | None:
    """Interpolate pose at `timestamp`. Returns None if spanning a gap > max_gap."""
    if timestamp < poses[0].timestamp or timestamp > poses[-1].timestamp:
        return None
    if timestamps is None:
        timestamps = [pose.timestamp for pose in poses]
    upper_index = bisect.bisect_left(timestamps, timestamp)
    upper = poses[upper_index]
    if upper.timestamp == timestamp or upper_index == 0:
        return Pose(timestamp, upper.translation, upper.quaternion)
    lower = poses[upper_index - 1]
    dt = upper.timestamp - lower.timestamp
    if max_gap is not None and dt > max_gap:
        return None
    ratio = (timestamp - lower.timestamp) / dt
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
    """Apply SE(2) transform in world XY while preserving 3D roll/pitch."""
    theta, translation_x, translation_y = transform
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x, y, z = pose.translation
    half_theta = theta / 2.0
    q_rot = (0.0, 0.0, math.sin(half_theta), math.cos(half_theta))
    return Pose(
        pose.timestamp,
        (
            cos_theta * x - sin_theta * y + translation_x,
            sin_theta * x + cos_theta * y + translation_y,
            z,
        ),
        multiply_quaternions(q_rot, pose.quaternion),
    )


def rms(values: Iterable[float]) -> float:
    values = list(values)
    return math.sqrt(sum(value * value for value in values) / len(values))


def calculate_ape(
    ground_truth: list[Pose], estimates: list[Pose]
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (trans_2d_errors, rot_yaw_errors, trans_3d_errors, rot_so3_errors)."""
    trans_2d_errors: list[float] = []
    rot_yaw_errors: list[float] = []
    trans_3d_errors: list[float] = []
    rot_so3_errors: list[float] = []
    for reference, estimate in zip(ground_truth, estimates):
        dx = reference.translation[0] - estimate.translation[0]
        dy = reference.translation[1] - estimate.translation[1]
        dz = reference.translation[2] - estimate.translation[2]
        trans_2d_errors.append(math.hypot(dx, dy))
        rot_yaw_errors.append(
            abs(wrap_angle(yaw(reference.quaternion) - yaw(estimate.quaternion)))
        )
        trans_3d_errors.append(math.sqrt(dx * dx + dy * dy + dz * dz))
        rot_so3_errors.append(
            so3_angle_error(reference.quaternion, estimate.quaternion)
        )
    return trans_2d_errors, rot_yaw_errors, trans_3d_errors, rot_so3_errors


def local_se2_delta(
    start: Pose, end: Pose
) -> tuple[float, float, float]:
    """Compute (dx_local, dy_local, dyaw) of `end` in the SE(2) frame of `start`."""
    dx_world = end.translation[0] - start.translation[0]
    dy_world = end.translation[1] - start.translation[1]
    start_yaw = yaw(start.quaternion)
    cos_y = math.cos(start_yaw)
    sin_y = math.sin(start_yaw)
    dx_local = cos_y * dx_world + sin_y * dy_world
    dy_local = -sin_y * dx_world + cos_y * dy_world
    dyaw = wrap_angle(yaw(end.quaternion) - start_yaw)
    return dx_local, dy_local, dyaw


def estimate_median_dt(poses: list[Pose]) -> float:
    if len(poses) < 2:
        return 0.0
    dts = sorted(
        b.timestamp - a.timestamp for a, b in zip(poses, poses[1:])
    )
    return dts[len(dts) // 2]


def calculate_rpe(
    raw_ground_truth: list[Pose],
    raw_estimates: list[Pose],
    anchor_timestamps: list[float],
    time_step: float,
    distance_step: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute local-frame SE(2) RPE over exact time and arc-length windows."""
    gt_timestamps = [p.timestamp for p in raw_ground_truth]
    est_timestamps = [p.timestamp for p in raw_estimates]

    # Reject interpolation across dropouts larger than 2.5x median sampling period
    max_gt_gap = max(0.1, 2.5 * estimate_median_dt(raw_ground_truth))
    max_est_gap = max(0.5, 2.5 * estimate_median_dt(raw_estimates))

    gt_cum_dist = [0.0]
    for prev, curr in zip(raw_ground_truth, raw_ground_truth[1:]):
        gt_cum_dist.append(
            gt_cum_dist[-1]
            + math.hypot(
                curr.translation[0] - prev.translation[0],
                curr.translation[1] - prev.translation[1],
            )
        )

    def interpolate_distance_at_timestamp(t: float) -> float | None:
        if t < gt_timestamps[0] or t > gt_timestamps[-1]:
            return None
        idx = bisect.bisect_left(gt_timestamps, t)
        if gt_timestamps[idx] == t or idx == 0:
            return gt_cum_dist[idx]
        dt = gt_timestamps[idx] - gt_timestamps[idx - 1]
        if dt > max_gt_gap:
            return None
        ratio = (t - gt_timestamps[idx - 1]) / dt
        return gt_cum_dist[idx - 1] + ratio * (
            gt_cum_dist[idx] - gt_cum_dist[idx - 1]
        )

    def timestamp_at_distance(target_dist: float, lo_idx: int) -> float | None:
        if target_dist > gt_cum_dist[-1]:
            return None
        idx = bisect.bisect_left(gt_cum_dist, target_dist, lo=lo_idx)
        if idx >= len(gt_cum_dist):
            return None
        if gt_cum_dist[idx] == target_dist or idx == 0:
            return gt_timestamps[idx]
        dd = gt_cum_dist[idx] - gt_cum_dist[idx - 1]
        dt = gt_timestamps[idx] - gt_timestamps[idx - 1]
        if dt > max_gt_gap or dd <= 1e-9:
            return None
        ratio = (target_dist - gt_cum_dist[idx - 1]) / dd
        return gt_timestamps[idx - 1] + ratio * dt

    def compute_pair_error(
        t_start: float, t_end: float
    ) -> tuple[float, float] | None:
        gt_start = interpolate(
            raw_ground_truth, t_start, gt_timestamps, max_gt_gap
        )
        gt_end = interpolate(
            raw_ground_truth, t_end, gt_timestamps, max_gt_gap
        )
        est_start = interpolate(
            raw_estimates, t_start, est_timestamps, max_est_gap
        )
        est_end = interpolate(
            raw_estimates, t_end, est_timestamps, max_est_gap
        )
        if (
            gt_start is None
            or gt_end is None
            or est_start is None
            or est_end is None
        ):
            return None
        gt_dx, gt_dy, gt_dyaw = local_se2_delta(gt_start, gt_end)
        est_dx, est_dy, est_dyaw = local_se2_delta(est_start, est_end)
        trans_err = math.hypot(gt_dx - est_dx, gt_dy - est_dy)
        rot_err = abs(wrap_angle(gt_dyaw - est_dyaw))
        return trans_err, rot_err

    time_translation: list[float] = []
    time_rotation: list[float] = []
    distance_translation: list[float] = []
    distance_rotation: list[float] = []

    for t_start in anchor_timestamps:
        # 1. Fixed time-step RPE
        t_time_end = t_start + time_step
        err = compute_pair_error(t_start, t_time_end)
        if err is not None:
            time_translation.append(err[0])
            time_rotation.append(err[1])

        # 2. Fixed distance-step RPE
        d_start = interpolate_distance_at_timestamp(t_start)
        if d_start is not None:
            lo_idx = bisect.bisect_left(gt_timestamps, t_start)
            t_dist_end = timestamp_at_distance(d_start + distance_step, lo_idx)
            if t_dist_end is not None:
                err_dist = compute_pair_error(t_start, t_dist_end)
                if err_dist is not None:
                    distance_translation.append(err_dist[0])
                    distance_rotation.append(err_dist[1])

    return (
        time_translation,
        time_rotation,
        distance_translation,
        distance_rotation,
    )


def metric_block(values: list[float], unit_scale: float) -> dict[str, float]:
    if not values:
        return {"rmse": float("nan"), "max": float("nan"), "samples": 0}
    return {
        "rmse": rms(values) * unit_scale,
        "max": max(values) * unit_scale,
        "samples": len(values),
    }


def format_step_label(value: float, unit: str) -> str:
    if value.is_integer():
        return f"{int(value)}{unit}"
    return f"{value:g}{unit}"


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

    t_overlap_start = max(ground_truth[0].timestamp, raw_estimates[0].timestamp)
    t_overlap_end = min(ground_truth[-1].timestamp, raw_estimates[-1].timestamp)

    gt_timestamps = [p.timestamp for p in ground_truth]
    est_timestamps = [p.timestamp for p in raw_estimates]

    # Evaluate at the estimate timestamps. Sampling at the much denser ground truth would measure the
    # chord between keyframes.
    anchor_timestamps = [
        p.timestamp
        for p in raw_estimates
        if t_overlap_start <= p.timestamp <= t_overlap_end
    ]

    if len(anchor_timestamps) < 2:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "insufficient_data",
            "message": "At least two overlapping poses are required for APE/RPE metrics",
            "estimate_samples": len(raw_estimates),
            "ground_truth_overlap_samples": len(anchor_timestamps),
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

    matched_ground_truth: list[Pose] = []
    matched_estimates: list[Pose] = []
    for t in anchor_timestamps:
        gt_pose = interpolate(ground_truth, t, gt_timestamps)
        est_pose = interpolate(raw_estimates, t, est_timestamps)
        if gt_pose is not None and est_pose is not None:
            matched_ground_truth.append(gt_pose)
            matched_estimates.append(est_pose)

    transform = (0.0, 0.0, 0.0)
    if args.alignment_mode == "se2":
        transform = align_se2(matched_ground_truth, matched_estimates)
        matched_estimates = [
            transform_se2(pose, transform) for pose in matched_estimates
        ]
        aligned_raw_estimates = [
            transform_se2(pose, transform) for pose in raw_estimates
        ]
    else:
        aligned_raw_estimates = raw_estimates

    ape_trans_2d, ape_rot_yaw, ape_trans_3d, ape_rot_so3 = calculate_ape(
        matched_ground_truth, matched_estimates
    )
    rpe_translation, rpe_rotation, rpe_distance_translation, rpe_distance_rotation = (
        calculate_rpe(
            ground_truth,
            aligned_raw_estimates,
            anchor_timestamps,
            args.rpe_time_step,
            args.rpe_distance_step,
        )
    )

    time_label = format_step_label(args.rpe_time_step, "s")
    dist_label = format_step_label(args.rpe_distance_step, "m")
    rpe_time_key = f"rpe_{time_label}"
    rpe_dist_key = f"rpe_{dist_label}"

    result = {
        "alignment_mode": args.alignment_mode,
        "alignment_se2": {
            "yaw_rad": transform[0],
            "x_m": transform[1],
            "y_m": transform[2],
        },
        "timestamp_overlap": {
            "start": anchor_timestamps[0],
            "end": anchor_timestamps[-1],
        },
        "ape": {
            "translation_cm": metric_block(ape_trans_2d, 100.0),
            "rotation_deg": metric_block(ape_rot_yaw, 180.0 / math.pi),
            "translation_3d_cm": metric_block(ape_trans_3d, 100.0),
            "rotation_3d_deg": metric_block(ape_rot_so3, 180.0 / math.pi),
            "translation_below_5cm": sum(error < 0.05 for error in ape_trans_2d)
            / len(ape_trans_2d),
            "translation_below_10cm": sum(error < 0.10 for error in ape_trans_2d)
            / len(ape_trans_2d),
            "rotation_below_1deg": sum(
                error < math.radians(1.0) for error in ape_rot_yaw
            )
            / len(ape_rot_yaw),
            "rotation_below_2deg": sum(
                error < math.radians(2.0) for error in ape_rot_yaw
            )
            / len(ape_rot_yaw),
            "rotation_below_4deg": sum(
                error < math.radians(4.0) for error in ape_rot_yaw
            )
            / len(ape_rot_yaw),
        },
        rpe_time_key: {
            "translation_cm": metric_block(rpe_translation, 100.0),
            "rotation_deg": metric_block(rpe_rotation, 180.0 / math.pi),
        },
        rpe_dist_key: {
            "translation_cm": metric_block(rpe_distance_translation, 100.0),
            "rotation_deg": metric_block(rpe_distance_rotation, 180.0 / math.pi),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    summary = [
        "metric                    rmse       max       samples",
        f"APE 2D translation [cm]   {result['ape']['translation_cm']['rmse']:.3f}      {result['ape']['translation_cm']['max']:.3f}      {len(ape_trans_2d)}",
        f"APE yaw rotation [deg]    {result['ape']['rotation_deg']['rmse']:.3f}      {result['ape']['rotation_deg']['max']:.3f}      {len(ape_rot_yaw)}",
        f"APE 3D translation [cm]   {result['ape']['translation_3d_cm']['rmse']:.3f}      {result['ape']['translation_3d_cm']['max']:.3f}      {len(ape_trans_3d)}",
        f"APE 3D rotation [deg]     {result['ape']['rotation_3d_deg']['rmse']:.3f}      {result['ape']['rotation_3d_deg']['max']:.3f}      {len(ape_rot_so3)}",
        "",
        "Tracking Thresholds:",
        f"Translation < 5cm:        {result['ape']['translation_below_5cm'] * 100:.2f}%",
        f"Translation < 10cm:       {result['ape']['translation_below_10cm'] * 100:.2f}%",
        f"Rotation < 1 deg:         {result['ape']['rotation_below_1deg'] * 100:.2f}%",
        f"Rotation < 2 deg:         {result['ape']['rotation_below_2deg'] * 100:.2f}%",
        f"Rotation < 4 deg:         {result['ape']['rotation_below_4deg'] * 100:.2f}%",
        "",
        f"RPE {time_label} translation [cm]   {result[rpe_time_key]['translation_cm']['rmse']:.3f}      {result[rpe_time_key]['translation_cm']['max']:.3f}      {len(rpe_translation)}",
        f"RPE {time_label} rotation [deg]     {result[rpe_time_key]['rotation_deg']['rmse']:.3f}      {result[rpe_time_key]['rotation_deg']['max']:.3f}      {len(rpe_rotation)}",
        f"RPE {dist_label} translation [cm]   {result[rpe_dist_key]['translation_cm']['rmse']:.3f}      {result[rpe_dist_key]['translation_cm']['max']:.3f}      {len(rpe_distance_translation)}",
        f"RPE {dist_label} rotation [deg]     {result[rpe_dist_key]['rotation_deg']['rmse']:.3f}      {result[rpe_dist_key]['rotation_deg']['max']:.3f}      {len(rpe_distance_rotation)}",
    ]
    (args.output_dir / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="ascii"
    )
    print("\n".join(summary))
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
