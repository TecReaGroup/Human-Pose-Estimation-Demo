"""Compare native rtmlib balanced and configured single-person predictions."""

import csv
import json
import logging
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody

from rtmw_preview.bench import OUTPUT_DIRECTORY, VIDEO_DIRECTORY, VIDEO_SUFFIX
from rtmw_preview.model import MODEL_URL, download_model
from rtmw_preview.pose import BalancedPose, SinglePersonPose, load_pose
from rtmw_preview.runtime import ROOT, configure_logging

LOGGER = logging.getLogger("verify")
MATCH_IOU = 0.5
PIXEL_TOLERANCE = 5.0
NORMALIZED_TOLERANCE = 0.05
REPORT_INTERVAL = 1.0


def load_baseline(inference: dict) -> BalancedPose:
    """Use installed rtmlib balanced models and native CUDA execution with local assets."""
    preset = Wholebody.MODE["balanced"]
    if preset["det"] != MODEL_URL["yolox_m"] or preset["pose"] != MODEL_URL["rtmw_x"]:
        raise RuntimeError("Installed rtmlib balanced model preset differs from local model sources.")
    baseline = Wholebody(
        det=str(download_model("yolox_m")), det_input_size=preset["det_input_size"],
        pose=str(download_model("rtmw_x")), pose_input_size=preset["pose_input_size"],
        mode="balanced", to_openpose=False, backend="onnxruntime",
        device=f"cuda:{inference['device_id']}",
    )
    for estimator in (baseline.det_model, baseline.pose_model):
        estimator.session.disable_fallback()
        if "CUDAExecutionProvider" not in estimator.session.get_providers():
            raise RuntimeError("Baseline CUDA provider failed to initialize.")
    return BalancedPose(baseline.det_model, baseline.pose_model, inference["keypoint_threshold"])


def predict_single(estimator: BalancedPose, frame: np.ndarray) -> tuple[np.ndarray | None, SinglePersonPose]:
    """Select the largest detected box and run exactly one pose crop."""
    boxes = estimator.detector(frame)
    prediction = estimator.estimate_largest(frame, boxes, 0.0, None)
    if len(boxes) == 0:
        return None, prediction
    largest = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    return boxes[largest], prediction


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Compute IoU of positive-area xyxy person boxes."""
    overlap = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2])) - intersection
    return intersection / union


def distribution(samples: list[float]) -> dict:
    """Summarize pooled samples, retaining null metrics when no comparison is available."""
    if not samples:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"count": len(samples), "mean": float(np.mean(samples)),
            "p50": float(np.percentile(samples, 50)), "p95": float(np.percentile(samples, 95)),
            "max": float(np.max(samples))}


def serialize_prediction(box: np.ndarray | None, prediction: SinglePersonPose) -> dict:
    """Keep raw predictions for independently rechecking comparison metrics."""
    return {
        "detected_people": prediction.detected_people,
        "box": box.tolist() if box is not None else None,
        "keypoints": prediction.keypoints.tolist() if prediction.keypoints is not None else None,
        "scores": prediction.scores.tolist() if prediction.scores is not None else None,
    }


def verify_video(video: Path, baseline: BalancedPose, current: BalancedPose, inference: dict) -> None:
    """Compare every decoded frame and save raw predictions plus difference metrics."""
    destination = OUTPUT_DIRECTORY / video.stem / "verify"
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video))
    counts = dict.fromkeys(("frames", "both_present", "both_absent", "baseline_only",
                           "current_only", "matched_box", "low_iou_box", "confidence_disagreement"), 0)
    samples = {name: [] for name in (
        "box_iou", "all_point_distance_px", "all_score_absolute_difference",
        "matched_confident_distance_px", "matched_confident_normalized_distance",
    )}
    started = last_report = time.perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with (destination / "prediction.jsonl").open("w", encoding="utf-8") as raw_stream, (
            destination / "frame.csv"
        ).open("w", newline="", encoding="utf-8") as csv_stream:
            rows = csv.writer(csv_stream)
            rows.writerow(["frame", "baseline_people", "current_people", "box_iou", "matched_box",
                           "mean_distance_px", "mean_score_abs_diff", "joint_confident_points",
                           "matched_confident_mean_distance_px"])
            while True:
                available, frame = capture.read()
                if not available:
                    break
                baseline_box, reference = predict_single(baseline, frame.copy())
                current_box, candidate = predict_single(current, frame.copy())
                counts["frames"] += 1
                raw_stream.write(json.dumps({
                    "frame": counts["frames"],
                    "baseline": serialize_prediction(baseline_box, reference),
                    "current": serialize_prediction(current_box, candidate),
                }, allow_nan=False) + "\n")
                iou = mean_distance = mean_score = confident_distance = None
                matched = False
                confident_count = 0
                if baseline_box is not None and current_box is not None:
                    assert reference.keypoints is not None and candidate.keypoints is not None
                    assert reference.scores is not None and candidate.scores is not None
                    if reference.keypoints.shape != candidate.keypoints.shape:
                        raise ValueError("Baseline and current keypoint layouts differ.")
                    counts["both_present"] += 1
                    iou = box_iou(baseline_box, current_box)
                    matched = iou >= MATCH_IOU
                    counts["matched_box" if matched else "low_iou_box"] += 1
                    distances = np.linalg.norm(reference.keypoints[0] - candidate.keypoints[0], axis=1)
                    score_difference = np.abs(reference.scores[0] - candidate.scores[0])
                    reference_visible = reference.scores[0] >= baseline.threshold
                    candidate_visible = candidate.scores[0] >= current.threshold
                    confident = reference_visible & candidate_visible
                    confident_count = int(np.count_nonzero(confident))
                    counts["confidence_disagreement"] += int(np.count_nonzero(
                        reference_visible != candidate_visible
                    ))
                    samples["box_iou"].append(iou)
                    samples["all_point_distance_px"].extend(distances.tolist())
                    samples["all_score_absolute_difference"].extend(score_difference.tolist())
                    mean_distance, mean_score = float(np.mean(distances)), float(np.mean(score_difference))
                    if matched and confident_count:
                        selected_distances = distances[confident]
                        diagonal = float(np.linalg.norm(baseline_box[2:] - baseline_box[:2]))
                        samples["matched_confident_distance_px"].extend(selected_distances.tolist())
                        samples["matched_confident_normalized_distance"].extend(
                            (selected_distances / diagonal).tolist()
                        )
                        confident_distance = float(np.mean(selected_distances))
                elif baseline_box is not None:
                    counts["baseline_only"] += 1
                elif current_box is not None:
                    counts["current_only"] += 1
                else:
                    counts["both_absent"] += 1
                rows.writerow([counts["frames"], reference.detected_people, candidate.detected_people,
                               iou, matched, mean_distance, mean_score, confident_count, confident_distance])
                now = time.perf_counter()
                if now - last_report >= REPORT_INTERVAL:
                    LOGGER.info("%s: compared=%d/%d matched=%d baseline_only=%d current_only=%d",
                                video.name, counts["frames"], expected_frames, counts["matched_box"],
                                counts["baseline_only"], counts["current_only"])
                    csv_stream.flush()
                    raw_stream.flush()
                    last_report = now
        if counts["frames"] == 0 or (expected_frames > 0 and counts["frames"] < expected_frames):
            raise RuntimeError(f"Incomplete video decode: {counts['frames']}/{expected_frames}")
        confident_distances = samples["matched_confident_distance_px"]
        normalized_distances = samples["matched_confident_normalized_distance"]
        summary = {
            "schema_version": 1, "source": str(video), "rtmlib_version": version("rtmlib"),
            "baseline": {"preset": "balanced", "detector": "yolox_m", "pose": "rtmw_x",
                         "backend": "onnxruntime", "device": f"cuda:{inference['device_id']}",
                         "detector_input_size": list(baseline.detector.model_input_size),
                         "pose_input_size": list(baseline.pose.model_input_size),
                         "to_openpose": False},
            "current": inference,
            "baseline_providers": {"detector": baseline.detector.session.get_providers(),
                                   "pose": baseline.pose.session.get_providers()},
            "current_providers": {"detector": current.detector.session.get_providers(),
                                  "pose": current.pose.session.get_providers()},
            "selection": "largest_bbox_area_independently_per_frame",
            "match_iou_threshold": MATCH_IOU, "confidence_threshold": current.threshold,
            "counts": counts, "difference": {name: distribution(values) for name, values in samples.items()},
            "pixel_tolerance": PIXEL_TOLERANCE,
            "matched_confident_within_pixel_tolerance_ratio": float(np.mean(
                np.asarray(confident_distances) <= PIXEL_TOLERANCE
            )) if confident_distances else None,
            "normalized_tolerance": NORMALIZED_TOLERANCE,
            "matched_confident_within_normalized_tolerance_ratio": float(np.mean(
                np.asarray(normalized_distances) <= NORMALIZED_TOLERANCE
            )) if normalized_distances else None,
            "elapsed_seconds": time.perf_counter() - started,
            "interpretation": "Agreement comparison, not accuracy or recall against ground truth. "
                              "Models/backends may differ. Low IoU can mean different selected people "
                              "or different boxes; IoU matching is not identity tracking. All-point "
                              "metrics include low-confidence points and low-IoU frames. Matched-confident "
                              "metrics require IoU >= threshold and both scores >= confidence threshold. "
                              "Normalized distance uses baseline box diagonal. Null means no samples. "
                              "Timing includes both models and reporting, not comparable to make bench.",
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        LOGGER.info("%s verification complete: %s", video.name, summary_path)
    finally:
        capture.release()


def main() -> int:
    """Compare all input videos with installed rtmlib balanced on CUDA."""
    configure_logging()
    try:
        videos = sorted(path for path in VIDEO_DIRECTORY.iterdir()
                        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIX)
        if not videos:
            raise ValueError(f"请先将视频放入 {VIDEO_DIRECTORY}")
        if len({video.stem.casefold() for video in videos}) != len(videos):
            raise ValueError("视频文件名（不含扩展名）不能重复。")
        with (ROOT / "config" / "config.toml").open("rb") as stream:
            inference = tomllib.load(stream)["inference"]
        current = load_pose(inference)
        baseline = load_baseline(inference)
        failures = 0
        for video in videos:
            try:
                verify_video(video, baseline, current, inference)
            except Exception:
                failures += 1
                LOGGER.exception("Verification failed: %s", video)
        return int(failures > 0)
    except KeyboardInterrupt:
        LOGGER.warning("Verification interrupted; partial outputs retained without a summary")
        return 130
    except Exception:
        LOGGER.exception("Verification startup failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
