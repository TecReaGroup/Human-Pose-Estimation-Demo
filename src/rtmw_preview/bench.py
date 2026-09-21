"""Benchmark bounded video decoding, single-person inference and encoding pipelines."""

import csv
import json
import logging
import math
import os
import platform
import threading
import time
import tomllib
from collections.abc import Generator
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np

from rtmw_preview.detector import YOLO26, PreparedDetection
from rtmw_preview.pose import BalancedPose, SinglePersonPose, load_pose
from rtmw_preview.runtime import ROOT, configure_logging

LOGGER = logging.getLogger("bench")
VIDEO_DIRECTORY = ROOT / "data" / "video"
OUTPUT_DIRECTORY = ROOT / "data" / "output"
VIDEO_SUFFIX = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}
REPORT_INTERVAL = 1.0
OUTPUT_CODEC = "mp4v"
PIPELINE_QUEUE_SIZE = 2
QUEUE_TIMEOUT = 0.1
DETECTION_CALL_STAGE = (
    "detection_input_layout", "detection_feed_setup", "detection_session_run",
    "detection_input_cleanup", "detection_output_extract", "detection_call_unattributed",
    "inference_unattributed", "prepared_input_release",
)


@dataclass
class VideoFrame:
    """Keep frame ownership and timing attached while moving between stages."""

    image: np.ndarray
    started: float
    decode_seconds: float
    prediction: SinglePersonPose | None = None
    inference_started: float = 0.0
    inference_finished: float = 0.0
    prepared: PreparedDetection | None = None
    preparation_seconds: float = 0.0
    prepared_release_seconds: float = 0.0


def inferred_frames(
    capture: cv2.VideoCapture, pose: BalancedPose,
) -> Generator[VideoFrame, None, None]:
    """Overlap decoding and inference, propagating failures and joining workers on exit."""
    decoded_queue = Queue(maxsize=PIPELINE_QUEUE_SIZE)
    inferred_queue = Queue(maxsize=PIPELINE_QUEUE_SIZE)
    stopped = threading.Event()

    def publish(queue: Queue, packet: VideoFrame | Exception | None) -> None:
        while not stopped.is_set():
            try:
                queue.put(packet, timeout=QUEUE_TIMEOUT)
                return
            except Full:
                continue

    def decode() -> None:
        try:
            while not stopped.is_set():
                started = time.perf_counter()
                available, image = capture.read()
                duration = time.perf_counter() - started
                if not available:
                    publish(decoded_queue, None)
                    return
                packet = VideoFrame(image, started, duration)
                if isinstance(pose.detector, YOLO26):
                    preparation_started = time.perf_counter()
                    packet.prepared = pose.detector.prepare(image)
                    packet.preparation_seconds = time.perf_counter() - preparation_started
                    packet.prepared.preprocess_seconds = packet.preparation_seconds
                publish(decoded_queue, packet)
        except Exception as exc:
            publish(decoded_queue, exc)

    def infer() -> None:
        try:
            while not stopped.is_set():
                try:
                    packet = decoded_queue.get(timeout=QUEUE_TIMEOUT)
                except Empty:
                    continue
                if packet is None or isinstance(packet, Exception):
                    publish(inferred_queue, packet)
                    return
                packet.inference_started = time.perf_counter()
                if packet.prepared is None:
                    packet.prediction = pose.estimate(packet.image)
                else:
                    packet.prediction = pose.estimate_prepared(packet.image, packet.prepared)
                    release_started = time.perf_counter()
                    packet.prepared = None
                    packet.prepared_release_seconds = time.perf_counter() - release_started
                packet.inference_finished = time.perf_counter()
                publish(inferred_queue, packet)
        except Exception as exc:
            publish(inferred_queue, exc)

    workers = [threading.Thread(target=decode, name="video-decode"),
               threading.Thread(target=infer, name="pose-inference")]
    active_workers = []
    try:
        for worker in workers:
            worker.start()
            active_workers.append(worker)
        while True:
            try:
                packet = inferred_queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                continue
            if isinstance(packet, Exception):
                raise packet
            if packet is None:
                return
            yield packet
    finally:
        stopped.set()
        for worker in active_workers:
            worker.join()


def benchmark_video(video: Path, pose: BalancedPose, inference: dict, load_seconds: float) -> None:
    """Save every rendered frame and report wall-clock throughput without frame skipping."""
    destination = OUTPUT_DIRECTORY / video.stem
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    summary_path.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(video))
    writer = None
    frames = inferred_frames(capture, pose)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{video}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"视频帧率无效：{video}: {source_fps}")
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = height = 0
        frame_count = 0
        window_frames = 0
        render_seconds = 0.0
        window_render_seconds = 0.0
        decode_seconds = 0.0
        encode_seconds = 0.0
        detection_seconds = 0.0
        pose_seconds = 0.0
        draw_seconds = 0.0
        detected_people = 0
        pose_frames = 0
        frame_latency = []
        stage_samples = {name: [] for name in (
            "decode", "input_stage", "detection", "pose", "draw", "encode", "inference_stage",
            "output_stage", "before_inference_wait", "before_output_wait",
            "detection_preprocess", "detection_inference_call", "detection_postprocess",
            "detection_unattributed",
            *DETECTION_CALL_STAGE,
        )}
        LOGGER.info("Starting %s: source_fps=%.3f reported_frames=%d output=%s",
                    video.name, source_fps, reported_frames, destination)
        with (destination / "frame.csv").open("w", newline="", encoding="utf-8") as stream:
            telemetry = csv.writer(stream)
            telemetry.writerow([
                "frame", "elapsed_s", "decode_ms", "detection_ms", "pose_ms", "draw_ms",
                "pose_render_ms", "encode_ms", "frame_latency_ms", "detected_people", "pose_people",
                *[f"{name}_ms" for name in (
                    "inference_stage", "before_inference_wait", "before_output_wait",
                    "detection_preprocess", "detection_inference_call", "detection_postprocess",
                    "detection_unattributed", "input_stage",
                )],
                *[f"{name}_ms" for name in DETECTION_CALL_STAGE],
            ])
            started = last_report = time.perf_counter()
            for packet in frames:
                output_started = time.perf_counter()
                frame = packet.image
                prediction = packet.prediction
                assert prediction is not None
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(destination / "pose.mp4"), cv2.VideoWriter_fourcc(*OUTPUT_CODEC),
                        source_fps, (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建输出视频：{destination / 'pose.mp4'}")
                render_started = time.perf_counter()
                rendered = pose.draw(frame, prediction)
                rendered_at = time.perf_counter()
                writer.write(rendered)
                encoded = time.perf_counter()
                decode_duration = packet.decode_seconds
                draw_duration = rendered_at - render_started
                render_duration = prediction.detection_seconds + prediction.pose_seconds + draw_duration
                encode_duration = encoded - rendered_at
                frame_duration = encoded - packet.started
                frame_count += 1
                window_frames += 1
                decode_seconds += decode_duration
                render_seconds += render_duration
                window_render_seconds += render_duration
                encode_seconds += encode_duration
                detection_seconds += prediction.detection_seconds
                pose_seconds += prediction.pose_seconds
                draw_seconds += draw_duration
                detected_people += prediction.detected_people
                pose_frames += int(prediction.detected_people > 0)
                frame_latency.append(frame_duration * 1000)
                durations = {
                    "decode": decode_duration,
                    "input_stage": decode_duration + packet.preparation_seconds,
                    "detection": prediction.detection_seconds,
                    "pose": prediction.pose_seconds,
                    "draw": draw_duration,
                    "encode": encode_duration,
                    "inference_stage": packet.inference_finished - packet.inference_started,
                    "output_stage": encoded - output_started,
                    "before_inference_wait": (
                        packet.inference_started - packet.started - packet.decode_seconds
                        - packet.preparation_seconds
                    ),
                    "before_output_wait": output_started - packet.inference_finished,
                    "prepared_input_release": packet.prepared_release_seconds,
                }
                durations["inference_unattributed"] = (
                    durations["inference_stage"] - prediction.detection_seconds
                    + packet.preparation_seconds - prediction.pose_seconds
                    - packet.prepared_release_seconds
                )
                if prediction.detection_timing is not None:
                    durations.update({
                        f"detection_{name}": seconds
                        for name, seconds in prediction.detection_timing.items()
                    })
                    durations["detection_unattributed"] = (
                        prediction.detection_seconds - sum(
                            prediction.detection_timing[name]
                            for name in ("preprocess", "inference_call", "postprocess")
                        )
                    )
                    durations["detection_call_unattributed"] = (
                        prediction.detection_timing["inference_call"] - sum(
                            prediction.detection_timing[name] for name in (
                                "input_layout", "feed_setup", "session_run",
                                "input_cleanup", "output_extract",
                            )
                        )
                    )
                for name, seconds in durations.items():
                    stage_samples[name].append(seconds * 1000)
                telemetry.writerow([
                    frame_count, encoded - started, decode_duration * 1000,
                    prediction.detection_seconds * 1000, prediction.pose_seconds * 1000,
                    draw_duration * 1000, render_duration * 1000, encode_duration * 1000,
                    frame_duration * 1000, prediction.detected_people,
                    int(prediction.detected_people > 0),
                    *[durations[name] * 1000 for name in (
                        "inference_stage", "before_inference_wait", "before_output_wait",
                    )],
                    *[durations[name] * 1000 if name in durations else "" for name in (
                        "detection_preprocess", "detection_inference_call", "detection_postprocess",
                        "detection_unattributed", "input_stage",
                    )],
                    *[durations[name] * 1000 if name in durations else ""
                      for name in DETECTION_CALL_STAGE],
                ])
                now = time.perf_counter()
                if now - last_report >= REPORT_INTERVAL:
                    average_fps = frame_count / (now - started)
                    LOGGER.info(
                        "%s: frames=%d/%d interval_fps=%.2f average_fps=%.2f "
                        "pose_render_ms=%.2f average_pose_render_ms=%.2f speed=%.2fx elapsed=%.1fs "
                        "avg_detect_ms=%.2f avg_pose_ms=%.2f avg_draw_ms=%.2f avg_encode_ms=%.2f",
                        video.name, frame_count, reported_frames,
                        window_frames / (now - last_report), average_fps,
                        window_render_seconds * 1000 / window_frames,
                        render_seconds * 1000 / frame_count, average_fps / source_fps,
                        now - started,
                        detection_seconds * 1000 / frame_count, pose_seconds * 1000 / frame_count,
                        draw_seconds * 1000 / frame_count, encode_seconds * 1000 / frame_count,
                    )
                    stream.flush()
                    window_frames = 0
                    window_render_seconds = 0.0
                    last_report = now
            if frame_count == 0:
                raise RuntimeError(f"视频没有可解码帧：{video}")
            assert writer is not None
            finalize_started = time.perf_counter()
            writer.release()
            writer = None
            finalize_seconds = time.perf_counter() - finalize_started
        elapsed = time.perf_counter() - started
        if reported_frames > 0 and frame_count < reported_frames:
            raise RuntimeError(
                f"视频可能提前解码结束：{video}，读取 {frame_count}/{reported_frames} 帧"
            )
        average_fps = frame_count / elapsed
        stage_timing = {
            name: {
                "sample_count": len(samples),
                "average_ms": float(np.mean(samples)),
                "p50_ms": float(np.percentile(samples, 50)),
                "p95_ms": float(np.percentile(samples, 95)),
                "p99_ms": float(np.percentile(samples, 99)),
            } if samples else None
            for name, samples in stage_samples.items()
        }
        pipeline_stage = {}
        for name in ("input_stage", "inference_stage", "output_stage"):
            mean_seconds = float(np.mean(stage_samples[name])) / 1000
            pipeline_stage[name] = {
                "average_service_ms": mean_seconds * 1000,
                "estimated_capacity_fps": 1 / mean_seconds,
                "service_wall_time_ratio": sum(stage_samples[name]) / 1000 / elapsed,
            }
        bottleneck = max(pipeline_stage, key=lambda name: pipeline_stage[name]["average_service_ms"])
        summary = {
            "schema_version": 5,
            "detector_input_preparation": "direct_rgb_nchw_float32" if isinstance(pose.detector, YOLO26)
                                          else "rtmlib_default",
            "runtime_environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "onnxruntime_gpu": version("onnxruntime-gpu"),
                "rtmlib": version("rtmlib"),
            },
            "detector_preprocess_stage": "input_stage" if isinstance(pose.detector, YOLO26)
                                         else "inference_stage",
            "source": str(video),
            "inference": inference,
            "detector_providers": pose.detector.session.get_providers(),
            "pose_providers": pose.pose.session.get_providers(),
            "model_load_seconds": load_seconds,
            "source_fps": source_fps,
            "reported_frames": reported_frames,
            "processed_frames": frame_count,
            "width": width,
            "height": height,
            "elapsed_seconds": elapsed,
            "average_fps": average_fps,
            "realtime_speed": average_fps / source_fps,
            "average_decode_ms": decode_seconds * 1000 / frame_count,
            "average_pose_render_ms": render_seconds * 1000 / frame_count,
            "pose_render_fps": frame_count / render_seconds,
            "average_encode_ms": encode_seconds * 1000 / frame_count,
            "average_detection_ms": detection_seconds * 1000 / frame_count,
            "average_pose_ms": pose_seconds * 1000 / frame_count,
            "average_draw_ms": draw_seconds * 1000 / frame_count,
            "average_detected_people": detected_people / frame_count,
            "frames_with_pose": pose_frames,
            "frames_without_pose": frame_count - pose_frames,
            "average_pose_when_present_ms": pose_seconds * 1000 / pose_frames if pose_frames else None,
            "stage_timing": stage_timing,
            "pipeline_stage": pipeline_stage,
            "estimated_bottleneck_stage": bottleneck,
            "detector_input_size": list(pose.detector.model_input_size),
            "pose_input_size": list(pose.pose.model_input_size),
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "timing_notes": {
                "detection_breakdown": "Available for YOLO26 only; unsupported stages are null.",
                "detection_inference_call": "YOLO26 prepared-tensor call wall time, including feed "
                                            "setup, runtime execution, transfers and synchronization; "
                                            "not pure GPU time or an isolated session.run measurement.",
                "detection_input_layout": "Zero for prepared YOLO26 inputs: RGB channel ordering, "
                                          "float32 conversion and normalization are fused into final "
                                          "contiguous NCHW tensor creation in detection_preprocess.",
                "detection_input_cleanup": "Release of the feed dictionary only; the prepared tensor "
                                           "is released separately in prepared_input_release.",
                "detection_session_run": "Synchronous session.run wall time including runtime, "
                                         "transfers and synchronization; not pure GPU execution time.",
                "detection_call_breakdown": "input_layout + feed_setup + session_run + input_cleanup "
                                            "+ output_extract + call_unattributed partition inference_call. "
                                            "Do not add parent and child timings together.",
                "inference_unattributed": "Inference thread duration minus detection work on this "
                                          "thread, pose call and prepared input release.",
                "gpu_utilization": "Not sampled. GPU clocks, power, temperature and node placement "
                                   "cannot be inferred from these CPU wall-clock timings.",
                "queue_wait": "Time between stages includes queue backpressure and scheduling.",
                "input_stage": "Decode plus detector preparation for YOLO26; decode only for YOLOX.",
                "detection": "Total detection service time across input and inference threads; "
                             "not the inference thread duration. Preprocessing includes array cleanup.",
                "detection_unattributed": "Outer detection timing minus measured substages, "
                                          "including call overhead, cleanup and scheduling.",
                "stage_capacity": "Reciprocal of mean service time; an estimate, not measured FPS. "
                                  "Output service includes first-frame writer setup, excludes telemetry.",
                "service_wall_time_ratio": "Thread service time / video wall time; not GPU utilization.",
                "provider": "Registered providers do not identify per-node placement or fallback.",
            },
            "person_selection": "largest_bbox_area",
            "pipeline_queue_size": PIPELINE_QUEUE_SIZE,
            "encoder_finalize_seconds": finalize_seconds,
            "frame_latency_p50_ms": float(np.percentile(frame_latency, 50)),
            "frame_latency_p95_ms": float(np.percentile(frame_latency, 95)),
            "frame_latency_p99_ms": float(np.percentile(frame_latency, 99)),
            "timing_scope": "All frames, including first frame; model loading excluded. "
                            "Wall time includes decoding, pose rendering, encoding and telemetry. "
                            "Stages overlap; stage durations must not be added to estimate FPS. "
                            "Detection and pose include preprocessing and postprocessing. "
                            "Frame latency includes queue waits from decode start to encode return, "
                            "excluding telemetry and final encoder flush. pose_render_fps is the "
                            "reciprocal of summed mean detection, pose and draw service time, "
                            "not measured pipeline throughput.",
            "output_codec": OUTPUT_CODEC,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info(
            "%s complete: frames=%d average_fps=%.2f speed=%.2fx "
            "pose_render_ms=%.2f p95_frame_ms=%.2f elapsed=%.2fs summary=%s",
            video.name, frame_count, average_fps, average_fps / source_fps,
            summary["average_pose_render_ms"], summary["frame_latency_p95_ms"], elapsed, summary_path,
        )
    finally:
        frames.close()
        capture.release()
        if writer is not None:
            writer.release()


def main() -> int:
    """Load the configured engine once and benchmark all videos in data/video."""
    configure_logging()
    try:
        videos = sorted(
            path for path in VIDEO_DIRECTORY.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIX
        )
        if not videos:
            raise ValueError(f"请先将视频放入 {VIDEO_DIRECTORY}")
        stems = [video.stem.casefold() for video in videos]
        if len(stems) != len(set(stems)):
            raise ValueError("视频存在相同文件名（不含扩展名），请重命名以避免输出目录冲突。")
        with (ROOT / "config" / "config.toml").open("rb") as stream:
            inference = tomllib.load(stream)["inference"]
        started = time.perf_counter()
        pose = load_pose(inference)
        load_seconds = time.perf_counter() - started
        LOGGER.info("Model initialization: %.2fs; benchmarking %d video(s)", load_seconds, len(videos))
        failures = 0
        for video in videos:
            try:
                benchmark_video(video, pose, inference, load_seconds)
            except Exception:
                failures += 1
                LOGGER.exception("Video benchmark failed: %s", video)
        LOGGER.info("Benchmark finished: succeeded=%d failed=%d", len(videos) - failures, failures)
        return 1 if failures else 0
    except KeyboardInterrupt:
        LOGGER.warning("Benchmark interrupted; current video output may be incomplete")
        return 130
    except Exception:
        LOGGER.exception("Benchmark startup failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
