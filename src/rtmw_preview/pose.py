"""Whole-body pose inference with TensorRT or rtmlib ONNX Runtime CUDA."""

import logging
import os
import sys
import time
import tomllib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose, YOLOX, draw_skeleton

from rtmw_preview.detector import YOLO26, PreparedDetection
from rtmw_preview.model import configured_detector, download_model, export_yolo26m
from rtmw_preview.runtime import PERF, ROOT, configure_logging

LOGGER = logging.getLogger("inference")
TRT_PROVIDER = "TensorrtExecutionProvider"
# Keep the detector's small integer-index postprocessing partitions outside TRT.
TRT_MODEL_PARTITION = {
    "yolox_m": ("TopK,NonMaxSuppression", 50),
    "yolo26m": ("", 1),
    "rtmw_x": ("", 1),
}
DLL_DIRECTORY = []


@dataclass
class SinglePersonPose:
    """Carry single-person predictions and synchronous call timings to rendering."""

    keypoints: np.ndarray | None
    scores: np.ndarray | None
    detected_people: int
    detection_seconds: float
    pose_seconds: float
    detection_timing: dict[str, float] | None = None


def load_gpu_runtime() -> None:
    """Load packaged CUDA, cuDNN and TensorRT shared libraries."""
    if sys.platform == "win32":
        import site

        for site_path in site.getsitepackages():
            library_root = Path(site_path) / "tensorrt_libs"
            if library_root.is_dir():
                DLL_DIRECTORY.append(os.add_dll_directory(str(library_root)))
                os.environ["PATH"] = str(library_root) + os.pathsep + os.environ["PATH"]
    ort.preload_dlls(directory="")
    import tensorrt

    LOGGER.info("ONNX Runtime %s, TensorRT %s", ort.__version__, tensorrt.__version__)
    if TRT_PROVIDER not in ort.get_available_providers():
        raise RuntimeError("当前 ONNX Runtime 不提供 TensorRT EP，请执行 make install。")


def load_detector(name: str, device: str) -> YOLOX | YOLO26:
    """Create the selected person detector with its model-specific preprocessing."""
    if name == "yolo26m":
        return YOLO26(str(export_yolo26m()), device=device)
    return YOLOX(
        str(download_model(name)), model_input_size=(640, 640),
        backend="onnxruntime", device=device,
    )


class BalancedPose:
    """Own the selected detector and whole-body pose sessions."""

    def __init__(self, detector: YOLOX | YOLO26, pose: RTMPose, keypoint_threshold: float) -> None:
        self.threshold = keypoint_threshold
        self.detector = detector
        self.pose = pose

    @classmethod
    def from_cuda(
        cls, device_id: int, keypoint_threshold: float, detector_model: str,
    ) -> "BalancedPose":
        """Create rtmlib's native ONNX Runtime CUDA sessions."""
        ort.preload_dlls(directory="")
        LOGGER.info("Initializing ONNX Runtime %s CUDA on device %d", ort.__version__, device_id)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("当前 ONNX Runtime 不提供 CUDA EP，请执行 make install。")
        self = cls(
            load_detector(detector_model, f"cuda:{device_id}"),
            RTMPose(
                str(download_model("rtmw_x")), model_input_size=(192, 256),
                backend="onnxruntime", device=f"cuda:{device_id}",
            ),
            keypoint_threshold,
        )
        for name, estimator in ((detector_model, self.detector), ("rtmw_x", self.pose)):
            estimator.session.disable_fallback()
            if "CUDAExecutionProvider" not in estimator.session.get_providers():
                raise RuntimeError(f"{name}: CUDA EP 加载失败，请检查 CUDA/cuDNN DLL。")
            width, height = estimator.model_input_size
            estimator.inference(np.zeros((height, width, 3), dtype=np.float32))
            LOGGER.info("%s ready, providers=%s", name, estimator.session.get_providers())
        return self

    @classmethod
    def from_tensorrt(
        cls, device_id: int, workspace_mb: int, keypoint_threshold: float,
        detector_model: str,
    ) -> "BalancedPose":
        """Build or load and warm up TensorRT FP16 sessions."""
        load_gpu_runtime()
        self = cls(
            load_detector(detector_model, "cpu"),
            RTMPose(
                str(download_model("rtmw_x")), model_input_size=(192, 256),
                backend="onnxruntime", device="cpu",
            ),
            keypoint_threshold,
        )
        for name, estimator in ((detector_model, self.detector), ("rtmw_x", self.pose)):
            started = time.perf_counter()
            estimator.session.disable_fallback()
            runtime_version = f"ort_{ort.__version__}_trt_{version('tensorrt-cu12')}"
            excluded_op, min_subgraph_size = TRT_MODEL_PARTITION[name]
            partition = (
                "exclude_" + excluded_op.replace(",", "_").lower()
                + f"_min_{min_subgraph_size}"
            )
            cache = ROOT / "temp" / "engine" / runtime_version / partition / name
            cache.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Building/loading %s TensorRT FP16 engine; first build may take minutes", name)
            LOGGER.info(
                "%s: TensorRT excluded ops=%s, minimum subgraph size=%d; "
                "remaining nodes use ORT CUDA/CPU, cache=%s",
                name, excluded_op or "none", min_subgraph_size, cache,
            )
            estimator.session.set_providers([
                (TRT_PROVIDER, {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    "trt_op_types_to_exclude": excluded_op,
                    "trt_min_subgraph_size": min_subgraph_size,
                    "trt_max_workspace_size": workspace_mb * 1024 * 1024,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache),
                }),
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ])
            if TRT_PROVIDER not in estimator.session.get_providers():
                raise RuntimeError(f"{name}: TensorRT EP 加载失败，请检查 CUDA/TensorRT DLL。")
            width, height = estimator.model_input_size
            estimator.inference(np.zeros((height, width, 3), dtype=np.float32))
            engines = list(cache.glob("*.engine"))
            if not engines:
                raise RuntimeError(f"{name}: 预热完成但没有生成 TensorRT 引擎缓存：{cache}")
            LOGGER.log(
                PERF, "%s: %d engine file(s) persisted to %s, initialization took %.1fs",
                name, len(engines), cache, time.perf_counter() - started,
            )
            LOGGER.info("%s ready, providers=%s", name, estimator.session.get_providers())

        return self

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the largest person's 133-keypoint whole-body skeleton."""
        return self.draw(frame, self.estimate(frame))

    def estimate(self, frame: np.ndarray) -> SinglePersonPose:
        """Estimate only the largest xyxy person box by area on each frame."""
        started = time.perf_counter()
        boxes = self.detector(frame)
        detection_seconds = time.perf_counter() - started
        detection_timing = self.detector.last_timing if isinstance(self.detector, YOLO26) else None
        return self.estimate_largest(frame, boxes, detection_seconds, detection_timing)

    def estimate_prepared(self, frame: np.ndarray, prepared: PreparedDetection) -> SinglePersonPose:
        """Consume an independently prepared YOLO26 input on the inference thread."""
        assert isinstance(self.detector, YOLO26)
        started = time.perf_counter()
        boxes = self.detector.detect_prepared(prepared)
        detection_seconds = time.perf_counter() - started + prepared.preprocess_seconds
        return self.estimate_largest(frame, boxes, detection_seconds, self.detector.last_timing)

    def estimate_largest(
        self, frame: np.ndarray, boxes: np.ndarray, detection_seconds: float,
        detection_timing: dict[str, float] | None,
    ) -> SinglePersonPose:
        """Select the largest detected person and estimate its whole-body pose."""
        started = time.perf_counter()
        if len(boxes) == 0:
            return SinglePersonPose(None, None, 0, detection_seconds, 0.0, detection_timing)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest = int(np.argmax(areas))
        keypoints, scores = self.pose(frame, bboxes=boxes[largest:largest + 1])
        return SinglePersonPose(
            keypoints, scores, len(boxes), detection_seconds,
            time.perf_counter() - started,
            detection_timing,
        )

    def draw(self, frame: np.ndarray, prediction: SinglePersonPose) -> np.ndarray:
        """Render predictions without accessing inference sessions."""
        if prediction.keypoints is None:
            return frame
        return draw_skeleton(frame, prediction.keypoints, prediction.scores, kpt_thr=self.threshold)


def load_pose(inference: dict) -> BalancedPose:
    """Validate inference configuration and initialize the selected engine."""
    engine = inference["engine"]
    detector_model = configured_detector(inference)
    if engine not in ("tensorrt", "onnxruntime"):
        raise ValueError('engine 必须为 "tensorrt" 或 "onnxruntime"。')
    device_id = inference["device_id"]
    threshold = inference["keypoint_threshold"]
    if device_id < 0:
        raise ValueError("device_id 必须非负。")
    if not 0 <= threshold <= 1:
        raise ValueError("keypoint_threshold 必须介于 0 和 1。")
    LOGGER.info("Selected inference engine=%s, detector=%s", engine, detector_model)
    if engine == "onnxruntime":
        return BalancedPose.from_cuda(device_id, threshold, detector_model)
    workspace_mb = inference["workspace_mb"]
    if workspace_mb <= 0:
        raise ValueError("workspace_mb 必须大于 0。")
    return BalancedPose.from_tensorrt(device_id, workspace_mb, threshold, detector_model)


def main() -> int:
    """Build, warm up and persist both FP16 engines without opening the camera."""
    configure_logging()
    try:
        with (ROOT / "config" / "config.toml").open("rb") as stream:
            inference = tomllib.load(stream)["inference"]
        LOGGER.info("Preparing %s and RTMW-X TensorRT FP16 engines", configured_detector(inference))
        load_pose({**inference, "engine": "tensorrt"})
        LOGGER.info("Both TensorRT FP16 engines are cached. Start the preview with make run.")
    except Exception:
        LOGGER.exception("TensorRT engine preparation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
