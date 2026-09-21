"""rtmlib preprocessing and decoding with explicit TensorRT FP16 sessions."""

import logging
import os
import sys
import time
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose, YOLOX, draw_skeleton

from rtmw_preview.model import download_model
from rtmw_preview.runtime import ROOT, configure_logging

LOGGER = logging.getLogger("inference")
TRT_PROVIDER = "TensorrtExecutionProvider"
# TensorRT 10.9 limits TopK to K <= 3840; the detector exceeds this limit.
TRT_EXCLUDED_OP = "TopK"
DLL_DIRECTORY = []


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


class BalancedPose:
    """Own the balanced detector and whole-body pose sessions."""

    def __init__(self, device_id: int, workspace_mb: int, keypoint_threshold: float) -> None:
        load_gpu_runtime()
        self.threshold = keypoint_threshold
        self.detector = YOLOX(
            str(download_model("yolox_m")), model_input_size=(640, 640),
            backend="onnxruntime", device="cpu",
        )
        self.pose = RTMPose(
            str(download_model("rtmw_x")), model_input_size=(192, 256),
            backend="onnxruntime", device="cpu",
        )
        for name, estimator in (("yolox_m", self.detector), ("rtmw_x", self.pose)):
            started = time.perf_counter()
            estimator.session.disable_fallback()
            runtime_version = f"ort_{ort.__version__}_trt_{version('tensorrt-cu12')}"
            cache = ROOT / "temp" / "engine" / runtime_version / "exclude_topk" / name
            cache.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Building/loading %s TensorRT FP16 engine; first build may take minutes", name)
            LOGGER.info("%s: executing %s outside TensorRT via ORT CUDA/CPU", name, TRT_EXCLUDED_OP)
            estimator.session.set_providers([
                (TRT_PROVIDER, {
                    "device_id": device_id,
                    "trt_fp16_enable": True,
                    "trt_op_types_to_exclude": TRT_EXCLUDED_OP,
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
            LOGGER.info(
                "%s: %d engine file(s) persisted to %s, initialization took %.1fs",
                name, len(engines), cache, time.perf_counter() - started,
            )
            LOGGER.info("%s ready, providers=%s", name, estimator.session.get_providers())

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Detect people and draw 133-keypoint whole-body skeletons."""
        boxes = self.detector(frame)
        if len(boxes) == 0:
            return frame
        keypoints, scores = self.pose(frame, bboxes=boxes)
        return draw_skeleton(frame, keypoints, scores, kpt_thr=self.threshold)


def main() -> int:
    """Build, warm up and persist both FP16 engines without opening the camera."""
    configure_logging()
    try:
        with (ROOT / "config" / "config.toml").open("rb") as stream:
            inference = tomllib.load(stream)["inference"]
        LOGGER.info("Preparing YOLOX-M and RTMW-X TensorRT FP16 engines")
        BalancedPose(**inference)
        LOGGER.info("Both TensorRT FP16 engines are cached. Start the preview with make run.")
    except Exception:
        LOGGER.exception("TensorRT engine preparation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
