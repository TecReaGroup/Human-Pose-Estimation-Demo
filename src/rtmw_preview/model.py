"""Prepare detector and whole-body pose ONNX models."""

import logging
import os
import shutil
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from rtmw_preview.runtime import ROOT, configure_logging

MODEL_ROOT = ROOT / "data" / "model"
MODEL_URL = {
    "yolox_m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "rtmw_x": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
}
LOGGER = logging.getLogger("model")
DETECTOR_MODELS = ("yolox_m", "yolo26m")


def configured_detector(inference: dict) -> str:
    """Validate the configured detector at the configuration boundary."""
    name = inference.get("detector_model", "yolox_m")
    if name not in DETECTOR_MODELS:
        raise ValueError(f"detector_model 必须为以下之一：{', '.join(DETECTOR_MODELS)}")
    return name


def export_yolo26m() -> Path:
    """Export official YOLO26m weights to a static end-to-end ONNX model."""
    target = MODEL_ROOT / "yolo26m.onnx"
    if target.is_file():
        return target
    temporary = ROOT / "temp" / "export"
    temporary.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "temp" / "ultralytics"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    weights = MODEL_ROOT / "yolo26m.pt"
    if not weights.is_file():
        pending = temporary / "yolo26m.pt.part"
        LOGGER.info("Downloading official YOLO26m weights to %s", weights)
        try:
            url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
            with urllib.request.urlopen(url, timeout=60) as response, pending.open("wb") as output:
                shutil.copyfileobj(response, output)
            pending.replace(weights)
        finally:
            pending.unlink(missing_ok=True)
    export_weights = temporary / weights.name
    shutil.copyfile(weights, export_weights)
    LOGGER.info("Exporting YOLO26m ONNX (640x640, end-to-end, opset 17)")
    exported = Path(YOLO(str(export_weights)).export(
        format="onnx", imgsz=640, batch=1, dynamic=False, quantize=32,
        simplify=False, opset=17, nms=False, device="cpu",
    ))
    exported.replace(target)
    export_weights.unlink()
    LOGGER.info("Model ready: %s", target)
    return target


def download_model(name: str) -> Path:
    """Download and atomically publish one ONNX model from its official archive."""
    target = MODEL_ROOT / f"{name}.onnx"
    if target.is_file():
        return target
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / "temp" / "download"
    temporary.mkdir(parents=True, exist_ok=True)
    archive = temporary / f"{name}.zip"
    pending = temporary / f"{name}.onnx.part"
    LOGGER.info("Downloading %s to %s", name, MODEL_ROOT)
    try:
        with urllib.request.urlopen(MODEL_URL[name], timeout=60) as response:
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as bundle:
            candidates = [entry for entry in bundle.namelist() if entry.endswith(".onnx")]
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one ONNX model in {archive}: {candidates}")
            with bundle.open(candidates[0]) as source, pending.open("wb") as output:
                shutil.copyfileobj(source, output)
        pending.replace(target)
    finally:
        archive.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)
    LOGGER.info("Model ready: %s", target)
    return target


def main() -> None:
    """Prepare the configured detector and the whole-body pose model."""
    configure_logging()
    with (ROOT / "config" / "config.toml").open("rb") as stream:
        detector = configured_detector(tomllib.load(stream)["inference"])
    if detector == "yolo26m":
        export_yolo26m()
    else:
        download_model(detector)
    download_model("rtmw_x")
