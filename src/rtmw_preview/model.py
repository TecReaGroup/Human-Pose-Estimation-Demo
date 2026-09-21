"""Download the official rtmlib balanced model pair."""

import logging
import shutil
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
    """Download both balanced models."""
    configure_logging()
    for name in MODEL_URL:
        download_model(name)
