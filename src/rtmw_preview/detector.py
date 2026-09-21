"""Adapt YOLO26 end-to-end detections to original-image person boxes."""

import time
from dataclasses import dataclass

import cv2
import numpy as np
from rtmlib.tools.base import BaseTool

DETECTION_THRESHOLD = 0.3
PERSON_CLASS_ID = 0
INPUT_SIZE = (640, 640)


@dataclass
class PreparedDetection:
    """Own a frame's detector input and original-coordinate transform."""

    tensor: np.ndarray
    width: int
    height: int
    scale: float
    left: int
    top: int
    preprocess_seconds: float


class YOLO26(BaseTool):
    """Run a static YOLO26 ONNX detector through rtmlib's runtime interface."""

    def __init__(self, onnx_model: str, device: str) -> None:
        super().__init__(
            onnx_model, model_input_size=INPUT_SIZE,
            backend="onnxruntime", device=device,
        )
        shape = self.session.get_outputs()[0].shape
        if len(shape) != 3 or shape[-1] != 6:
            raise ValueError("YOLO26 requires end-to-end ONNX output shaped [1, N, 6].")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Return float32 person boxes in original-image xyxy coordinates."""
        return self.detect_prepared(self.prepare(frame))

    def prepare(self, frame: np.ndarray) -> PreparedDetection:
        """Prepare an independently owned input without accessing the runtime session."""
        started = time.perf_counter()
        height, width = frame.shape[:2]
        input_width, input_height = self.model_input_size
        scale = min(input_width / width, input_height / height)
        resized_width, resized_height = round(width * scale), round(height * scale)
        left = round((input_width - resized_width) / 2 - 0.1)
        top = round((input_height - resized_height) / 2 - 0.1)
        resized = cv2.resize(frame, (resized_width, resized_height))
        padded = cv2.copyMakeBorder(
            resized, top, input_height - resized_height - top,
            left, input_width - resized_width - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        tensor = np.empty((1, 3, input_height, input_width), dtype=np.float32)
        # Write RGB planes directly into the final tensor without an HWC float buffer.
        for channel in range(3):
            np.divide(
                padded[:, :, 2 - channel], np.float32(255.0),
                out=tensor[0, channel], dtype=np.float32,
            )
        return PreparedDetection(tensor, width, height, scale, left, top, time.perf_counter() - started)

    def detect_prepared(self, prepared: PreparedDetection) -> np.ndarray:
        """Run a prepared input and restore person boxes to original-image coordinates."""
        started = time.perf_counter()
        outputs = self.run_tensor(prepared.tensor)
        output_started = time.perf_counter()
        detections = outputs[0][0]
        inferred = time.perf_counter()
        selected = (
            (detections[:, 4] >= DETECTION_THRESHOLD)
            & (detections[:, 5] == PERSON_CLASS_ID)
        )
        boxes = detections[selected, :4].astype(np.float32, copy=True)
        boxes -= np.array([prepared.left, prepared.top, prepared.left, prepared.top], dtype=np.float32)
        boxes /= prepared.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, prepared.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, prepared.height)
        boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
        self.last_timing = {
            "preprocess": prepared.preprocess_seconds,
            "inference_call": inferred - started,
            "postprocess": time.perf_counter() - inferred,
            **self.last_inference_timing,
            "output_extract": inferred - output_started,
        }
        return boxes

    def inference(self, img: np.ndarray) -> list[np.ndarray]:
        """Accept rtmlib's HWC input for session warmup."""
        tensor = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)[None]
        return self.run_tensor(tensor)

    def run_tensor(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Execute an owned contiguous NCHW float32 tensor without layout conversion."""
        started = time.perf_counter()
        session_input = {self.session.get_inputs()[0].name: tensor}
        output_names = [output.name for output in self.session.get_outputs()]
        feed_finished = time.perf_counter()
        outputs = self.session.run(output_names, session_input)
        run_finished = time.perf_counter()
        del session_input
        cleanup_finished = time.perf_counter()
        self.last_inference_timing = {
            "input_layout": 0.0,
            "feed_setup": feed_finished - started,
            "session_run": run_finished - feed_finished,
            "input_cleanup": cleanup_finished - run_finished,
        }
        return outputs
