from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, MODEL_PATH


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]


class ObjectDetector:
    def __init__(self, model_path: str = MODEL_PATH, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self._confidence_threshold = confidence_threshold
        self._model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> YOLO:
        try:
            return YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_path}'. "
                "Ensure the model has been trained and models/best.pt exists."
            ) from exc

    def detect(self, image: np.ndarray) -> List[Detection]:
        try:
            results = self._model.predict(source=image, conf=self._confidence_threshold, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"Object detection inference failed: {exc}") from exc

        if not results:
            return []

        return self._extract_detections(results[0])

    def _extract_detections(self, result) -> List[Detection]:
        class_names = result.names
        detections: List[Detection] = []

        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < self._confidence_threshold:
                continue

            class_id = int(box.cls[0])
            label = class_names.get(class_id, f"class_{class_id}")
            x1, y1, x2, y2 = (float(coordinate) for coordinate in box.xyxy[0])

            detections.append(Detection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2)))

        return detections
