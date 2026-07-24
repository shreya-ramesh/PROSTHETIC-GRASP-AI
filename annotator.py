from typing import List

import cv2
import numpy as np

from detector import Detection

_BOX_COLOR = (0, 200, 0)
_TEXT_COLOR = (0 , 0 , 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1


def annotate_image(image: np.ndarray, detections: List[Detection], grasps: List[str]) -> np.ndarray:
    annotated_image = image.copy()

    for detection, grasp in zip(detections, grasps):
        _draw_detection(annotated_image, detection, grasp)

    return annotated_image


def _draw_detection(image: np.ndarray, detection: Detection, grasp: str) -> None:
    x1, y1, x2, y2 = (int(round(coordinate)) for coordinate in detection.bbox)
    label = f"{detection.label} | {detection.confidence:.2f} | {grasp}"

    cv2.rectangle(image, (x1, y1), (x2, y2), _BOX_COLOR, thickness=2)

    (text_width, text_height), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    label_top = max(y1, text_height + baseline)

    cv2.rectangle(
        image,
        (x1, label_top - text_height - baseline),
        (x1 + text_width, label_top),
        _BOX_COLOR,
        thickness=-1,
    )
    cv2.putText(
        image,
        label,
        (x1, label_top - baseline // 2),
        _FONT,
        _FONT_SCALE,
        _TEXT_COLOR,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
