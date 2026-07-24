import cv2
import numpy as np

from config import BRIGHTNESS_THRESHOLD


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    channel_count = image.shape[2]
    if channel_count == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if channel_count == 3:
        return image

    raise ValueError(f"Unsupported image format with {channel_count} channels")


def estimate_brightness(rgb_image: np.ndarray) -> float:
    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return float(np.mean(grayscale))


def apply_clahe(rgb_image: np.ndarray) -> np.ndarray:
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_image(image: np.ndarray, brightness_threshold: float = BRIGHTNESS_THRESHOLD) -> np.ndarray:
    rgb_image = to_rgb(image)
    brightness = estimate_brightness(rgb_image)

    if brightness < brightness_threshold:
        return apply_clahe(rgb_image)

    return rgb_image
