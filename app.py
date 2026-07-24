from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from annotator import annotate_image
from detector import Detection, ObjectDetector
from grasp_engine import GraspEngine, NO_RECOMMENDATION
from llm_reasoner import LLMReasoner
from preprocess import preprocess_image

st.set_page_config(page_title="Object Detection + Prosthetic Grasp Recommendation", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }
        h1 {
            font-size: 1.6rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

IMAGE_DISPLAY_WIDTH = 380


@st.cache_resource
def load_detector() -> ObjectDetector:
    return ObjectDetector()


@st.cache_resource
def load_reasoner() -> LLMReasoner:
    return LLMReasoner()


def run_pipeline(
    pil_image: Image.Image,
) -> Tuple[np.ndarray, List[Detection], List[str], List[str], np.ndarray]:
    image_array = np.array(pil_image.convert("RGB"))
    preprocessed_image = preprocess_image(image_array)

    detector = load_detector()
    detections = detector.detect(preprocessed_image)

    if not detections:
        return preprocessed_image, [], [], [], preprocessed_image

    grasp_engine = GraspEngine()
    grasps = [grasp_engine.recommend(detection.label) for detection in detections]

    reasoner = load_reasoner()
    object_grasp_pairs = [(detection.label, grasp) for detection, grasp in zip(detections, grasps)]
    reasons = reasoner.explain_batch(object_grasp_pairs)

    annotated_image = annotate_image(preprocessed_image, detections, grasps)

    return preprocessed_image, detections, grasps, reasons, annotated_image


def render_prediction_table(detections: List[Detection], grasps: List[str], reasons: List[str]) -> None:
    table_rows = [
        {
            "Object": detection.label,
            "Confidence": f"{detection.confidence:.2f}",
            "Recommended Grasp": grasp,
            "Reason": reason,
        }
        for detection, grasp, reason in zip(detections, grasps, reasons)
    ]

    st.subheader("Prediction Table")
    st.table(table_rows)


def main() -> None:
    st.title("Object Detection + Prosthetic Grasp Recommendation")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    pil_image = Image.open(uploaded_file)

    if not st.button("Predict"):
        st.image(pil_image, caption="Uploaded Image", width=IMAGE_DISPLAY_WIDTH)
        return

    with st.spinner("Running detection and grasp recommendation..."):
        try:
            preprocessed_image, detections, grasps, reasons, annotated_image = run_pipeline(pil_image)
        except Exception as exc:
            st.error(f"An error occurred during prediction: {exc}")
            return

    original_column, annotated_column = st.columns(2)
    with original_column:
        st.subheader("Original Image")
        st.image(pil_image, width=IMAGE_DISPLAY_WIDTH)

    with annotated_column:
        st.subheader("Annotated Image")
        st.image(annotated_image, width=IMAGE_DISPLAY_WIDTH)

    if not detections:
        st.warning("No objects were detected above the confidence threshold.")
        return

    st.success(f"Detected {len(detections)} object(s).")

    for detection, grasp in zip(detections, grasps):
        if grasp == NO_RECOMMENDATION:
            st.warning(f"'{detection.label}' is not supported by the current grasp dataset.")

    render_prediction_table(detections, grasps, reasons)


if __name__ == "__main__":
    main()