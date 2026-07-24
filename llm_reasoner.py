import json
import os
from typing import List, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

from config import GEMINI_MODEL_NAME
from grasp_engine import NO_RECOMMENDATION

load_dotenv()


class LLMReasoner:
    def __init__(self, api_key: str = None, model_name: str = GEMINI_MODEL_NAME):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file before running the application."
            )

        genai.configure(api_key=resolved_api_key)
        self._model = genai.GenerativeModel(model_name)

    def explain_batch(self, object_grasp_pairs: List[Tuple[str, str]]) -> List[str]:
        if not object_grasp_pairs:
            return []

        prompt = self._build_prompt(object_grasp_pairs)

        try:
            response = self._model.generate_content(prompt)
            raw_text = response.text.strip()
        except Exception as exc:
            raise RuntimeError(f"Gemini explanation request failed: {exc}") from exc

        return self._parse_response(raw_text, expected_count=len(object_grasp_pairs))

    @staticmethod
    def _build_prompt(object_grasp_pairs: List[Tuple[str, str]]) -> str:
        item_lines = "\n".join(
            f"{index + 1}. Object: {object_name}, Recommended Grasp: {grasp}"
            for index, (object_name, grasp) in enumerate(object_grasp_pairs)
        )

        return (
            "You are assisting a prosthetic hand control system that explains grasp choices "
            "to clinicians and users. For each item listed below, write one concise, "
            "professional explanation (1-2 sentences) of why the recommended grasp suits the "
            "object's physical properties.\n\n"
            f"If the recommended grasp is '{NO_RECOMMENDATION}', explain that the detected "
            "object falls outside the supported dataset and therefore a safe grasp "
            "recommendation cannot be provided.\n\n"
            f"Items:\n{item_lines}\n\n"
            "Respond ONLY with a valid JSON array of strings, containing exactly one "
            "explanation per item, in the same order as listed. Do not include markdown "
            "formatting or any text outside the JSON array."
        )

    @staticmethod
    def _parse_response(raw_text: str, expected_count: int) -> List[str]:
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(cleaned_text)
        except json.JSONDecodeError:
            return [cleaned_text] * expected_count

        if not isinstance(parsed, list) or len(parsed) != expected_count:
            return [str(parsed)] * expected_count

        return [str(explanation) for explanation in parsed]
