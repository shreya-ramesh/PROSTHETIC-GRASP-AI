from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR / "models" / "best.pt")

CONFIDENCE_THRESHOLD = 0.20
BRIGHTNESS_THRESHOLD = 90

GEMINI_MODEL_NAME = "gemini-2.5-flash"
