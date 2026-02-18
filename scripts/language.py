from __future__ import annotations
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0

def detect_language_label(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return "English"
    try:
        code = detect(value)
    except LangDetectException:
        return "English"
    if code == "hi":
        return "Hindi"
    return "English"
