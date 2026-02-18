from __future__ import annotations

import re

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def clean_text(text: str) -> str:
    if text is None:
        return ""
    value = str(text)
    value = _URL_RE.sub(" ", value)
    value = _EMOJI_RE.sub(" ", value)
    value = value.lower()
    value = re.sub(r"[^0-9a-z\u0900-\u097f\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value
