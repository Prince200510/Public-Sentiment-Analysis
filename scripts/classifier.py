from __future__ import annotations

_NATURE_ALIGNED = [
    "lightness",
    "clarity",
    "natural energy",
    "deep sleep",
    "satvic",
    "prana",
    "shuddhi",
    "tapas",
    "ahimsa",
    "mother earth",
    "seasonal eating",
    "living food",
    "returning to my roots",
    "new person",
]

_NOT_NATURE_ALIGNED = [
    "weight loss",
    "sugar levels",
    "kilograms",
    "science says otherwise",
    "protein deficiency",
    "too expensive",
    "office lunch",
    "society pressure",
    "cant live without tea",
    "can t live without tea",
    "too bland",
]

def classify_nature_alignment(text: str) -> int:
    value = (text or "").lower()
    has_aligned = any(k in value for k in _NATURE_ALIGNED)
    has_not = any(k in value for k in _NOT_NATURE_ALIGNED)
    if has_aligned and not has_not:
        return 1
    if has_not and not has_aligned:
        return 0
    return -1
