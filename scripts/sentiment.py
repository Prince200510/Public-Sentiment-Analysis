from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_ANALYZER = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> float:
    return float(_ANALYZER.polarity_scores(text or "").get("compound", 0.0))

def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"
