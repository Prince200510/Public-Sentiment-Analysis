from __future__ import annotations
import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import pandas as pd

LOGICAL_KEYWORDS = ["weight", "vajan", "loss", "kam", "sugar", "diabetes", "protein", "calories", "evidence", "study", "science", "doctor", "bpm", "medicine", "cure", "bimari", "ilaj", "doctor", "kaise", "kya", "how", "gym",]
CULTURAL_KEYWORDS = ["energy", "urja", "light", "halka", "peace", "shanti", "calm", "fresh", "nature", "prakriti", "transformation", "feeling", "soul", "earth", "natural", "kudrat", "habit", "routine", "prana", "satvic", "shuddhi", "tapas", "ahimsa", "anubhuti", "shakti",]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ASCII_LETTERS_RE = re.compile(r"[^a-z\s]+", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")

def normalize_roman_text(text: str) -> str:
    text = (text or "").strip().lower()
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _NON_ASCII_LETTERS_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text

def _compile_keywords(keywords: Iterable[str]) -> re.Pattern[str]:
    escaped = [re.escape(k.strip().lower()) for k in keywords if k and k.strip()]
    escaped = sorted(set(escaped), key=len, reverse=True)
    if not escaped:
        return re.compile(r"$^")
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)

LOGICAL_RE = _compile_keywords(LOGICAL_KEYWORDS)
CULTURAL_RE = _compile_keywords(CULTURAL_KEYWORDS)

@dataclass(frozen=True)
class PhaseResult:
    phase_name: str
    token_counts: Counter[str]
    matched_comments: int
    total_comments: int

def tokenize(text: str, extra_stopwords: set[str]) -> list[str]:
    tokens = [t for t in text.split() if len(t) >= 2]
    return [t for t in tokens if t not in extra_stopwords]

def build_phase_result(comments: Iterable[str], matcher: re.Pattern[str], phase_name: str, extra_stopwords: set[str],) -> PhaseResult:
    total = 0
    matched = 0
    counts: Counter[str] = Counter()

    for raw in comments:
        total += 1
        norm = normalize_roman_text(str(raw) if raw is not None else "")
        if not norm:
            continue
        if not matcher.search(norm):
            continue
        matched += 1

        for token in tokenize(norm, extra_stopwords):
            counts[token] += 1

    return PhaseResult(phase_name=phase_name, token_counts=counts, matched_comments=matched, total_comments=total,)

def _default_stopwords() -> set[str]:
    return {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "she", "so", "that", "the", "their", "them", "there", "they", "this", "to", "up", "was", "we", "were", "what", "when", "which", "who", "with", "you", "your",}

def save_wordcloud_from_counts(counts: Counter[str], out_path: Path, title: str, max_words: int = 180,) -> None:
    items = counts.most_common(max_words)
    if not items:
        return

    freqs = [c for _, c in items]
    max_c = max(freqs) if freqs else 1
    min_c = min(freqs) if freqs else 1

    def scale_font(c: int) -> float:
        if max_c == min_c:
            return 40
        return 14 + (c - min_c) * (76 / (max_c - min_c))

    cols, rows = 18, 10
    xs = [i / (cols - 1) for i in range(cols)]
    ys = [i / (rows - 1) for i in range(rows)]

    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title)

    idx = 0
    for word, c in items:
        x = xs[idx % cols]
        y = ys[(idx // cols) % rows]
        fontsize = scale_font(c)
        ax.text(
            x,
            y,
            word,
            fontsize=fontsize,
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="black",
            alpha=0.9,
        )
        idx += 1

    fig.tight_layout(pad=1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_top_keywords_bar(counts: Counter[str], out_path: Path, title: str, top_n: int,) -> None:
    items = counts.most_common(top_n)
    if not items:
        return
    words = [w for w, _ in items][::-1]
    freqs = [c for _, c in items][::-1]

    fig = plt.figure(figsize=(10, 8), dpi=150)
    plt.barh(words, freqs)
    plt.title(title)
    plt.xlabel("Count")
    fig.tight_layout(pad=1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_counts_csv(counts: Counter[str], out_path: Path) -> None:
    if not counts:
        return
    df = pd.DataFrame(counts.most_common(), columns=["token", "count"])
    df.to_csv(out_path, index=False)

def process_file(csv_path: Path, out_dir: Path, top_n: int) -> None:
    df = pd.read_csv(csv_path)
    if "Comment_Text" not in df.columns:
        raise ValueError(f"Missing Comment_Text column in {csv_path.name}")

    comments = df["Comment_Text"].astype(str).tolist()
    channel_name = csv_path.stem
    channel_out = out_dir / channel_name
    channel_out.mkdir(parents=True, exist_ok=True)
    extra_stopwords = _default_stopwords()
    extra_stopwords.update({"video", "channel", "subscribe", "thanks", "thank", "u", "ur", "please", "sir", "mam", "guruji", "ji", "bhai", "didi", "bro", "hi", "hello", "nice", "good", "great", "best",})
    logical = build_phase_result(comments, matcher=LOGICAL_RE, phase_name="Logical Entry (Gatekeeper)", extra_stopwords=extra_stopwords,)
    cultural = build_phase_result(comments, matcher=CULTURAL_RE, phase_name="Cultural Alignment (Sustainer)", extra_stopwords=extra_stopwords,)
    save_wordcloud_from_counts(logical.token_counts, channel_out / "logical_phase_wordcloud.png", f"{channel_name} - {logical.phase_name} (matched {logical.matched_comments}/{logical.total_comments})",)
    save_top_keywords_bar(logical.token_counts, channel_out / "logical_phase_top_keywords.png", f"Top Keywords - {channel_name} - {logical.phase_name}", top_n=top_n,)
    save_counts_csv(logical.token_counts, channel_out / "logical_phase_token_counts.csv")
    save_wordcloud_from_counts(cultural.token_counts, channel_out / "cultural_phase_wordcloud.png", f"{channel_name} - {cultural.phase_name} (matched {cultural.matched_comments}/{cultural.total_comments})",)
    save_top_keywords_bar(cultural.token_counts, channel_out / "cultural_phase_top_keywords.png", f"Top Keywords - {channel_name} - {cultural.phase_name}", top_n=top_n,)
    save_counts_csv(cultural.token_counts, channel_out / "cultural_phase_token_counts.csv")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"), help="Folder containing channel CSVs",)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"), help="Output folder for images and CSVs",)
    parser.add_argument("--top-n", type=int, default=25, help="Top N keywords to chart")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    for csv_path in csv_files:
        process_file(csv_path, out_dir, top_n=args.top_n)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
