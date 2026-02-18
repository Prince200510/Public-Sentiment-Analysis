from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_analysis(csv_path: str | os.PathLike[str], *, output_dir: str | os.PathLike[str]) -> None:
    df = pd.read_csv(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_sentiment_distribution(df, out / "sentiment_distribution.png")
    _plot_nature_alignment_percentage(df, out / "nature_alignment_percentage.png")
    _plot_sentiment_like_scatter(df, out / "sentiment_vs_like_scatter.png")
    _plot_language_sentiment_breakdown(df, out / "language_sentiment_breakdown.png")
    _plot_timeline_trend(df, out / "timeline_trend.png")


def _plot_sentiment_distribution(df: pd.DataFrame, path: Path) -> None:
    counts = df["Sentiment_Label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    plt.figure()
    plt.bar(counts.index.tolist(), counts.values.tolist())
    plt.xlabel("Sentiment_Label")
    plt.ylabel("Count")
    plt.title("Sentiment distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_nature_alignment_percentage(df: pd.DataFrame, path: Path) -> None:
    total = max(int(len(df)), 1)
    aligned = int((df["Nature_Aligned"] == 1).sum())
    not_aligned = int((df["Nature_Aligned"] == 0).sum())
    unknown = total - aligned - not_aligned
    labels = ["Nature-Aligned", "Not Nature-Aligned", "Unknown"]
    values = [aligned / total * 100.0, not_aligned / total * 100.0, unknown / total * 100.0]
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Nature_Aligned")
    plt.ylabel("Percentage")
    plt.title("Nature alignment percentage")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_sentiment_like_scatter(df: pd.DataFrame, path: Path) -> None:
    likes = pd.to_numeric(df["Like_Count"], errors="coerce").fillna(0)
    score = pd.to_numeric(df["Sentiment_Score"], errors="coerce").fillna(0)
    corr = float(score.corr(likes)) if len(df) > 1 else 0.0
    plt.figure()
    plt.scatter(score, likes)
    plt.xlabel("Sentiment_Score")
    plt.ylabel("Like_Count")
    plt.title(f"Sentiment vs Like count correlation: {corr:.4f}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_language_sentiment_breakdown(df: pd.DataFrame, path: Path) -> None:
    pivot = (
        df.pivot_table(
            index="Language_Label",
            columns="Sentiment_Label",
            values="Like_Count",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(index=["English", "Hindi"], fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    )

    plt.figure()
    x = range(len(pivot.index))
    pos = pivot["Positive"].tolist()
    neu = pivot["Neutral"].tolist()
    neg = pivot["Negative"].tolist()

    bottom_neu = pos
    bottom_neg = [pos[i] + neu[i] for i in range(len(pos))]

    plt.bar(x, pos, label="Positive")
    plt.bar(x, neu, bottom=bottom_neu, label="Neutral")
    plt.bar(x, neg, bottom=bottom_neg, label="Negative")

    plt.xticks(list(x), pivot.index.tolist())
    plt.xlabel("Language_Label")
    plt.ylabel("Count")
    plt.title("Hindi vs English sentiment breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_timeline_trend(df: pd.DataFrame, path: Path) -> None:
    dates = pd.to_datetime(df["Comment_Date"], errors="coerce", utc=True)
    tmp = df.copy()
    tmp["_date"] = dates.dt.date
    daily = tmp.dropna(subset=["_date"]).groupby("_date")["Sentiment_Score"].mean()
    plt.figure()
    plt.plot(daily.index.astype(str).tolist(), daily.values.tolist())
    plt.xlabel("Comment_Date")
    plt.ylabel("Average Sentiment_Score")
    plt.title("Timeline trend")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
