from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def _list_channel_csvs(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    items = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
    return items


def _sentiment_pie(df: pd.DataFrame):
    counts = df["Sentiment_Label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    fig = plt.figure()
    plt.pie(counts.values.tolist(), labels=counts.index.tolist(), autopct="%1.1f%%")
    plt.title("Sentiment")
    return fig


def _timeline_plot(df: pd.DataFrame):
    dates = pd.to_datetime(df["Comment_Date"], errors="coerce", utc=True)
    tmp = df.copy()
    tmp["_date"] = dates.dt.date
    daily = tmp.dropna(subset=["_date"]).groupby("_date")["Sentiment_Score"].mean()
    fig = plt.figure()
    plt.plot(daily.index.astype(str).tolist(), daily.values.tolist())
    plt.xlabel("Comment_Date")
    plt.ylabel("Average Sentiment_Score")
    plt.title("Timeline trend")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def _language_distribution(df: pd.DataFrame):
    counts = df["Language_Label"].value_counts().reindex(["English", "Hindi"]).fillna(0)
    fig = plt.figure()
    plt.bar(counts.index.tolist(), counts.values.tolist())
    plt.xlabel("Language_Label")
    plt.ylabel("Count")
    plt.title("Language distribution")
    plt.tight_layout()
    return fig


def main() -> None:
    st.title("Public Sentiment Analysis")
    data_dir = Path(__file__).resolve().parents[1] / "data"
    csvs = _list_channel_csvs(data_dir)

    if not csvs:
        st.write("No channel CSVs found in data/")
        return

    options = {p.stem: p for p in csvs}
    selected = st.selectbox("Channel", list(options.keys()))
    df = pd.read_csv(options[selected])

    st.pyplot(_sentiment_pie(df))

    total = max(int(len(df)), 1)
    aligned = int((df["Nature_Aligned"] == 1).sum())
    st.metric("Nature-aligned %", f"{aligned / total * 100.0:.2f}")

    st.pyplot(_timeline_plot(df))
    st.pyplot(_language_distribution(df))


if __name__ == "__main__":
    main()
