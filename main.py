from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from analysis.analysis import run_analysis
from scripts.classifier import classify_nature_alignment
from scripts.cleaning import clean_text
from scripts.language import detect_language_label
from scripts.scraper import YouTubeCommentScraper
from scripts.sentiment import sentiment_label, sentiment_score

def _slugify(value: str) -> str:
    v = (value or "").strip().lower()
    v = "".join(ch if ch.isalnum() else "_" for ch in v)
    while "__" in v:
        v = v.replace("__", "_")
    return v.strip("_") or "channel"

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--channels-file", default=str(Path(__file__).resolve().parent / "channels.json"))
    p.add_argument("--channel-name")
    p.add_argument("--channel-id")
    p.add_argument("--video-ids")
    p.add_argument("--per-video", type=int, default=200)
    p.add_argument("--max-videos", type=int)
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "data"))
    p.add_argument("--analysis-dir", default=str(Path(__file__).resolve().parent / "analysis" / "outputs"))
    return p.parse_args()

def _load_channels(channels_file: str) -> list[dict[str, str]]:
    path = Path(channels_file)
    raw = json.loads(path.read_text(encoding="utf-8"))
    channels = raw.get("channels", [])
    if not isinstance(channels, list):
        raise ValueError("channels.json must contain a 'channels' list")
    normalized: list[dict[str, str]] = []
    for item in channels:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        channel_id = str(item.get("channel_id", "")).strip()
        if name:
            normalized.append({"name": name, "channel_id": channel_id})
    if not normalized:
        raise ValueError("No channels found in channels.json")
    return normalized

def _write_channels(channels_file: str, channels: list[dict[str, str]]) -> None:
    path = Path(channels_file)
    payload = {"channels": [{"name": c.get("name", ""), "channel_id": c.get("channel_id", "")} for c in channels]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _get_api_key() -> str:
    load_dotenv()
    for k in ["YOUTUBE_API_KEY", "YOUTUBE_APIKEY", "YT_API_KEY"]:
        v = os.getenv(k)
        if v:
            return v
    raise RuntimeError("YouTube API key not found in .env. Set YOUTUBE_API_KEY.")

def _run_for_channel(scraper: YouTubeCommentScraper,*,channel_name: str,channel_id: str | None,video_ids_arg: str | None,per_video: int,max_videos: int | None,output_dir: Path,analysis_dir: Path,) -> Path:
    if channel_id:
        video_ids = scraper.fetch_video_ids_from_channel(channel_id, max_videos=max_videos)
    else:
        video_ids = [v.strip() for v in str(video_ids_arg).split(",") if v.strip()]

    if not video_ids:
        raise ValueError("No video IDs found")

    records = scraper.fetch_comments_for_videos(video_ids, per_video_limit=int(per_video))

    df = pd.DataFrame(
        [
            {
                "Video_Title": r.video_title,
                "Comment_Text": r.comment_text,
                "Comment_Date": r.comment_date,
                "Like_Count": r.like_count,
                "Username": r.username_hash,
            }
            for r in records
        ]
    )

    df["Comment_Text"] = df["Comment_Text"].map(clean_text)
    df["Language_Label"] = df["Comment_Text"].map(detect_language_label)
    df["Sentiment_Score"] = df["Comment_Text"].map(sentiment_score)
    df["Sentiment_Label"] = df["Sentiment_Score"].map(sentiment_label)
    df["Nature_Aligned"] = df["Comment_Text"].map(classify_nature_alignment)

    final = df[[
        "Video_Title",
        "Comment_Text",
        "Comment_Date",
        "Like_Count",
        "Language_Label",
        "Sentiment_Score",
        "Sentiment_Label",
        "Nature_Aligned",
    ]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_slug = _slugify(channel_name)
    csv_path = output_dir / f"{channel_slug}.csv"
    final.to_csv(csv_path, index=False)

    per_channel_analysis = analysis_dir / channel_slug
    run_analysis(csv_path, output_dir=per_channel_analysis)
    return csv_path


def main() -> None:
    args = _parse_args()

    api_key = _get_api_key()
    scraper = YouTubeCommentScraper(api_key)

    out_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)

    if args.all:
        channels = _load_channels(str(args.channels_file))
        updated = False
        for ch in channels:
            name = ch.get("name", "").strip()
            channel_id = ch.get("channel_id", "").strip()
            if not channel_id:
                try:
                    channel_id = scraper.resolve_channel_id(name)
                except Exception as exc:
                    raise SystemExit(f"Missing channel_id for {name} in channels.json and auto-resolve failed: {exc}")
                ch["channel_id"] = channel_id
                updated = True
            _run_for_channel(
                scraper,
                channel_name=name,
                channel_id=channel_id,
                video_ids_arg=None,
                per_video=int(args.per_video),
                max_videos=args.max_videos,
                output_dir=out_dir,
                analysis_dir=analysis_dir,
            )
        if updated:
            _write_channels(str(args.channels_file), channels)
        return

    if not args.channel_name:
        raise SystemExit("Provide --channel-name or use --all")

    if bool(args.channel_id) == bool(args.video_ids):
        raise SystemExit("Provide exactly one of --channel-id or --video-ids")

    _run_for_channel(scraper,channel_name=str(args.channel_name),channel_id=str(args.channel_id) if args.channel_id else None,video_ids_arg=str(args.video_ids) if args.video_ids else None,per_video=int(args.per_video),max_videos=args.max_videos,output_dir=out_dir,analysis_dir=analysis_dir,)

if __name__ == "__main__":
    main()
