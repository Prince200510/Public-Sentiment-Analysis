import os
import glob
import hashlib
import re
import argparse
from datetime import datetime

import pandas as pd
try:
    from langdetect import detect
except Exception:
    detect = None


POSITIVE_KEYWORDS = [
    r"\bhalka\b",
    r"lightness",
    r"clarity",
    r"natural energy",
    r"deep sleep",
    r"satvic",
    r"prana",
    r"shuddhi",
    r"tapas",
    r"ahimsa",
    r"mother earth",
    r"seasonal eating",
    r"living food",
    r"returning to my roots",
    r"i feel like a new person",
]

NEGATIVE_KEYWORDS = [
    r"weight loss",
    r"sugar",
    r"kilograms",
    r"protein deficiency",
    r"too expensive",
    r"family won't allow",
    r"office lunch",
    r"society pressure",
    r"i can't live without tea",
    r"too bland",
    r"tasteless",
    r"milk",
    r"medicine",
]


def anonymize(value: str) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    h = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return h[:12]


def detect_language(text: str) -> str:
    if text is None:
        return ""
    if detect is None:
        return ""
    try:
        return detect(text)
    except Exception:
        return ""


def keyword_label(text: str) -> int:
    if not isinstance(text, str) or text.strip() == "":
        return -1
    t = text.lower()
    pos = any(re.search(p, t) for p in POSITIVE_KEYWORDS)
    neg = any(re.search(p, t) for p in NEGATIVE_KEYWORDS)
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    if pos and neg:
        return 1
    return -1


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("video_title", "video title"):
            mapping[c] = "Video_Title"
        if lc in ("comment_text", "comment text", "comment"):
            mapping[c] = "Comment_Text"
        if lc in ("comment_date", "comment date", "date"):
            mapping[c] = "Comment_Date"
        if lc in ("like_count", "likes", "like"):
            mapping[c] = "Like_Count"
        if lc in ("language_label", "language"):
            mapping[c] = "Language_Label"
        if lc in ("username", "user", "author"):
            mapping[c] = "Username"
        if lc in ("nature_aligned",):
            mapping[c] = "Nature_Aligned_Manual"

    df = df.rename(columns=mapping)

    if "Video_Title" not in df.columns:
        df["Video_Title"] = ""
    if "Comment_Text" not in df.columns:
        df["Comment_Text"] = ""
    if "Comment_Date" in df.columns:
        df["Comment_Date"] = pd.to_datetime(df["Comment_Date"], errors="coerce")
    else:
        df["Comment_Date"] = pd.NaT

    if "Like_Count" in df.columns:
        df["Like_Count"] = pd.to_numeric(df["Like_Count"], errors="coerce").fillna(0).astype(int)
    else:
        df["Like_Count"] = 0

    if "Username" not in df.columns:
        df["Username"] = ""

    df["Username_Anon"] = df["Username"].apply(anonymize)

    if "Language_Label" not in df.columns:
        df["Language_Label"] = df["Comment_Text"].apply(detect_language)

    df["Nature_Aligned_Auto"] = df["Comment_Text"].fillna("").apply(keyword_label)

    return df[["Video_Title", "Comment_Text", "Comment_Date", "Like_Count", "Language_Label", "Username_Anon", "Nature_Aligned_Manual", "Nature_Aligned_Auto"]]

def process_file(path: str, out_dir: str, top_k: int = 100):
    df = pd.read_csv(path)
    df_norm = normalize_df(df)
    basename = os.path.splitext(os.path.basename(path))[0]
    channel_out = os.path.join(out_dir, basename)
    os.makedirs(channel_out, exist_ok=True)
    processed_path = os.path.join(channel_out, f"{basename}_processed.csv")
    df_norm.to_csv(processed_path, index=False)
    top = df_norm.sort_values("Like_Count", ascending=False).head(top_k)
    top_path = os.path.join(channel_out, f"{basename}_top_{top_k}_liked.csv")
    top.to_csv(top_path, index=False)

    try:
        xlsx_path = os.path.join(channel_out, f"{basename}_processed.xlsx")
        with pd.ExcelWriter(xlsx_path) as w:
            df_norm.to_excel(w, sheet_name="processed", index=False)
            top.to_excel(w, sheet_name="top_liked", index=False)
    except Exception:
        pass

    return df_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Directory with input CSV files")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "..", "analysis", "outputs"), help="Directory to write outputs")
    parser.add_argument("--top_k", type=int, default=100, help="Top K liked comments to save per channel")
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    for f in glob.glob(os.path.join(input_dir, "*.csv")):
        print("Processing:", f)
        df_norm = process_file(f, output_dir, top_k=args.top_k)
        df_norm["Source_File"] = os.path.basename(f)
        all_dfs.append(df_norm)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(output_dir, "all_channels_comments.csv")
        combined.to_csv(combined_path, index=False)
        try:
            combined.to_excel(os.path.join(output_dir, "all_channels_comments.xlsx"), index=False)
        except Exception:
            pass

    print("Done. Outputs are in:", output_dir)

if __name__ == "__main__":
    main()
