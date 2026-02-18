from __future__ import annotations
import hashlib
import re
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

@dataclass(frozen=True)
class CommentRecord:
    video_title: str
    comment_text: str
    comment_date: str
    like_count: int
    username_hash: str

class YouTubeCommentScraper:
    def __init__(self, api_key: str, *, sleep_seconds: float = 0.1) -> None:
        if not api_key:
            raise ValueError("Missing YouTube API key")
        self._youtube = build("youtube", "v3", developerKey=api_key)
        self._sleep_seconds = float(sleep_seconds)

    def fetch_video_ids_from_channel(self, channel_id: str, *, max_videos: Optional[int] = None) -> list[str]:
        uploads_playlist_id = self._get_uploads_playlist_id(channel_id)
        video_ids: list[str] = []
        page_token: Optional[str] = None
        while True:
            request = self._youtube.playlistItems().list(part="contentDetails", playlistId=uploads_playlist_id, maxResults=50, pageToken=page_token,)
            response = self._execute_with_retry(request)
            for item in response.get("items", []):
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    video_ids.append(vid)
                    if max_videos is not None and len(video_ids) >= max_videos:
                        return video_ids
            page_token = response.get("nextPageToken")
            if not page_token:
                return video_ids

    def resolve_channel_id(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("Missing channel identifier")

        channel_id = self._try_extract_channel_id(raw)
        if channel_id:
            return channel_id

        handle = self._try_extract_handle(raw)
        if handle:
            channel_id = self._resolve_channel_id_from_handle(handle)
            if channel_id:
                return channel_id

        channel_id = self._resolve_channel_id_from_search(raw)
        if channel_id:
            return channel_id

        raise ValueError("Unable to resolve channel_id")

    def _try_extract_channel_id(self, raw: str) -> str | None:
        m = re.search(r"(?i)(?:youtube\.com/channel/)(UC[0-9A-Za-z_-]{20,})", raw)
        if m:
            return m.group(1)
        if raw.startswith("UC") and len(raw) >= 20:
            return raw
        return None

    def _try_extract_handle(self, raw: str) -> str | None:
        m = re.search(r"(?i)(?:youtube\.com/)(@[^/?#]+)", raw)
        if m:
            return m.group(1)
        if raw.startswith("@") and len(raw) > 1:
            return raw
        return None

    def _resolve_channel_id_from_handle(self, handle: str) -> str | None:
        h = handle.lstrip("@").strip()
        if not h:
            return None
        request = self._youtube.channels().list(part="id", forHandle=h, maxResults=1)
        response = self._execute_with_retry(request)
        items = response.get("items", [])
        if not items:
            return None
        return str(items[0].get("id", "")).strip() or None

    def _resolve_channel_id_from_search(self, query: str) -> str | None:
        request = self._youtube.search().list(part="snippet", q=query, type="channel", maxResults=1)
        response = self._execute_with_retry(request)
        items = response.get("items", [])
        if not items:
            return None
        channel_id = items[0].get("snippet", {}).get("channelId")
        if channel_id:
            return str(channel_id).strip() or None
        return None

    def fetch_comments_for_videos(self, video_ids: Iterable[str],*, per_video_limit: int,) -> list[CommentRecord]:
        if per_video_limit < 100 or per_video_limit > 500:
            raise ValueError("per_video_limit must be between 100 and 500")
        all_records: list[CommentRecord] = []
        for video_id in video_ids:
            title = self._get_video_title(video_id)
            records = list(self._fetch_top_level_comments(video_id, title, limit=per_video_limit))
            all_records.extend(records)
        return all_records

    def _get_uploads_playlist_id(self, channel_id: str) -> str:
        request = self._youtube.channels().list(part="contentDetails", id=channel_id, maxResults=1)
        response = self._execute_with_retry(request)
        items = response.get("items", [])
        if not items:
            raise ValueError("Channel not found")
        uploads = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        if not uploads:
            raise ValueError("Uploads playlist not available for channel")
        return str(uploads)

    def _get_video_title(self, video_id: str) -> str:
        request = self._youtube.videos().list(part="snippet", id=video_id, maxResults=1)
        response = self._execute_with_retry(request)
        items = response.get("items", [])
        if not items:
            return ""
        return str(items[0].get("snippet", {}).get("title", ""))

    def _fetch_top_level_comments(self, video_id: str, video_title: str, *, limit: int) -> Iterator[CommentRecord]:
        page_token: Optional[str] = None
        fetched = 0
        while fetched < limit:
            batch_size = min(100, limit - fetched)
            request = self._youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                pageToken=page_token,
                textFormat="plainText",
                order="relevance",
            )
            try:
                response = self._execute_with_retry(request)
            except HttpError as exc:
                if getattr(exc, "status_code", None) in {403, 404}:
                    return
                raise

            items = response.get("items", [])
            for item in items:
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = str(snippet.get("textDisplay", ""))
                published_at = str(snippet.get("publishedAt", ""))
                like_count = int(snippet.get("likeCount", 0) or 0)
                author = str(snippet.get("authorDisplayName", ""))
                username_hash = hashlib.sha256(author.encode("utf-8")).hexdigest()
                yield CommentRecord(
                    video_title=video_title,
                    comment_text=text,
                    comment_date=published_at,
                    like_count=like_count,
                    username_hash=username_hash,
                )
                fetched += 1
                if fetched >= limit:
                    return

            page_token = response.get("nextPageToken")
            if not page_token:
                return

    def _execute_with_retry(self, request, *, max_retries: int = 5):
        attempt = 0
        while True:
            try:
                if self._sleep_seconds > 0:
                    time.sleep(self._sleep_seconds)
                return request.execute()
            except HttpError as exc:
                attempt += 1
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "resp", None), "status", None)
                if attempt > max_retries:
                    raise
                if status in {429, 500, 503}:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
