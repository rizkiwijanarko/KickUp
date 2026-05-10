"""YouTube Comments scraper — extracts pain points from video comments.

Uses YouTube Data API v3 to:
  1. Search for videos related to domain + complaint keywords
  2. Fetch top-level comments from relevant videos
  3. Filter comments for pain point indicators (frustration, problems, complaints)
  4. Return structured comment data with verbatim text + video URLs

Requires: YOUTUBE_API_KEY in environment
Get API key at: https://console.cloud.google.com/apis/credentials
Free quota: 10,000 units/day (1 search = 100 units, 1 comment thread = 1 unit)

API Documentation: https://developers.google.com/youtube/v3/docs
"""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlencode

import diskcache
import requests

from src.config import settings
from src.tools.reddit_scraper import ScrapedComment

logger = logging.getLogger(__name__)

# Disk-backed cache for YouTube API responses
_CACHE = diskcache.Cache(settings.cache_dir)
_TTL_S: int = settings.cache_ttl_hours * 3600

# ------------------------------------------------------------------
# Search parameters
# ------------------------------------------------------------------
_SEARCH_QUERIES: list[str] = [
    "{domain} frustrating",
    "{domain} problems",
    "{domain} challenges",
    "{domain} pain points",
    "{domain} issues",
    "why I hate {domain}",
    "{domain} worst parts",
]

_COMPLAINT_KEYWORDS: list[str] = [
    "frustrated", "frustrating", "frustration",
    "hate", "hating", "hated",
    "annoying", "annoyed", "annoy",
    "problem", "problems", "problematic",
    "issue", "issues",
    "struggle", "struggling", "struggled",
    "pain", "painful",
    "wish", "wished", "wishing",
    "difficult", "difficulty",
    "terrible", "terribly",
    "awful", "awfully",
    "bad", "badly", "worse", "worst",
    "sucks", "sucked", "suck",
    "complaint", "complain", "complaining",
    "nightmare",
    "broken",
    "useless",
    "waste", "wasted", "wasting",
    "disappointed", "disappointing", "disappointment",
]

_MAX_VIDEOS_PER_QUERY: int = 3
_MAX_COMMENTS_PER_VIDEO: int = 50
_MAX_TOTAL_COMMENTS: int = 200

# YouTube API endpoints
_YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------
def _make_request(endpoint: str, params: dict[str, Any]) -> dict | None:
    """Make a cached GET request to YouTube API."""
    if not settings.youtube_api_key:
        logger.warning("[youtube] YOUTUBE_API_KEY not set")
        return None

    params["key"] = settings.youtube_api_key
    url = f"{_YOUTUBE_API_BASE}/{endpoint}"
    cache_key = f"youtube:{endpoint}:{urlencode(sorted(params.items()))}"

    # Check cache
    cached = _CACHE.get(cache_key)
    if cached is not None:
        logger.debug(f"[youtube] Cache hit for {endpoint}")
        return cached

    # Make request
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Cache successful response
        _CACHE.set(cache_key, data, expire=_TTL_S)
        return data
    except requests.exceptions.RequestException as e:
        logger.warning(f"[youtube] API request failed: {e}")
        return None


def _search_videos(query: str, max_results: int = 5) -> list[dict]:
    """Search for videos matching query, filtering for those with comments enabled.
    
    Returns list of video items with id and snippet.
    Cost: 100 units per search call + 1 unit per video statistics check.
    """
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results * 2,  # Request more to account for filtering
        "order": "relevance",
        "videoDefinition": "any",
        "relevanceLanguage": "en",
    }
    
    data = _make_request("search", params)
    if not data or "items" not in data:
        return []
    
    # Filter for videos with comments enabled
    videos_with_comments = []
    for item in data["items"]:
        video_id = item["id"]["videoId"]
        
        # Check if comments are enabled via statistics API
        stats_params = {
            "part": "statistics",
            "id": video_id,
        }
        stats_data = _make_request("videos", stats_params)
        
        if stats_data and "items" in stats_data and len(stats_data["items"]) > 0:
            stats = stats_data["items"][0].get("statistics", {})
            comment_count = int(stats.get("commentCount", 0))
            
            if comment_count > 0:
                videos_with_comments.append(item)
                
                if len(videos_with_comments) >= max_results:
                    break
    
    logger.debug(f"[youtube] Filtered {len(videos_with_comments)}/{len(data['items'])} videos with comments enabled")
    return videos_with_comments


def _get_video_comments(video_id: str, max_results: int = 50) -> list[dict]:
    """Fetch top-level comments for a video.
    
    Returns list of comment items with snippet.
    Cost: 1 unit per call.
    """
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "order": "relevance",
        "textFormat": "plainText",
    }
    
    data = _make_request("commentThreads", params)
    if not data or "items" not in data:
        return []
    
    return data["items"]


# ------------------------------------------------------------------
# Comment filtering
# ------------------------------------------------------------------
def _has_complaint_signal(text: str) -> bool:
    """Check if comment contains complaint/frustration keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in _COMPLAINT_KEYWORDS)


def _is_substantial(text: str) -> bool:
    """Filter out short/spam comments."""
    # At least 20 characters and 3 words
    return len(text) >= 20 and len(text.split()) >= 3


# ------------------------------------------------------------------
# Main scraper
# ------------------------------------------------------------------
def scrape_for_domain(domain: str, max_total_comments: int = 200) -> list[ScrapedComment]:
    """Scrape YouTube comments for pain points in the given domain.
    
    Strategy:
    1. Search for videos using complaint-focused queries
    2. Fetch comments from top relevant videos
    3. Filter for comments with complaint signals
    4. Return as ScrapedComment objects
    
    Args:
        domain: Target domain (e.g., "developer tools", "meal prep")
        max_total_comments: Maximum comments to return
        
    Returns:
        List of ScrapedComment objects with text, url, subreddit="youtube", post_title=video_title
    """
    if not settings.youtube_api_key:
        logger.warning("[youtube] YOUTUBE_API_KEY not set, skipping YouTube scraper")
        return []
    
    logger.info(f"[youtube] Scraping comments for domain: {domain}")
    
    all_comments: list[ScrapedComment] = []
    seen_comment_ids: set[str] = set()
    
    # Try multiple search queries
    for query_template in _SEARCH_QUERIES:
        if len(all_comments) >= max_total_comments:
            break
            
        query = query_template.format(domain=domain)
        logger.debug(f"[youtube] Searching: {query}")
        
        # Search for videos
        videos = _search_videos(query, max_results=_MAX_VIDEOS_PER_QUERY)
        if not videos:
            logger.debug(f"[youtube] No videos found for query: {query}")
            continue
        
        logger.info(f"[youtube] Found {len(videos)} videos for query: {query}")
        
        # Fetch comments from each video
        for video_item in videos:
            if len(all_comments) >= max_total_comments:
                break
            
            video_id = video_item["id"]["videoId"]
            video_title = video_item["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            logger.debug(f"[youtube] Fetching comments from: {video_title}")
            
            # Get comments
            comment_threads = _get_video_comments(video_id, max_results=_MAX_COMMENTS_PER_VIDEO)
            if not comment_threads:
                logger.debug(f"[youtube] No comments found for video: {video_id}")
                continue
            
            # Process comments
            for thread in comment_threads:
                if len(all_comments) >= max_total_comments:
                    break
                
                try:
                    comment_data = thread["snippet"]["topLevelComment"]["snippet"]
                    comment_id = thread["snippet"]["topLevelComment"]["id"]
                    comment_text = comment_data["textDisplay"]
                    
                    # Skip duplicates
                    if comment_id in seen_comment_ids:
                        continue
                    seen_comment_ids.add(comment_id)
                    
                    # Filter for substantial comments with complaint signals
                    if not _is_substantial(comment_text):
                        continue
                    if not _has_complaint_signal(comment_text):
                        continue
                    
                    # Create ScrapedComment
                    comment_url = f"{video_url}&lc={comment_id}"
                    scraped = ScrapedComment(
                        text=comment_text[:800],  # Truncate to keep token count sane
                        url=comment_url,
                        subreddit="youtube",  # Reuse field for source identification
                        post_title=video_title[:120],
                    )
                    all_comments.append(scraped)
                    
                except (KeyError, TypeError) as e:
                    logger.debug(f"[youtube] Failed to parse comment: {e}")
                    continue
    
    logger.info(
        f"[youtube] Scraped {len(all_comments)} comments with complaint signals "
        f"for domain '{domain}'"
    )
    
    return all_comments


# ------------------------------------------------------------------
# Validation helper (reused from reddit_scraper)
# ------------------------------------------------------------------
def validate_quote(quote: str, comments: list[ScrapedComment]) -> ScrapedComment | None:
    """Find the comment containing the given quote (case-insensitive substring match).
    
    Returns the matching ScrapedComment or None if not found.
    """
    quote_lower = quote.lower().strip()
    if not quote_lower:
        return None
    
    for comment in comments:
        if quote_lower in comment.text.lower():
            return comment
    
    return None
