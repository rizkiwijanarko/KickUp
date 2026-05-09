"""Tavily web content scraper — searches the broader web for user complaints and opinions.

Unlike tavily_fallback.py (which only discovers subreddit names), this module
uses Tavily to actually extract user opinions and complaints from forums,
blogs, Q&A sites, and community discussions across the web.

Required env: ``TAVILY_API_KEY``
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import diskcache
import requests

from src.config import settings
from src.tools.reddit_scraper import ScrapedComment

logger = logging.getLogger(__name__)

# Disk-backed cache
_CACHE = diskcache.Cache(settings.cache_dir)
_TTL_S: int = settings.cache_ttl_hours * 3600
_MISSING = object()

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_MAX_RESULTS: int = 10
_REQUEST_TIMEOUT: int = 20
_REQUEST_DELAY_S: float = 0.5
_MIN_CONTENT_LENGTH: int = 50

# Search query templates targeting user frustrations
_SEARCH_TEMPLATES: list[str] = [
    '"{domain}" frustrated users forum',
    '"{domain}" biggest problem complaint',
    '"{domain}" "I wish" OR "I hate" OR "pain point"',
    '"{domain}" user feedback negative review',
    '"{domain}" community discussion problem',
]

# Domains to prioritize (forums, Q&A, communities)
_PRIORITY_DOMAINS: list[str] = [
    "stackoverflow.com",
    "news.ycombinator.com",
    "dev.to",
    "community.",
    "forum.",
    "discuss.",
    "github.com/issues",
    "producthunt.com",
    "indiehackers.com",
    "lobste.rs",
]

# Domains to exclude (not useful for pain points)
_EXCLUDED_DOMAINS: list[str] = [
    "youtube.com",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "pinterest.com",
    "amazon.com",
]


def _is_useful_source(url: str) -> bool:
    """Check if a URL is from a useful source for pain point extraction."""
    url_lower = url.lower()
    for excluded in _EXCLUDED_DOMAINS:
        if excluded in url_lower:
            return False
    return True


def _clean_content(text: str) -> str:
    """Clean extracted web content."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common boilerplate patterns
    text = re.sub(r"(Sign up|Log in|Subscribe|Cookie|Privacy Policy).*?(\.|$)", "", text, flags=re.IGNORECASE)
    return text.strip()


def _search_tavily(query: str, include_domains: list[str] | None = None) -> list[dict]:
    """Execute a single Tavily search query."""
    if not settings.tavily_enabled:
        return []

    cache_key = ("tavily_content", query, str(include_domains))
    cached = _CACHE.get(cache_key, default=_MISSING)
    if cached is not _MISSING:
        return cached

    payload: dict[str, Any] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": _MAX_RESULTS,
        "include_answer": False,
        "include_raw_content": True,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    try:
        time.sleep(_REQUEST_DELAY_S)
        r = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        _CACHE.set(cache_key, results, expire=_TTL_S)
        return results
    except requests.HTTPError as e:
        logger.warning(f"[tavily_content] HTTP error: {e}")
        return []
    except Exception as e:
        logger.warning(f"[tavily_content] request error: {e}")
        return []


def _result_to_comments(result: dict) -> list[ScrapedComment]:
    """Extract usable comment-like content from a Tavily search result.

    Splits long content into paragraph-sized chunks that can serve as
    individual 'comments' for the pain point extraction pipeline.
    """
    url = result.get("url", "")
    title = result.get("title", "")
    content = result.get("raw_content", "") or result.get("content", "")

    if not content or not _is_useful_source(url):
        return []

    content = _clean_content(content)
    if len(content) < _MIN_CONTENT_LENGTH:
        return []

    # Determine source label from URL
    source_label = "web"
    if "stackoverflow.com" in url:
        source_label = "stackoverflow"
    elif "github.com" in url:
        source_label = "github"
    elif "dev.to" in url:
        source_label = "devto"
    elif "indiehackers.com" in url:
        source_label = "indiehackers"
    elif "producthunt.com" in url:
        source_label = "producthunt"
    elif "lobste.rs" in url:
        source_label = "lobsters"

    # Split content into meaningful chunks (paragraphs or sentences)
    # Each chunk becomes a separate "comment" for the LLM to analyze
    chunks = _split_into_chunks(content, min_length=60, max_length=800)

    comments: list[ScrapedComment] = []
    for chunk in chunks:
        comments.append(
            ScrapedComment(
                text=chunk,
                url=url,
                subreddit=source_label,
                post_title=title,
            )
        )

    return comments


def _split_into_chunks(text: str, min_length: int = 60, max_length: int = 800) -> list[str]:
    """Split text into paragraph-sized chunks suitable for pain point extraction."""
    # First try splitting by double newlines (paragraphs)
    paragraphs = re.split(r"\n\s*\n|\. (?=[A-Z])", text)

    chunks: list[str] = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= max_length:
            current_chunk = f"{current_chunk} {para}".strip() if current_chunk else para
        else:
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk and len(current_chunk) >= min_length:
        chunks.append(current_chunk)

    # If no good chunks found, just use the whole text truncated
    if not chunks and len(text) >= min_length:
        chunks = [text[:max_length]]

    return chunks


def scrape_for_domain(domain: str, max_total_comments: int = 100) -> list[ScrapedComment]:
    """Main entry point: search the web for user complaints about a domain.

    Returns a list of ScrapedComment objects compatible with the
    pain_point_miner pipeline.
    """
    if not settings.tavily_enabled:
        logger.info("[tavily_content] skipped — TAVILY_API_KEY not set")
        return []

    all_comments: list[ScrapedComment] = []
    seen_urls: set[str] = set()

    for template in _SEARCH_TEMPLATES:
        if len(all_comments) >= max_total_comments:
            break

        query = template.replace("{domain}", domain)
        results = _search_tavily(query)

        for result in results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            comments = _result_to_comments(result)
            for comment in comments:
                all_comments.append(comment)
                if len(all_comments) >= max_total_comments:
                    break

    logger.info(f"[tavily_content] scraped {len(all_comments)} content chunks for domain='{domain}'")
    return all_comments
