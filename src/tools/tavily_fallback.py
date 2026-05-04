"""Tavily community-discovery fallback.

Uses Tavily search to find additional Reddit subreddits when the static
COMMUNITY_MAP yields fewer than ``threshold`` comments.  **This module never
extracts pain points from Tavily snippets** — it only returns subreddit names
that should be scraped by the Reddit JSON scraper.

Required env: ``TAVILY_API_KEY``
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from src.config import settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
_MAX_RESULTS: int = 5
_REQUEST_TIMEOUT: int = 20
_REQUEST_DELAY_S: float = 0.5

# Subreddits we never want to scrape (meta, huge, off-topic)
_SUBREDDIT_DENYLIST: set[str] = {
    "all", "popular", "askreddit", "announcements", "blog",
    "pics", "funny", "memes", "aww", "gifs", "videos", "news",
    "worldnews", "politics", "science", "IAmA", "bestof",
    "lifeprotips", "personalfinance", "amitheasshole", "tifu",
    "todayilearned",
}


def _url_to_subreddit(url: str) -> str | None:
    """Extract ``subreddit_name`` from a Reddit URL or return ``None``."""
    # Match /r/subreddit/... or /r/subreddit (trailing slash optional)
    m = re.search(r"reddit\.com/r/([A-Za-z0-9_]+)", url)
    if m:
        name = m.group(1).lower()
        return name
    return None


def _snippet_to_subreddits(text: str) -> set[str]:
    """Extract r/name references from raw text."""
    found: set[str] = set()
    for m in re.finditer(r"/?r/([A-Za-z0-9_]+)", text):
        name = m.group(1).lower()
        if name not in _SUBREDDIT_DENYLIST:
            found.add(name)
    return found


def _is_valid_subreddit(name: str) -> bool:
    """HEAD-check whether ``r/name`` actually exists and is accessible."""
    url = f"https://www.reddit.com/r/{name}.json"
    try:
        time.sleep(_REQUEST_DELAY_S)
        r = requests.head(
            url,
            headers={"User-Agent": "ventureforge/0.1.0 (academic research)"},
            timeout=10,
            allow_redirects=True,
        )
        return r.status_code == 200
    except Exception as e:
        logger.debug(f"HEAD check failed for r/{name}: {e}")
        return False


def search_communities(domain: str) -> list[str]:
    """Ask Tavily for Reddit communities related to *domain* complaints.

    Returns a deduplicated, validated list of subreddit names (lowercased)
    sorted by confidence.  Empty list if Tavily is mis-configured, rate-
    limited, or returns no Reddit results.
    """
    if not settings.tavily_enabled:
        logger.info("[tavily] fallback skipped — TAVILY_API_KEY not set")
        return []

    query = f'site:reddit.com "{domain}" frustration OR complaint OR problem OR hate community subreddit'
    payload: dict[str, Any] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": _MAX_RESULTS,
        "domain": "reddit.com",
    }

    try:
        time.sleep(_REQUEST_DELAY_S)
        r = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        logger.warning(f"[tavily] HTTP error {r.status_code}: {e}")
        return []
    except Exception as e:
        logger.warning(f"[tavily] request error: {e}")
        return []

    results = data.get("results", [])
    if not results:
        logger.info("[tavily] no results returned")
        return []

    # ---- Extract candidate subreddits from URLs and snippets ----
    candidates: set[str] = set()
    for item in results:
        url = item.get("url", "")
        snippet = item.get("content", "")
        if "reddit.com" in url:
            sr = _url_to_subreddit(url)
            if sr:
                candidates.add(sr)
        candidates.update(_snippet_to_subreddits(snippet))

    # Remove denylisted and already-known subreddits
    known = set()
    from src.tools.reddit_scraper import COMMUNITY_MAP
    for subs in COMMUNITY_MAP.values():
        known.update(s.lower() for s in subs)
    candidates -= known
    candidates -= _SUBREDDIT_DENYLIST

    if not candidates:
        logger.info("[tavily] no new subreddit candidates found")
        return []

    # Validate existence via HEAD request
    valid: list[str] = []
    for name in sorted(candidates):
        if _is_valid_subreddit(name):
            valid.append(name)
            logger.info(f"[tavily] validated r/{name}")
        else:
            logger.debug(f"[tavily] r/{name} rejected (HEAD check failed)")

    logger.info(f"[tavily] discovered {len(valid)} new subreddits: {valid}")
    return valid
