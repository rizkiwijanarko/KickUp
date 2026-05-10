"""Hacker News scraper — uses the free Algolia HN Search API.

No API key required. Searches for user complaints and frustrations
in HN comments and stories, returning structured comment data compatible
with the pain_point_miner pipeline.

API docs: https://hn.algolia.com/api
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

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
_BASE_URL = "https://hn.algolia.com/api/v1"
_REQUEST_DELAY_S: float = 0.3
_REQUEST_TIMEOUT: int = 15
_MAX_RESULTS_PER_QUERY: int = 50
_MIN_COMMENT_LENGTH: int = 40

# Filter for content from the last 1 years (365 days)
_TWO_YEARS_AGO = int(time.time()) - (365 * 24 * 60 * 60)

# Search queries targeting frustration/pain points
_SEARCH_QUERIES: list[str] = [
    '"{domain}" frustrated',
    '"{domain}" annoying problem',
    '"{domain}" I wish there was',
    '"{domain}" biggest pain',
    '"{domain}" hate dealing with',
    '"{domain}" waste time',
]

# Domain-specific keyword expansions
_DOMAIN_EXPANSIONS: dict[str, list[str]] = {
    "developer tools": ["devtools", "IDE", "CI/CD", "debugging", "testing framework"],
    "healthcare": ["EHR", "patient portal", "medical software", "telehealth"],
    "finance": ["fintech", "banking app", "payment processing", "accounting software"],
    "education": ["edtech", "LMS", "online learning", "course platform"],
    "e-commerce": ["shopify", "online store", "checkout", "inventory management"],
    "marketing": ["SEO tool", "analytics", "email marketing", "social media management"],
    "ai": ["LLM", "machine learning", "AI tool", "model training"],
    "productivity": ["project management", "task manager", "note-taking", "workflow"],
}


def _get_domain_keywords(domain: str) -> list[str]:
    """Get additional search keywords based on domain."""
    domain_lower = domain.lower()
    keywords = [domain]
    for key, expansions in _DOMAIN_EXPANSIONS.items():
        if key in domain_lower:
            keywords.extend(expansions)
            break
    return keywords


def _make_request(url: str, params: dict | None = None) -> dict | None:
    """GET request to HN Algolia API with caching."""
    cache_key = ("hn_api", url, str(sorted(params.items())) if params else "")
    cached = _CACHE.get(cache_key, default=_MISSING)
    if cached is not _MISSING:
        return cached

    try:
        time.sleep(_REQUEST_DELAY_S)
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "ventureforge/0.1.0 (academic research)"},
            timeout=_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        _CACHE.set(cache_key, data, expire=_TTL_S)
        return data
    except Exception as e:
        logger.warning(f"[hackernews] request error for {url}: {e}")
        return None


def _search_comments(query: str, num_results: int = _MAX_RESULTS_PER_QUERY) -> list[dict]:
    """Search HN comments via Algolia API."""
    params = {
        "query": query,
        "tags": "comment",
        "hitsPerPage": num_results,
        "numericFilters": f"created_at_i>{_TWO_YEARS_AGO}",
    }
    data = _make_request(f"{_BASE_URL}/search_by_date", params)
    if not data:
        return []
    return data.get("hits", [])


def _search_stories(query: str, num_results: int = 20) -> list[dict]:
    """Search HN stories (Ask HN, Show HN, etc.) via Algolia API."""
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": num_results,
        "numericFilters": f"created_at_i>{_TWO_YEARS_AGO}",
    }
    data = _make_request(f"{_BASE_URL}/search", params)
    if not data:
        return []
    return data.get("hits", [])


def _hit_to_comment(hit: dict) -> ScrapedComment | None:
    """Convert an Algolia hit to a ScrapedComment."""
    text = hit.get("comment_text", "")
    if not text:
        text = hit.get("story_text", "")
    if not text:
        return None

    # Strip HTML tags (HN API returns HTML in comment_text)
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = " ".join(text.split()).strip()

    if len(text) < _MIN_COMMENT_LENGTH:
        return None

    object_id = hit.get("objectID", "")
    story_id = hit.get("story_id", object_id)
    story_title = hit.get("story_title", "") or hit.get("title", "")
    url = f"https://news.ycombinator.com/item?id={object_id}"

    return ScrapedComment(
        text=text,
        url=url,
        subreddit=f"hackernews",  # reuse field name for compatibility
        post_title=story_title,
    )


def scrape_for_domain(domain: str, max_total_comments: int = 150) -> list[ScrapedComment]:
    """Main entry point: scrape HN comments related to a domain's pain points.

    Returns a list of ScrapedComment objects compatible with the
    pain_point_miner pipeline.
    """
    keywords = _get_domain_keywords(domain)
    all_comments: list[ScrapedComment] = []
    seen_ids: set[str] = set()

    for keyword in keywords:
        if len(all_comments) >= max_total_comments:
            break

        for query_template in _SEARCH_QUERIES:
            if len(all_comments) >= max_total_comments:
                break

            query = query_template.replace("{domain}", keyword)
            hits = _search_comments(query, num_results=30)

            for hit in hits:
                oid = hit.get("objectID", "")
                if oid in seen_ids:
                    continue
                seen_ids.add(oid)

                comment = _hit_to_comment(hit)
                if comment:
                    all_comments.append(comment)

                if len(all_comments) >= max_total_comments:
                    break

    # Also search "Ask HN" stories which often contain pain points
    ask_hn_queries = [
        f"Ask HN: {domain} frustrating",
        f"Ask HN: {domain} problem",
        f"Ask HN: what tools {domain}",
    ]
    for query in ask_hn_queries:
        if len(all_comments) >= max_total_comments:
            break
        stories = _search_stories(query, num_results=10)
        for story in stories:
            story_id = story.get("objectID", "")
            if story_id in seen_ids:
                continue
            seen_ids.add(story_id)

            # Get comments from this story
            story_comments = _get_story_comments(story_id, limit=10)
            for comment in story_comments:
                if comment.url.split("=")[-1] not in seen_ids:
                    all_comments.append(comment)
                    seen_ids.add(comment.url.split("=")[-1])

                if len(all_comments) >= max_total_comments:
                    break

    logger.info(f"[hackernews] scraped {len(all_comments)} comments for domain='{domain}'")
    return all_comments


def _get_story_comments(story_id: str, limit: int = 10) -> list[ScrapedComment]:
    """Fetch top comments from a specific HN story."""
    params = {
        "tags": f"comment,story_{story_id}",
        "hitsPerPage": limit,
    }
    data = _make_request(f"{_BASE_URL}/search", params)
    if not data:
        return []

    comments: list[ScrapedComment] = []
    for hit in data.get("hits", []):
        comment = _hit_to_comment(hit)
        if comment:
            comments.append(comment)
    return comments
