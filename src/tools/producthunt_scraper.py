"""Product Hunt scraper — uses Product Hunt API v2.

Scrapes product comments and discussions from Product Hunt to find
user pain points and feedback across various industries.

API docs: https://api.producthunt.com/v2/docs
Requires: PRODUCT_HUNT_API_KEY in environment (OAuth bearer token)

To get an API key:
1. Go to https://api.producthunt.com/v1/oauth/authorize
2. Create an application and get your OAuth token
3. Set PRODUCT_HUNT_API_KEY to your bearer token (starts with "phc_...")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
_BASE_URL = "https://api.producthunt.com/v2"
_API_VERSION = "2024-09-04"
_REQUEST_DELAY_S: float = 0.5
_REQUEST_TIMEOUT: int = 20
_MAX_RESULTS_PER_QUERY: int = 20
_MIN_COMMENT_LENGTH: int = 40

# Search queries targeting frustration/pain points
_SEARCH_QUERIES: list[str] = [
    "{domain}",
    "{domain} tools",
    "{domain} software",
    "{domain} app",
    "{domain} platform",
]

# Domain-specific keyword expansions (similar to HN)
_DOMAIN_EXPANSIONS: dict[str, list[str]] = {
    "developer tools": ["devtools", "IDE", "CI/CD", "debugging", "testing framework", "API"],
    "healthcare": ["EHR", "patient portal", "medical software", "telehealth", "clinic"],
    "finance": ["fintech", "banking app", "payment processing", "accounting software", "investment"],
    "education": ["edtech", "LMS", "online learning", "course platform", "teaching"],
    "e-commerce": ["shopify", "online store", "checkout", "inventory management", "selling"],
    "marketing": ["SEO tool", "analytics", "email marketing", "social media management", "advertising"],
    "ai": ["LLM", "machine learning", "AI tool", "model training", "chatbot"],
    "productivity": ["project management", "task manager", "note-taking", "workflow", "calendar"],
    "design": ["UI design", "UX tool", "design system", "prototyping", "Figma"],
    "sales": ["CRM", "sales tool", "lead generation", "outreach", "sales automation"],
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


def _make_request(url: str, method: str = "GET", params: dict | None = None, json_data: dict | None = None) -> dict | None:
    """Make authenticated request to Product Hunt API with caching."""
    # Re-check settings at runtime to support testing with mocked settings
    from src.config import get_settings

    current_settings = get_settings()
    if not current_settings.product_hunt_api_key:
        logger.warning("[producthunt] PRODUCT_HUNT_API_KEY not set")
        return None

    cache_key = ("ph_api", method, url, str(sorted(params.items())) if params else "", str(json_data) if json_data else "")
    cached = _CACHE.get(cache_key, default=_MISSING)
    if cached is not _MISSING:
        return cached

    try:
        time.sleep(_REQUEST_DELAY_S)
        headers = {
            "Authorization": f"Bearer {settings.product_hunt_api_key}",
            "Producthunt-Version": _API_VERSION,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ventureforge/0.1.0 (academic research)",
        }

        if method == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=_REQUEST_TIMEOUT)
        else:  # POST
            r = requests.post(url, headers=headers, json=json_data, timeout=_REQUEST_TIMEOUT)

        r.raise_for_status()
        data = r.json()
        _CACHE.set(cache_key, data, expire=_TTL_S)
        return data
    except requests.HTTPError as e:
        logger.warning(f"[producthunt] HTTP error {r.status_code} for {url}: {e}")
        if r.status_code == 401:
            logger.error("[producthunt] Invalid API key - check PRODUCT_HUNT_API_KEY")
        return None
    except Exception as e:
        logger.warning(f"[producthunt] request error for {url}: {e}")
        return None


def _search_posts(query: str, num_results: int = _MAX_RESULTS_PER_QUERY) -> list[dict]:
    """Search Product Hunt posts via GraphQL API."""
    # Product Hunt uses GraphQL, so we need to construct a query
    # Note: The API may not support search with query parameter, so we'll get recent posts
    graphql_query = """
    query($first: Int) {
      posts(first: $first) {
        edges {
          node {
            id
            name
            tagline
            description
            url
            commentsCount
            createdAt
          }
        }
      }
    }
    """

    json_data = {
        "query": graphql_query,
        "variables": {"first": num_results},
    }

    data = _make_request(f"{_BASE_URL}/api/graphql", method="POST", json_data=json_data)
    if not data:
        return []

    posts = []
    for edge in data.get("data", {}).get("posts", {}).get("edges", []):
        node = edge.get("node", {})
        if node:
            posts.append(node)

    return posts


def _get_post_comments(post_id: str, limit: int = 20) -> list[dict]:
    """Fetch comments for a specific post."""
    graphql_query = """
    query($post_id: ID!, $first: Int) {
      post(id: $post_id) {
        comments(first: $first) {
          edges {
            node {
              id
              body
              createdAt
              user {
                name
                username
              }
            }
          }
        }
      }
    }
    """

    json_data = {
        "query": graphql_query,
        "variables": {"post_id": post_id, "first": limit},
    }

    data = _make_request(f"{_BASE_URL}/api/graphql", method="POST", json_data=json_data)
    if not data:
        return []

    comments = []
    for edge in data.get("data", {}).get("post", {}).get("comments", {}).get("edges", []):
        node = edge.get("node", {})
        if node:
            comments.append(node)

    return comments


def _comment_to_scraped(comment: dict, post_name: str, post_url: str) -> ScrapedComment | None:
    """Convert a Product Hunt comment to ScrapedComment."""
    body = comment.get("body", "").strip()

    if not body:
        return None

    # Clean up the comment text
    import re

    # Remove HTML tags (Product Hunt API returns HTML in comment body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = re.sub(r"&[a-z]+;", " ", body)
    # Remove excessive whitespace
    body = re.sub(r"\s+", " ", body).strip()

    if len(body) < _MIN_COMMENT_LENGTH:
        return None

    comment_id = comment.get("id", "")
    url = f"{post_url}?comment={comment_id}"

    return ScrapedComment(
        text=body,
        url=url,
        subreddit="producthunt",
        post_title=post_name,
    )


def scrape_for_domain(domain: str, max_total_comments: int = 100) -> list[ScrapedComment]:
    """Main entry point: scrape Product Hunt comments related to a domain's pain points.

    Returns a list of ScrapedComment objects compatible with the
    pain_point_miner pipeline.
    """
    # Re-check settings at runtime to support testing with mocked settings
    from src.config import get_settings

    current_settings = get_settings()
    if not current_settings.product_hunt_api_key:
        logger.info("[producthunt] skipped — PRODUCT_HUNT_API_KEY not set")
        return []

    keywords = _get_domain_keywords(domain)
    all_comments: list[ScrapedComment] = []
    seen_ids: set[str] = set()

    # Get recent posts (Product Hunt API doesn't support search by query)
    # We'll filter posts by domain keywords in name/tagline
    posts = _search_posts("", num_results=50)

    # Filter posts by domain keywords
    matching_posts = []
    for post in posts:
        post_name = post.get("name", "").lower()
        post_tagline = post.get("tagline", "").lower()
        post_description = post.get("description", "").lower()

        # Check if any keyword matches
        matches_domain = any(
            keyword.lower() in post_name or keyword.lower() in post_tagline or keyword.lower() in post_description
            for keyword in keywords
        )

        if matches_domain:
            matching_posts.append(post)

    logger.info(f"[producthunt] found {len(matching_posts)} matching posts out of {len(posts)} total")

    for post in matching_posts:
        if len(all_comments) >= max_total_comments:
            break

        post_id = post.get("id", "")
        post_name = post.get("name", "")
        post_url = post.get("url", "")

        if post_id in seen_ids:
            continue
        seen_ids.add(post_id)

        # Get comments for this post
        comments = _get_post_comments(post_id, limit=10)
        for comment in comments:
            comment_id = comment.get("id", "")
            if comment_id in seen_ids:
                continue
            seen_ids.add(comment_id)

            scraped = _comment_to_scraped(comment, post_name, post_url)
            if scraped:
                all_comments.append(scraped)

            if len(all_comments) >= max_total_comments:
                break

    logger.info(f"[producthunt] scraped {len(all_comments)} comments for domain='{domain}'")
    return all_comments