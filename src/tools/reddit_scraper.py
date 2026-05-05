"""Reddit JSON scraper — no API key, no PRAW.

Uses Reddit's public `.json` endpoints with search-first strategy:
  1. Search a subreddit for posts matching complaint keywords (`self:yes` + `t=month`)
  2. Filter posts by complaint keywords in title
  3. Fetch top-level comments from the matching posts
  4. Return structured comment data with verbatim text + direct URLs

Rate-limiting: 0.5 s sleep between every HTTP request to Reddit.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Iterator

import diskcache
import requests

from src.config import settings

logger = logging.getLogger(__name__)

# Disk-backed cache for Reddit JSON responses
_CACHE = diskcache.Cache(settings.cache_dir)
_TTL_S: int = settings.cache_ttl_hours * 3600
_MISSING = object()

# ------------------------------------------------------------------
# Search parameters (locked by design)
# ------------------------------------------------------------------
_SEARCH_QUERIES: list[str] = [
    "frustrated with self:yes",
    "I wish there was self:yes",
    "biggest problem self:yes",
]
_TITLE_KEYWORDS: list[str] = [
    "frustrated", "hate", "annoying", "problem", "issue",
    "struggle", "pain", "wish", "difficult", "terrible",
    "awful", "bad", "sucks", "complaint", "worst",
]
_MAX_POSTS_PER_SUBREDDIT: int = 5
_MAX_COMMENTS_PER_POST: int = 10
_MAX_COMMENTS_PER_SUBREDDIT: int = 50
_REQUEST_DELAY_S: float = 0.5
_SEARCH_LIMIT: int = 25
_TIME_WINDOW: str = "month"

# ------------------------------------------------------------------
# Community map — ordered most-specific → most-general
# ------------------------------------------------------------------
COMMUNITY_MAP: dict[str, list[str]] = {
    "developer_tools":     ["devops", "sysadmin", "webdev", "programming", "SaaS", "startups"],
    "healthcare":          ["physicaltherapy", "healthIT", "nursing", "medicine", "dentistry"],
    "finance_fintech":     ["financialindependence", "personalfinance", "smallbusiness", "Entrepreneur"],
    "education":           ["Homeschooling", "highereducation", "edtech", "Teachers", "studying"],
    "food_service":        ["restaurantowners", "KitchenConfidential", "smallbusiness", "Entrepreneur"],
    "e_commerce":          ["shopify", "ecommerce", "marketing", "smallbusiness", "Entrepreneur"],
    "marketing_social":    ["SEO", "marketing", "socialmedia", "advertising", "startups"],
    "real_estate":         ["Landlord", "realestateinvesting", "RealEstate", "smallbusiness"],
    "transportation":      ["trucking", "UberDrivers", "cars", "dashcam", "smallbusiness"],
    "ai_ml":               ["LocalLLaMA", "artificial", "MachineLearning", "SaaS", "startups"],
    "productivity":        ["Notion", "ObsidianMD", "productivity", "Entrepreneur", "smallbusiness"],
    "fashion_retail":      ["fashion", "malefashionadvice", "femalefashionadvice", "smallbusiness"],
    "sports_fitness":      ["CrossFit", "yoga", "running", "loseit", "fitness"],
    "agriculture":         ["farming", "homestead", "Agriculture", "smallbusiness"],
    "content_creator":     ["SmallYTChannel", "NewTubers", "podcasting", "VideoEditing", "YouTubers", "ContentCreation"],
    "general_other":       ["Entrepreneur", "smallbusiness", "startups", "SaaS"],
}

# Build a keyword → category reverse index for free-text matching
_KEYWORD_TO_CATEGORY: dict[str, str] = {}
for _cat, _subs in COMMUNITY_MAP.items():
    _KEYWORD_TO_CATEGORY[_cat.lower()] = _cat
    for _kw in _cat.lower().split("_"):
        _KEYWORD_TO_CATEGORY[_kw] = _cat

REDIRECT_MAP: dict[str, str] = {
    "sysadmin": "sysadmin",
    "healthit": "healthIT",
    "saas": "SaaS",
    "seo": "SEO",
    "crossfit": "CrossFit",
}


def resolve_domain(domain: str) -> tuple[str, list[str]]:
    """Map a free-text domain to a category key and its ordered subreddit list."""
    domain_lower = domain.lower().strip()

    # Exact match
    if domain_lower in COMMUNITY_MAP:
        return domain_lower, COMMUNITY_MAP[domain_lower]

    # Keyword/token matching
    tokens = re.findall(r"[a-z]+", domain_lower)
    scores: dict[str, int] = {}
    for tok in tokens:
        cat = _KEYWORD_TO_CATEGORY.get(tok)
        if cat:
            scores[cat] = scores.get(cat, 0) + 1

    if scores:
        best = max(scores, key=lambda k: (scores[k], k))
        return best, COMMUNITY_MAP[best]

    return "general_other", COMMUNITY_MAP["general_other"]


# ------------------------------------------------------------------
# Scraped-comment data structure
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ScrapedComment:
    text: str
    url: str
    subreddit: str
    post_title: str


# ------------------------------------------------------------------
# Low-level HTTP helpers
# ------------------------------------------------------------------
def _make_request(url: str, retries: int = 2) -> dict | list | None:
    """GET a Reddit JSON endpoint with automatic delay + retry, cached via diskcache.

    We key purely on the URL since all requests are GETs to Reddit's
    `.json` endpoints. Successful JSON responses are cached for
    ``settings.cache_ttl_hours`` (default 24h) to avoid repeatedly
    hitting Reddit during development.
    """
    cache_key = ("reddit_json", url)
    cached = _CACHE.get(cache_key, default=_MISSING)
    if cached is not _MISSING:
        return cached

    for attempt in range(retries + 1):
        try:
            time.sleep(_REQUEST_DELAY_S)
            headers = {
                "User-Agent": "ventureforge/0.1.0 (academic research)",
                "Accept": "application/json",
            }
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 404:
                # Cache 404s as None to avoid repeated misses
                _CACHE.set(cache_key, None, expire=_TTL_S)
                return None
            r.raise_for_status()
            data = r.json()
            _CACHE.set(cache_key, data, expire=_TTL_S)
            return data
        except requests.HTTPError as e:
            logger.warning(f"Reddit HTTP error {r.status_code} for {url}: {e}")
        except Exception as e:
            logger.warning(f"Reddit request error for {url}: {e}")
        if attempt < retries:
            time.sleep(_REQUEST_DELAY_S * 2)
    return None


def _post_title_matches(post_data: dict) -> bool:
    """Check if a post title contains complaint keywords."""
    title = post_data.get("title", "").lower()
    return any(kw in title for kw in _TITLE_KEYWORDS)


def _extract_top_level_comments(comments_listing: dict, post_title: str, subreddit: str) -> list[ScrapedComment]:
    """Walk top-level (depth-0) comments.  Skip AutoModerator / deleted."""
    out: list[ScrapedComment] = []
    for child in comments_listing.get("data", {}).get("children", []):
        if child.get("kind") != "t1":
            continue
        d = child["data"]
        body = d.get("body", "").strip()
        permalink = d.get("permalink", "").strip()
        author = d.get("author", "").strip()

        if not body or len(body) < 30:
            continue
        if author.lower() in {"automoderator", "[deleted]", "deleted"}:
            continue
        if not permalink:
            continue

        url = f"https://www.reddit.com{permalink}"
        out.append(ScrapedComment(text=body, url=url, subreddit=subreddit, post_title=post_title))
        if len(out) >= _MAX_COMMENTS_PER_POST:
            break
    return out


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def search_posts(subreddit: str, query: str) -> list[dict]:
    """Run a single search query on a subreddit; return raw post children."""
    q = requests.utils.quote(query)
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={q}&sort=new&t={_TIME_WINDOW}&limit={_SEARCH_LIMIT}&restrict_sr=on"
    )
    data = _make_request(url)
    if not data or "data" not in data:
        return []
    return data["data"].get("children", [])


def fetch_post_comments(subreddit: str, post_id: str) -> tuple[str, list[ScrapedComment]] | None:
    """Fetch a post and its top-level comments.

    Returns ``(post_title, comments)`` or ``None`` on failure.
    """
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    resp = _make_request(url)
    if not resp or not isinstance(resp, list) or len(resp) < 2:
        return None

    post_listing = resp[0]
    comments_listing = resp[1]

    children = post_listing.get("data", {}).get("children", [])
    if not children:
        return None
    post_title = children[0].get("data", {}).get("title", "")

    comments = _extract_top_level_comments(comments_listing, post_title, subreddit)
    return post_title, comments


def scrape_subreddit(subreddit: str, cap: int = _MAX_COMMENTS_PER_SUBREDDIT) -> list[ScrapedComment]:
    """Scrape a single subreddit, returning verbatim top-level comments."""
    all_comments: list[ScrapedComment] = []
    seen: set[str] = set()

    for query in _SEARCH_QUERIES:
        posts = search_posts(subreddit, query)
        for child in posts:
            if child.get("kind") != "t3":
                continue
            pid = child["data"].get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            if not _post_title_matches(child["data"]):
                continue

            result = fetch_post_comments(subreddit, pid)
            if not result:
                continue
            _, comments = result
            all_comments.extend(comments)

            if len(all_comments) >= cap or len(seen) >= _MAX_POSTS_PER_SUBREDDIT:
                return all_comments[:cap]

    return all_comments[:cap]


def scrape_for_domain(domain: str, max_total_comments: int = 200) -> list[ScrapedComment]:
    """Main entry point.

    1. Resolve domain → category → ordered subreddit list.
    2. Scrape subreddits sequentially (niche → general).
    3. Stop when ``max_total_comments`` collected.
    """
    category, subreddits = resolve_domain(domain)
    logger.info(
        f"[reddit] domain='{domain}' → category='{category}' → "
        f"subreddits={[f'r/{s}' for s in subreddits]}"
    )

    all_comments: list[ScrapedComment] = []
    for sr in subreddits:
        if len(all_comments) >= max_total_comments:
            break
        batch = scrape_subreddit(sr, cap=max_total_comments - len(all_comments))
        all_comments.extend(batch)
        logger.info(f"[reddit] r/{sr}: scraped {len(batch)} comments (running total {len(all_comments)})")

    logger.info(f"[reddit] finished with {len(all_comments)} comments")
    return all_comments


# ------------------------------------------------------------------
# Helpers consumed by other modules
# ------------------------------------------------------------------
def validate_quote(raw_quote: str, comments: list[ScrapedComment]) -> tuple[bool, str]:
    """Return ``(found, url_of_matching_comment)``.

    Performs an exact (case-sensitive) substring match first, then a
    relaxed match after stripping common Reddit markdown characters.
    """
    stripped = raw_quote.strip()
    if not stripped:
        return False, ""

    for c in comments:
        if stripped in c.text:
            return True, c.url

    # Relaxed: strip asterisks, angle brackets, collapse whitespace
    def _clean(text: str) -> str:
        return " ".join(
            text.replace("*", "").replace(">", "").replace("#", "").split()
        )

    c_stripped = _clean(stripped)
    for c in comments:
        if c_stripped in _clean(c.text):
            return True, c.url

    return False, ""
