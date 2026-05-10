"""Component-level test for the Product Hunt scraper.

Tests the Product Hunt API integration with real network calls.
Also includes unit tests for helper functions.

Run:
    uv run test_producthunt_scraper_component.py

Tip: set DOMAIN env var to override the default domain.
Note: Requires PRODUCT_HUNT_API_KEY in environment for real API tests.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.tools import producthunt_scraper as phs


def _patch_delay_to_zero() -> None:
    """Set request delay to zero for faster tests."""
    if hasattr(phs, "_REQUEST_DELAY_S"):
        phs._REQUEST_DELAY_S = 0.0  # type: ignore[attr-defined]


def test_domain_keyword_expansion() -> None:
    """Test that domain keywords are expanded correctly."""
    # Test exact match
    keywords = phs._get_domain_keywords("developer tools")
    assert "developer tools" in keywords or any("developer tools" in k.lower() for k in keywords)
    assert "devtools" in keywords
    assert "IDE" in keywords

    # Test case insensitive
    keywords = phs._get_domain_keywords("Developer Tools")
    assert "developer tools" in keywords or any("developer tools" in k.lower() for k in keywords)
    assert "devtools" in keywords

    # Test no expansion for unknown domain
    keywords = phs._get_domain_keywords("random topic")
    assert "random topic" in keywords
    assert len(keywords) == 1

    # Test healthcare expansion
    keywords = phs._get_domain_keywords("healthcare")
    assert "EHR" in keywords
    assert "patient portal" in keywords

    print("  PASS")


def test_comment_to_scraped_basic() -> None:
    """Test conversion of Product Hunt comment to ScrapedComment."""
    # Valid comment
    comment = {
        "id": "12345",
        "body": "This is a test comment about a productivity tool being frustrating to use.",
        "createdAt": "2024-01-01T00:00:00Z",
        "user": {"name": "Test User", "username": "testuser"},
    }
    scraped = phs._comment_to_scraped(comment, "Test Product", "https://producthunt.com/products/test")
    assert scraped is not None
    assert scraped.text == "This is a test comment about a productivity tool being frustrating to use."
    assert scraped.url == "https://producthunt.com/products/test?comment=12345"
    assert scraped.subreddit == "producthunt"
    assert scraped.post_title == "Test Product"

    # Comment too short
    short_comment = {
        "id": "12346",
        "body": "Too short",
        "createdAt": "2024-01-01T00:00:00Z",
        "user": {"name": "Test User", "username": "testuser"},
    }
    scraped_short = phs._comment_to_scraped(short_comment, "Test Product", "https://producthunt.com/products/test")
    assert scraped_short is None

    # Empty comment
    empty_comment = {
        "id": "12347",
        "body": "",
        "createdAt": "2024-01-01T00:00:00Z",
        "user": {"name": "Test User", "username": "testuser"},
    }
    scraped_empty = phs._comment_to_scraped(empty_comment, "Test Product", "https://producthunt.com/products/test")
    assert scraped_empty is None

    print("  PASS")


def test_comment_text_cleaning() -> None:
    """Test that comment text is cleaned properly."""
    # Comment with excessive whitespace
    comment = {
        "id": "12348",
        "body": "This   has    excessive     whitespace    that should be cleaned.",
        "createdAt": "2024-01-01T00:00:00Z",
        "user": {"name": "Test User", "username": "testuser"},
    }
    scraped = phs._comment_to_scraped(comment, "Test Product", "https://producthunt.com/products/test")
    assert scraped is not None
    assert "   " not in scraped.text
    assert scraped.text == "This has excessive whitespace that should be cleaned."

    print("  PASS")


def test_search_queries_structure() -> None:
    """Test that search queries are properly formatted."""
    assert len(phs._SEARCH_QUERIES) > 0
    assert all("{domain}" in q for q in phs._SEARCH_QUERIES)

    # Test query formatting
    formatted = [q.replace("{domain}", "productivity") for q in phs._SEARCH_QUERIES]
    assert all("productivity" in q for q in formatted)

    print("  PASS")


def test_domain_expansions_comprehensive() -> None:
    """Test that domain expansions cover expected domains."""
    expected_domains = [
        "developer tools",
        "healthcare",
        "finance",
        "education",
        "e-commerce",
        "marketing",
        "ai",
        "productivity",
        "design",
        "sales",
    ]

    for domain in expected_domains:
        keywords = phs._get_domain_keywords(domain)
        assert len(keywords) > 1, f"No expansion for {domain}"
        assert domain.lower() in [k.lower() for k in keywords]

    print("  PASS")


def test_no_api_key_returns_empty() -> None:
    """Test that scraper returns empty list when no API key is set."""
    # This test doesn't require a real API key
    from unittest.mock import patch

    with patch("src.config.settings.product_hunt_api_key", None):
        comments = phs.scrape_for_domain("productivity", max_total_comments=10)
        assert comments == []

    print("  PASS")


def test_scrape_for_domain_real_api() -> None:
    """Test real Product Hunt API call (slow, requires network + API key)."""
    from src.config import settings

    if not settings.product_hunt_enabled:
        print("  SKIP (PRODUCT_HUNT_API_KEY not set)")
        return

    domain = os.getenv("DOMAIN", "productivity")
    _patch_delay_to_zero()

    print(f"  Testing with domain: {domain}")

    try:
        comments = phs.scrape_for_domain(domain, max_total_comments=20)

        # Basic sanity checks
        assert isinstance(comments, list)
        assert len(comments) <= 20

        if comments:
            # Check structure of first comment
            c = comments[0]
            assert hasattr(c, "text")
            assert hasattr(c, "url")
            assert hasattr(c, "subreddit")
            assert hasattr(c, "post_title")
            assert c.subreddit == "producthunt"
            assert c.url.startswith("https://")
            assert len(c.text) >= phs._MIN_COMMENT_LENGTH

            # Check for duplicates
            urls = [c.url for c in comments]
            assert len(urls) == len(set(urls)), "Duplicate URLs found"

        print(f"  PASS (scraped {len(comments)} comments)")
    except Exception as e:
        # If we get a 401, the API key is invalid - skip the test
        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"  SKIP (Invalid API key - {e})")
            return
        raise


def test_scrape_limits_respected() -> None:
    """Test that max_total_comments limit is respected."""
    from src.config import settings

    if not settings.product_hunt_enabled:
        print("  SKIP (PRODUCT_HUNT_API_KEY not set)")
        return

    _patch_delay_to_zero()
    domain = "productivity"

    try:
        # Test with small limit
        comments = phs.scrape_for_domain(domain, max_total_comments=5)
        assert len(comments) <= 5

        # Test with zero limit
        comments_zero = phs.scrape_for_domain(domain, max_total_comments=0)
        assert len(comments_zero) == 0

        print("  PASS")
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"  SKIP (Invalid API key - {e})")
            return
        raise


def test_comment_quality() -> None:
    """Test that scraped comments meet quality standards."""
    from src.config import settings

    if not settings.product_hunt_enabled:
        print("  SKIP (PRODUCT_HUNT_API_KEY not set)")
        return

    _patch_delay_to_zero()
    domain = os.getenv("DOMAIN", "productivity")

    try:
        comments = phs.scrape_for_domain(domain, max_total_comments=30)

        if not comments:
            print("  SKIP (no comments scraped)")
            return

        # Check all comments meet minimum length
        for c in comments:
            assert len(c.text) >= phs._MIN_COMMENT_LENGTH, f"Comment too short: {c.text[:50]}"

        # Check URLs are valid
        for c in comments:
            assert c.url.startswith("https://")

        # Check subreddit is always producthunt
        for c in comments:
            assert c.subreddit == "producthunt"

        print(f"  PASS (validated {len(comments)} comments)")
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"  SKIP (Invalid API key - {e})")
            return
        raise


def test_summarize_sample_comments() -> None:
    """Print summary statistics for a sample scrape."""
    from src.config import settings

    if not settings.product_hunt_enabled:
        print("  SKIP (PRODUCT_HUNT_API_KEY not set)")
        return

    _patch_delay_to_zero()
    domain = os.getenv("DOMAIN", "productivity")

    try:
        comments = phs.scrape_for_domain(domain, max_total_comments=50)

        if not comments:
            print("  SKIP (no comments scraped)")
            return

        lengths = [len(c.text) for c in comments]
        avg_len = sum(lengths) / len(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        print(f"  Sample statistics:")
        print(f"    Total comments: {len(comments)}")
        print(f"    Avg length: {avg_len:.1f} chars")
        print(f"    Min length: {min_len} chars")
        print(f"    Max length: {max_len} chars")
        print(f"    Sample URL: {comments[0].url}")
        print(f"    Sample text: {comments[0].text[:100]}...")

        print("  PASS")
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            print(f"  SKIP (Invalid API key - {e})")
            return
        raise


_TESTS = [
    ("Domain keyword expansion", test_domain_keyword_expansion),
    ("Comment to scraped basic conversion", test_comment_to_scraped_basic),
    ("Comment text cleaning", test_comment_text_cleaning),
    ("Search queries structure", test_search_queries_structure),
    ("Domain expansions comprehensive", test_domain_expansions_comprehensive),
    ("No API key returns empty", test_no_api_key_returns_empty),
    ("Real Product Hunt API call (slow)", test_scrape_for_domain_real_api),
    ("Scrape limits respected", test_scrape_limits_respected),
    ("Comment quality validation", test_comment_quality),
    ("Summarize sample comments", test_summarize_sample_comments),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Product Hunt Scraper Component Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, fn in _TESTS:
        print(f"\n[{passed + failed + 1}] {name}...")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback

            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed:
        import sys

        sys.exit(1)