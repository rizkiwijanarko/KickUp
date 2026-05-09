"""Component-level test for the Hacker News scraper.

Tests the HN Algolia API integration with real network calls.
Also includes unit tests for helper functions.

Run:
    uv run test_hackernews_scraper_component.py

Tip: set DOMAIN env var to override the default domain.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

from src.tools import hackernews_scraper as hns


def _patch_delay_to_zero() -> None:
    """Set request delay to zero for faster tests."""
    if hasattr(hns, "_REQUEST_DELAY_S"):
        hns._REQUEST_DELAY_S = 0.0  # type: ignore[attr-defined]


def test_domain_keyword_expansion() -> None:
    """Test that domain keywords are expanded correctly."""
    # Test exact match
    keywords = hns._get_domain_keywords("developer tools")
    assert "developer tools" in keywords or any("developer tools" in k.lower() for k in keywords)
    assert "devtools" in keywords
    assert "IDE" in keywords

    # Test case insensitive
    keywords = hns._get_domain_keywords("Developer Tools")
    assert "developer tools" in keywords or any("developer tools" in k.lower() for k in keywords)
    assert "devtools" in keywords

    # Test no expansion for unknown domain
    keywords = hns._get_domain_keywords("random topic")
    assert "random topic" in keywords
    assert len(keywords) == 1

    # Test healthcare expansion
    keywords = hns._get_domain_keywords("healthcare")
    assert "EHR" in keywords
    assert "patient portal" in keywords

    print("  PASS")


def test_hit_to_comment_basic() -> None:
    """Test conversion of Algolia hit to ScrapedComment."""
    # Valid comment with text
    hit = {
        "objectID": "12345",
        "comment_text": "This is a test comment about Docker being frustrating.",
        "story_title": "Docker woes",
        "story_id": "67890",
    }
    comment = hns._hit_to_comment(hit)
    assert comment is not None
    assert comment.text == "This is a test comment about Docker being frustrating."
    assert comment.url == "https://news.ycombinator.com/item?id=12345"
    assert comment.subreddit == "hackernews"
    assert comment.post_title == "Docker woes"

    # Comment with HTML tags (longer to pass min length check)
    hit_html = {
        "objectID": "12346",
        "comment_text": "<p>This is a longer comment that has <b>HTML</b> tags in it to ensure it passes the minimum length requirement after stripping.</p>",
        "story_title": "HTML test",
    }
    comment_html = hns._hit_to_comment(hit_html)
    assert comment_html is not None
    assert "<p>" not in comment_html.text
    assert "<b>" not in comment_html.text
    assert "HTML" in comment_html.text

    # Comment too short
    hit_short = {
        "objectID": "12347",
        "comment_text": "Too short",
        "story_title": "Short",
    }
    comment_short = hns._hit_to_comment(hit_short)
    assert comment_short is None

    # Comment with HTML entities (longer to pass min length check)
    hit_entities = {
        "objectID": "12348",
        "comment_text": "This is a longer comment &amp; that has &lt;test&gt; entities to ensure it passes the minimum length requirement after stripping.",
        "story_title": "Entities",
    }
    comment_entities = hns._hit_to_comment(hit_entities)
    assert comment_entities is not None
    assert "&amp;" not in comment_entities.text
    assert "&lt;" not in comment_entities.text
    assert "&gt;" not in comment_entities.text

    print("  PASS")


def test_hit_to_comment_story_text_fallback() -> None:
    """Test that story_text is used when comment_text is missing."""
    hit = {
        "objectID": "12349",
        "story_text": "This is story text instead of comment text.",
        "story_title": "Story fallback",
    }
    comment = hns._hit_to_comment(hit)
    assert comment is not None
    assert comment.text == "This is story text instead of comment text."

    # No text at all
    hit_no_text = {
        "objectID": "12350",
        "story_title": "No text",
    }
    comment_no_text = hns._hit_to_comment(hit_no_text)
    assert comment_no_text is None

    print("  PASS")


def test_scrape_for_domain_real_api() -> None:
    """Test real HN API call (slow, requires network)."""
    domain = os.getenv("DOMAIN", "developer tools")
    _patch_delay_to_zero()

    print(f"  Testing with domain: {domain}")

    comments = hns.scrape_for_domain(domain, max_total_comments=20)

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
        assert c.subreddit == "hackernews"
        assert c.url.startswith("https://news.ycombinator.com/item?id=")
        assert len(c.text) >= hns._MIN_COMMENT_LENGTH

        # Check for duplicates
        urls = [c.url for c in comments]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    print(f"  PASS (scraped {len(comments)} comments)")


def test_scrape_limits_respected() -> None:
    """Test that max_total_comments limit is respected."""
    _patch_delay_to_zero()
    domain = "docker"

    # Test with small limit
    comments = hns.scrape_for_domain(domain, max_total_comments=5)
    assert len(comments) <= 5

    # Test with zero limit
    comments_zero = hns.scrape_for_domain(domain, max_total_comments=0)
    assert len(comments_zero) == 0

    print("  PASS")


def test_comment_quality() -> None:
    """Test that scraped comments meet quality standards."""
    _patch_delay_to_zero()
    domain = os.getenv("DOMAIN", "developer tools")

    comments = hns.scrape_for_domain(domain, max_total_comments=30)

    if not comments:
        print("  SKIP (no comments scraped)")
        return

    # Check all comments meet minimum length
    for c in comments:
        assert len(c.text) >= hns._MIN_COMMENT_LENGTH, f"Comment too short: {c.text[:50]}"

    # Check URLs are valid
    for c in comments:
        assert "news.ycombinator.com/item?id=" in c.url

    # Check no HTML in text
    for c in comments:
        assert "<" not in c.text or ">" not in c.text, f"HTML found in: {c.text[:100]}"

    print(f"  PASS (validated {len(comments)} comments)")


def test_search_queries_structure() -> None:
    """Test that search queries are properly formatted."""
    assert len(hns._SEARCH_QUERIES) > 0
    assert all("{domain}" in q for q in hns._SEARCH_QUERIES)

    # Test query formatting
    formatted = [q.replace("{domain}", "docker") for q in hns._SEARCH_QUERIES]
    assert all("docker" in q for q in formatted)

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
    ]

    for domain in expected_domains:
        keywords = hns._get_domain_keywords(domain)
        assert len(keywords) > 1, f"No expansion for {domain}"
        assert domain.lower() in [k.lower() for k in keywords]

    print("  PASS")


def test_summarize_sample_comments() -> None:
    """Print summary statistics for a sample scrape."""
    _patch_delay_to_zero()
    domain = os.getenv("DOMAIN", "developer tools")

    comments = hns.scrape_for_domain(domain, max_total_comments=50)

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


_TESTS = [
    ("Domain keyword expansion", test_domain_keyword_expansion),
    ("Hit to comment basic conversion", test_hit_to_comment_basic),
    ("Hit to comment story text fallback", test_hit_to_comment_story_text_fallback),
    ("Search queries structure", test_search_queries_structure),
    ("Domain expansions comprehensive", test_domain_expansions_comprehensive),
    ("Scrape limits respected", test_scrape_limits_respected),
    ("Real HN API call (slow)", test_scrape_for_domain_real_api),
    ("Comment quality validation", test_comment_quality),
    ("Summarize sample comments", test_summarize_sample_comments),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Hacker News Scraper Component Tests")
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
