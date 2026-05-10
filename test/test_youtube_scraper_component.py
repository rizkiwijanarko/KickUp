"""Component-level test for the YouTube Comments scraper.

Tests the YouTube Data API v3 integration with real network calls.
Also includes unit tests for helper functions.

Run:
    uv run test/test_youtube_scraper_component.py

Tip: set DOMAIN env var to override the default domain.
Requires: YOUTUBE_API_KEY in .env
"""
from __future__ import annotations

import os

from src.tools import youtube_scraper as yts


def test_search_queries_structure() -> None:
    """Test that search queries are properly formatted."""
    assert len(yts._SEARCH_QUERIES) > 0
    assert all("{domain}" in q for q in yts._SEARCH_QUERIES)

    # Test query formatting
    formatted = [q.format(domain="meal prep") for q in yts._SEARCH_QUERIES]
    assert all("meal prep" in q for q in formatted)
    assert "meal prep frustrating" in formatted
    assert "meal prep problems" in formatted

    print("  PASS")


def test_complaint_keywords_exist() -> None:
    """Test that complaint keywords are defined."""
    assert len(yts._COMPLAINT_KEYWORDS) > 0
    assert "frustrated" in yts._COMPLAINT_KEYWORDS
    assert "problem" in yts._COMPLAINT_KEYWORDS
    assert "hate" in yts._COMPLAINT_KEYWORDS

    print("  PASS")


def test_has_complaint_signal() -> None:
    """Test complaint signal detection."""
    # Positive cases
    assert yts._has_complaint_signal("This is so frustrating!")
    assert yts._has_complaint_signal("I hate this problem")
    assert yts._has_complaint_signal("The worst issue ever")
    assert yts._has_complaint_signal("This is terrible and annoying")

    # Negative cases
    assert not yts._has_complaint_signal("This is great!")
    assert not yts._has_complaint_signal("I love this")
    assert not yts._has_complaint_signal("Works perfectly")

    # Case insensitive
    assert yts._has_complaint_signal("FRUSTRATED with this")
    assert yts._has_complaint_signal("Hate HATE hate")

    print("  PASS")


def test_is_substantial() -> None:
    """Test comment length filtering."""
    # Too short
    assert not yts._is_substantial("Too short")
    assert not yts._is_substantial("Yes")
    assert not yts._is_substantial("I agree")

    # Substantial
    assert yts._is_substantial("This is a longer comment with enough words and characters")
    assert yts._is_substantial("I really hate how this tool works, it's so frustrating")

    # Edge cases
    assert not yts._is_substantial("a b c")  # 3 words but < 20 chars
    assert not yts._is_substantial("12345678901234567890")  # 20 chars but 1 word

    print("  PASS")


def test_api_key_configured() -> None:
    """Test that YouTube API key is configured."""
    from src.config import settings

    if not settings.youtube_api_key:
        print("  SKIP (YOUTUBE_API_KEY not set in .env)")
        raise RuntimeError("YOUTUBE_API_KEY not configured - add it to .env to run this test")

    assert len(settings.youtube_api_key) > 10
    print("  PASS")


def test_scrape_for_domain_real_api() -> None:
    """Test real YouTube API call (slow, requires network and API key)."""
    from src.config import settings

    if not settings.youtube_api_key:
        print("  SKIP (YOUTUBE_API_KEY not set)")
        return

    domain = os.getenv("DOMAIN", "meal prep")
    print(f"  Testing with domain: {domain}")

    comments = yts.scrape_for_domain(domain, max_total_comments=20)

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
        assert c.subreddit == "youtube"
        assert "youtube.com/watch?v=" in c.url
        assert len(c.text) >= 20  # Minimum substantial length

        # Check for duplicates
        urls = [c.url for c in comments]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

        # Check all comments have complaint signals
        for comment in comments:
            assert yts._has_complaint_signal(comment.text), f"No complaint signal in: {comment.text[:50]}"

    print(f"  PASS (scraped {len(comments)} comments)")


def test_scrape_limits_respected() -> None:
    """Test that max_total_comments limit is respected."""
    from src.config import settings

    if not settings.youtube_api_key:
        print("  SKIP (YOUTUBE_API_KEY not set)")
        return

    domain = "fitness"

    # Test with small limit
    comments = yts.scrape_for_domain(domain, max_total_comments=5)
    assert len(comments) <= 5

    # Test with zero limit
    comments_zero = yts.scrape_for_domain(domain, max_total_comments=0)
    assert len(comments_zero) == 0

    print("  PASS")


def test_comment_quality() -> None:
    """Test that scraped comments meet quality standards."""
    from src.config import settings

    if not settings.youtube_api_key:
        print("  SKIP (YOUTUBE_API_KEY not set)")
        return

    domain = os.getenv("DOMAIN", "developer tools")

    comments = yts.scrape_for_domain(domain, max_total_comments=30)

    if not comments:
        print("  SKIP (no comments scraped)")
        return

    # Check all comments meet minimum length
    for c in comments:
        assert len(c.text) >= 20, f"Comment too short: {c.text[:50]}"
        assert len(c.text.split()) >= 3, f"Comment has too few words: {c.text[:50]}"

    # Check URLs are valid
    for c in comments:
        assert "youtube.com/watch?v=" in c.url
        assert "&lc=" in c.url  # Comment ID parameter

    # Check all have complaint signals
    for c in comments:
        assert yts._has_complaint_signal(c.text), f"No complaint signal: {c.text[:50]}"

    # Check post_title is populated
    for c in comments:
        assert len(c.post_title) > 0, "Empty post_title"

    print(f"  PASS (validated {len(comments)} comments)")


def test_validate_quote() -> None:
    """Test quote validation helper."""
    from src.tools.reddit_scraper import ScrapedComment

    comments = [
        ScrapedComment(
            text="This is frustrating and annoying",
            url="https://youtube.com/watch?v=123&lc=abc",
            subreddit="youtube",
            post_title="Test Video",
        ),
        ScrapedComment(
            text="I hate this problem so much",
            url="https://youtube.com/watch?v=456&lc=def",
            subreddit="youtube",
            post_title="Another Video",
        ),
    ]

    # Exact match
    result = yts.validate_quote("This is frustrating and annoying", comments)
    assert result is not None
    assert result.url == "https://youtube.com/watch?v=123&lc=abc"

    # Substring match
    result = yts.validate_quote("frustrating", comments)
    assert result is not None

    # Case insensitive
    result = yts.validate_quote("HATE THIS PROBLEM", comments)
    assert result is not None

    # No match
    result = yts.validate_quote("nonexistent quote", comments)
    assert result is None

    print("  PASS")


def test_summarize_sample_comments() -> None:
    """Print summary statistics for a sample scrape."""
    from src.config import settings

    if not settings.youtube_api_key:
        print("  SKIP (YOUTUBE_API_KEY not set)")
        return

    domain = os.getenv("DOMAIN", "meal prep")

    comments = yts.scrape_for_domain(domain, max_total_comments=50)

    if not comments:
        print("  SKIP (no comments scraped)")
        return

    lengths = [len(c.text) for c in comments]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    # Count unique videos
    video_ids = set()
    for c in comments:
        if "watch?v=" in c.url:
            video_id = c.url.split("watch?v=")[1].split("&")[0]
            video_ids.add(video_id)

    print(f"  Sample statistics:")
    print(f"    Total comments: {len(comments)}")
    print(f"    Unique videos: {len(video_ids)}")
    print(f"    Avg length: {avg_len:.1f} chars")
    print(f"    Min length: {min_len} chars")
    print(f"    Max length: {max_len} chars")
    print(f"    Sample URL: {comments[0].url}")
    print(f"    Sample video: {comments[0].post_title}")
    print(f"    Sample text: {comments[0].text[:100]}...")

    print("  PASS")


def test_multiple_domains() -> None:
    """Test scraping across different domain types."""
    from src.config import settings

    if not settings.youtube_api_key:
        print("  SKIP (YOUTUBE_API_KEY not set)")
        return

    test_domains = [
        "developer tools",  # Tech
        "meal prep",  # Consumer
        "fitness coaching",  # Health/wellness
    ]

    results = {}
    for domain in test_domains:
        comments = yts.scrape_for_domain(domain, max_total_comments=10)
        results[domain] = len(comments)
        print(f"    {domain}: {len(comments)} comments")

    # At least one domain should return results
    assert sum(results.values()) > 0, "No comments scraped for any domain"

    print("  PASS")


_TESTS = [
    ("Search queries structure", test_search_queries_structure),
    ("Complaint keywords exist", test_complaint_keywords_exist),
    ("Complaint signal detection", test_has_complaint_signal),
    ("Substantial comment filtering", test_is_substantial),
    ("Quote validation helper", test_validate_quote),
    ("API key configured", test_api_key_configured),
    ("Scrape limits respected", test_scrape_limits_respected),
    ("Real YouTube API call (slow)", test_scrape_for_domain_real_api),
    ("Comment quality validation", test_comment_quality),
    ("Multiple domain types", test_multiple_domains),
    ("Summarize sample comments", test_summarize_sample_comments),
]


if __name__ == "__main__":
    print("=" * 60)
    print("YouTube Comments Scraper Component Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, fn in _TESTS:
        print(f"\n[{passed + failed + skipped + 1}] {name}...")
        try:
            fn()
            passed += 1
        except RuntimeError as e:
            if "not configured" in str(e):
                print(f"  SKIP: {e}")
                skipped += 1
            else:
                import traceback

                print(f"  FAIL: {e}")
                traceback.print_exc()
                failed += 1
        except Exception as e:
            import traceback

            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed:
        import sys

        sys.exit(1)
