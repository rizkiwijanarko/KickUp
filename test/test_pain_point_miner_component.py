"""Component-level test for the Pain Point Miner agent.

Fast, deterministic, offline by mocking Reddit/Tavily + the LLM.
Run with:
    uv run test_pain_point_miner_component.py
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.pain_point_miner import run as run_pain_point_miner
from src.state.schema import DataSource, PainPoint, PipelineStage, VentureForgeState
from src.tools.reddit_scraper import ScrapedComment

logging.basicConfig(level=logging.INFO)


def _make_comments() -> list[ScrapedComment]:
    c1 = ScrapedComment(
        text="I spend more time debugging docker-compose.yml than writing actual code.",
        url="https://www.reddit.com/r/docker/comments/abc123/comment/xyz",
        subreddit="docker",
        post_title="Frustrated with docker compose",
    )
    c2 = ScrapedComment(
        text="Why does my test pass locally but fail in CI with the exact same Dockerfile?",
        url="https://www.reddit.com/r/devops/comments/def456/comment/qwe",
        subreddit="devops",
        post_title="CI debugging is awful",
    )
    return [c1, c2]


def _make_pp_item(*, quote: str, url: str) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "title": "Docker Compose debugging pain",
        "description": "Developers struggle to manage multi-service setups and chase config errors.",
        "rubric": {
            "is_genuine_current_frustration": True,
            "has_verbatim_quote": False,  # should be forced True by validation
            "user_segment_specific": True,
        },
        "passes_rubric": "yes",
        "source_url": url,
        "raw_quote": quote,
        "source": DataSource.REDDIT.value,
    }


def test_no_comments_returns_empty() -> None:
    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    with patch("src.agents.pain_point_miner._tavily_enriched_scrape", return_value=([], DataSource.HACKERNEWS)):
        result = run_pain_point_miner(state)
    assert result["pain_points"] == []
    assert result["current_stage"] == PipelineStage.MINING
    assert result["next_node"] == "orchestrator"
    print("  PASS")


def test_wellformed_response_validates_quotes_and_overwrites_url() -> None:
    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    comments = _make_comments()

    # LLM-proposed URL is wrong; validator should overwrite with matched comment URL.
    item = _make_pp_item(
        quote="I spend more time debugging docker-compose.yml than writing actual code.",
        url="https://reddit.com/wrong/url",
    )

    with patch("src.agents.pain_point_miner._tavily_enriched_scrape", return_value=(comments, DataSource.HACKERNEWS)), patch(
        "src.agents.pain_point_miner.get_llm"
    ) as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps([item])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_pain_point_miner(state)

    pps: list[PainPoint] = result["pain_points"]
    assert len(pps) == 1
    assert pps[0].raw_quote in comments[0].text
    assert pps[0].source_url == comments[0].url
    assert pps[0].rubric.has_verbatim_quote is True
    assert pps[0].passes_rubric is True
    print("  PASS")


def test_quote_not_found_is_rejected() -> None:
    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    comments = _make_comments()

    item = _make_pp_item(quote="This quote does not exist in any comment.", url="https://reddit.com/x")

    with patch("src.agents.pain_point_miner._tavily_enriched_scrape", return_value=(comments, DataSource.HACKERNEWS)), patch(
        "src.agents.pain_point_miner.get_llm"
    ) as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps({"pain_points": [item]})
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_pain_point_miner(state)

    pps: list[PainPoint] = result["pain_points"]
    assert len(pps) == 0
    print("  PASS")


def test_live_llm_produces_valid_pain_points() -> None:
    """Uses real API call + real scraping. Requires any LLM API key."""
    import os

    from src.config import settings

    # Check for any configured LLM API key
    if not (settings.llm_api_key or settings.fast_llm_api_key or os.getenv("OPENAI_API_KEY")):
        print("  SKIP (no LLM_API_KEY, FAST_LLM_API_KEY, or OPENAI_API_KEY)")
        return

    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    result = run_pain_point_miner(state)
    pps: list[PainPoint] = result["pain_points"]
    assert all(pp.passes_rubric for pp in pps)
    print("  PASS")


_TESTS = [
    ("No comments returns empty", test_no_comments_returns_empty),
    ("Well-formed response validates quotes + overwrites URL", test_wellformed_response_validates_quotes_and_overwrites_url),
    ("Quote not found rejected", test_quote_not_found_is_rejected),
    ("Live LLM (slow)", test_live_llm_produces_valid_pain_points),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Pain Point Miner Component Tests")
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

