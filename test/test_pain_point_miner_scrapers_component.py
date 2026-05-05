"""Component-level evaluation for Pain Point Miner *scrapers* (real network).

Goal: debug the scraper side quickly, without paying LLM cost.

What it does:
- Runs the real Reddit JSON scraper (and Tavily fallback if configured)
- Prints counts, per-subreddit coverage, and sample comment stats
- Optionally runs the full pain_point_miner agent with a mocked LLM output
- Optional live LLM path (requires OPENAI_API_KEY)

Run:
    uv run python test_pain_point_miner_scrapers_component.py

Tip: set DOMAIN env var to override the default.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from unittest.mock import MagicMock, patch

from src.agents.pain_point_miner import run as run_pain_point_miner
from src.config import settings
from src.state.schema import VentureForgeState
from src.tools import reddit_scraper as rs
from src.tools import tavily_fallback as tf


def _has_llm_key() -> bool:
    # App config uses LLM_API_KEY / FAST_LLM_API_KEY; some older tests use OPENAI_API_KEY.
    return bool(settings.llm_api_key or settings.fast_llm_api_key or os.getenv("OPENAI_API_KEY"))


def _patch_delays_to_zero() -> None:
    # These are module-level constants; safe to patch in a test harness.
    if hasattr(rs, "_REQUEST_DELAY_S"):
        rs._REQUEST_DELAY_S = 0.0  # type: ignore[attr-defined]
    if hasattr(tf, "_REQUEST_DELAY_S"):
        tf._REQUEST_DELAY_S = 0.0  # type: ignore[attr-defined]


def _summarize_comments(comments: list[rs.ScrapedComment]) -> dict[str, Any]:
    by_sr: dict[str, int] = {}
    lengths: list[int] = []
    urls: set[str] = set()
    dup_urls = 0
    for c in comments:
        by_sr[c.subreddit] = by_sr.get(c.subreddit, 0) + 1
        lengths.append(len(c.text or ""))
        if c.url in urls:
            dup_urls += 1
        urls.add(c.url)

    lengths_sorted = sorted(lengths)
    p50 = lengths_sorted[len(lengths_sorted) // 2] if lengths_sorted else 0
    p90 = lengths_sorted[int(len(lengths_sorted) * 0.9)] if lengths_sorted else 0

    return {
        "count": len(comments),
        "subreddits": dict(sorted(by_sr.items(), key=lambda kv: kv[1], reverse=True)),
        "len_min": min(lengths) if lengths else 0,
        "len_p50": p50,
        "len_p90": p90,
        "len_max": max(lengths) if lengths else 0,
        "unique_urls": len(urls),
        "dup_urls": dup_urls,
    }


def _make_mock_llm_pain_points(comments: list[rs.ScrapedComment], max_pp: int) -> list[dict[str, Any]]:
    # Build valid-looking LLM output where raw_quote is a verbatim substring of provided comments.
    picked = random.sample(comments, k=min(len(comments), max_pp))
    out: list[dict[str, Any]] = []
    for c in picked:
        quote = c.text.strip()
        # Keep quotes short-ish but still valid for validate_quote; take a contiguous slice.
        if len(quote) > 220:
            quote = quote[:220]
        out.append(
            {
                "title": f"Pain from r/{c.subreddit}",
                "description": "Derived from scraped comment for debugging scraper->LLM plumbing.",
                "rubric": {
                    "is_genuine_current_frustration": True,
                    "has_verbatim_quote": True,
                    "user_segment_specific": True,
                },
                "passes_rubric": "yes",
                "source_url": c.url,
                "raw_quote": quote,
                "source": "reddit",
            }
        )
    return out


def main() -> None:
    random.seed(7)
    _patch_delays_to_zero()

    domain = os.getenv("DOMAIN", "developer tools")
    max_pp = int(os.getenv("MAX_PP", "10"))
    max_total_comments = int(os.getenv("MAX_TOTAL_COMMENTS", "60"))

    print("=" * 60, flush=True)
    print("Pain Point Miner — Scrapers Component Eval", flush=True)
    print("=" * 60, flush=True)
    print(f"domain={domain!r} max_pp={max_pp} max_total_comments={max_total_comments}", flush=True)

    # 1) Directly exercise scraper_for_domain (what pain_point_miner uses)
    t0 = time.monotonic()
    comments = rs.scrape_for_domain(domain, max_total_comments=max_total_comments)
    scrape_elapsed = time.monotonic() - t0
    summary = _summarize_comments(comments)
    print("\n[Scrape] reddit_scraper.scrape_for_domain", flush=True)
    print(f"elapsed_s={scrape_elapsed:.2f}", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    # 2) If low comments, show what Tavily would add (if enabled)
    if len(comments) < max(3, max_pp // 2):
        print("\n[Tavily] attempting community discovery (if enabled)", flush=True)
        t1 = time.monotonic()
        extra = tf.search_communities(domain)
        tav_elapsed = time.monotonic() - t1
        print(f"enabled={bool(getattr(tf, 'settings', None) and tf.settings.tavily_enabled)} elapsed_s={tav_elapsed:.2f}", flush=True)
        print(f"extra_subreddits={extra}", flush=True)

    # 3) Validate quote matcher on a few samples (core miner invariant)
    if comments:
        print("\n[Validate] validate_quote sampling", flush=True)
        for c in random.sample(comments, k=min(3, len(comments))):
            quote = c.text.strip()
            if len(quote) > 120:
                quote = quote[:120]
            found, url = rs.validate_quote(quote, comments)
            print(json.dumps({"found": found, "matched_url": url, "subreddit": c.subreddit, "quote_preview": quote[:60]}, ensure_ascii=False), flush=True)

    # 4) Run pain_point_miner with *real* scrape but mocked LLM output (fast)
    print("\n[Agent] run(pain_point_miner) with mocked LLM output", flush=True)
    state = VentureForgeState(domain=domain, max_pain_points=max_pp)
    if not comments:
        print("No comments scraped; skipping mocked-LLM agent run.", flush=True)
    else:
        mock_payload = _make_mock_llm_pain_points(comments, max_pp=max_pp)
        with patch("src.agents.pain_point_miner.get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_response = MagicMock()
            fake_response.content = json.dumps(mock_payload, ensure_ascii=False)
            fake_llm.invoke.return_value = fake_response
            mock_get_llm.return_value = fake_llm

            # Also patch the miner's internal scrape function to reuse our already-fetched comments,
            # so the run is deterministic and doesn't re-hit the network.
            with patch("src.agents.pain_point_miner._tavily_enriched_scrape", return_value=comments):
                result = run_pain_point_miner(state)

        pain_points = result.get("pain_points", [])
        print(f"pain_points_out={len(pain_points)} stage={result.get('current_stage')} next={result.get('next_node')}", flush=True)
        if pain_points:
            pp0 = pain_points[0]
            print(f"sample_pp.title={pp0.title!r}", flush=True)
            print(f"sample_pp.source_url={pp0.source_url!r}", flush=True)
            print(f"sample_pp.quote_preview={pp0.raw_quote[:80]!r}", flush=True)

    # 5) Optional: real LLM run (slow + costs tokens)
    # Opt-in so the scraper eval stays cheap by default.
    if _has_llm_key() and os.getenv("LIVE_LLM") == "1":
        print("\n[Agent] run(pain_point_miner) live LLM (slow)", flush=True)
        live_state = VentureForgeState(domain=domain, max_pain_points=max_pp)
        live = run_pain_point_miner(live_state)
        print(f"pain_points_out={len(live.get('pain_points', []))}", flush=True)
    else:
        print("\n[Agent] live LLM skipped (set LIVE_LLM=1 to enable)", flush=True)


if __name__ == "__main__":
    main()

