"""
Hacker News data mining provider using Algolia API.
"""

from __future__ import annotations

import logging

from src.mining.provider import RawEvidence
from src.models.common import DataSource
from src.tools.hackernews_scraper import scrape_for_domain

logger = logging.getLogger(__name__)


class HackerNewsProvider:
    """Mines Hacker News comments and stories via the free Algolia API."""

    @property
    def name(self) -> str:
        return "hackernews"

    @property
    def source_type(self) -> DataSource:
        return DataSource.HACKERNEWS

    def is_available(self) -> bool:
        """Hacker News Algolia search is public and requires no API key."""
        return True

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch pain point comments from Hacker News."""
        try:
            comments = scrape_for_domain(domain, max_total_comments=limit)
            results: list[RawEvidence] = []
            for c in comments:
                composite_score = int((c.score or 0) * 1.0 + (c.num_comments or 0) * 2.0)
                results.append(
                    RawEvidence(
                        text=c.text,
                        url=c.url,
                        source=DataSource.HACKERNEWS,
                        title=c.post_title,
                        score=composite_score,
                        metadata={"points": c.score, "num_comments": c.num_comments},
                    )
                )

            return results
        except Exception as e:
            logger.warning(f"[HackerNewsProvider] fetch error: {e}")
            return []
