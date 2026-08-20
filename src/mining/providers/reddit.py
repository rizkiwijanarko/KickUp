"""
Reddit data mining provider using PRAW.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.mining.provider import RawEvidence, SourceProvider
from src.models.common import DataSource
from src.tools.reddit_scraper import scrape_for_domain

logger = logging.getLogger(__name__)


class RedditProvider:
    """Mines Reddit threads and comments using PRAW."""

    @property
    def name(self) -> str:
        return "reddit"

    @property
    def source_type(self) -> DataSource:
        return DataSource.REDDIT

    def is_available(self) -> bool:
        """Check if Reddit API credentials are configured."""
        return bool(settings.reddit_client_id and settings.reddit_client_secret)

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch pain point comments from Reddit."""
        if not self.is_available():
            logger.info("[RedditProvider] skipped — credentials not configured")
            return []

        try:
            comments = scrape_domain(domain, max_total_comments=limit)
            results: list[RawEvidence] = []
            for c in comments:
                results.append(
                    RawEvidence(
                        text=c.text,
                        url=c.url,
                        source=DataSource.REDDIT,
                        title=c.post_title,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"[RedditProvider] fetch error: {e}")
            return []
