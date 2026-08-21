"""
Reddit data mining provider using PRAW.
"""

from __future__ import annotations

import logging

from src.config import settings
from src.mining.provider import RawEvidence
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
        """Check if Reddit API credentials are configured with valid non-placeholder values."""
        client_id = settings.reddit_client_id
        client_secret = settings.reddit_client_secret
        if not client_id or not client_secret:
            return False
        if client_id.startswith("your_") or client_secret.startswith("your_"):
            return False
        return True

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch pain point comments from Reddit."""
        if not self.is_available():
            logger.info("[RedditProvider] skipped — credentials not configured")
            return []

        try:
            comments = scrape_for_domain(domain, max_total_comments=limit)
            results: list[RawEvidence] = []
            for c in comments:
                results.append(
                    RawEvidence(
                        text=c.text,
                        url=c.url,
                        source=DataSource.REDDIT,
                        title=c.post_title,
                        score=int((c.score or 0) * 1.0 + (c.num_comments or 0) * 0.5),
                        metadata={"points": c.score, "num_comments": c.num_comments},
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"[RedditProvider] fetch error: {e}")
            return []
