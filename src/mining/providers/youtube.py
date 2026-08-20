"""
YouTube comment data mining provider.
"""

from __future__ import annotations

import logging

from src.config import settings
from src.mining.provider import RawEvidence, SourceProvider
from src.models.common import DataSource
from src.tools.youtube_scraper import scrape_for_domain

logger = logging.getLogger(__name__)


class YouTubeProvider:
    """Mines YouTube video comments for domain frustrations."""

    @property
    def name(self) -> str:
        return "youtube"

    @property
    def source_type(self) -> DataSource:
        return DataSource.YOUTUBE

    def is_available(self) -> bool:
        """Check if YouTube API key is configured."""
        return bool(settings.youtube_api_key)

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch pain points from YouTube video comments."""
        if not self.is_available():
            logger.info("[YouTubeProvider] skipped — YOUTUBE_API_KEY not set")
            return []

        try:
            comments = scrape_for_domain(domain, max_total_comments=limit)
            results: list[RawEvidence] = []
            for c in comments:
                results.append(
                    RawEvidence(
                        text=c.text,
                        url=c.url,
                        source=DataSource.YOUTUBE,
                        title=c.post_title,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"[YouTubeProvider] fetch error: {e}")
            return []
