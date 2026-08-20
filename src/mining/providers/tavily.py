"""
Tavily web search data mining provider.
"""

from __future__ import annotations

import logging

from src.config import settings
from src.mining.provider import RawEvidence, SourceProvider
from src.models.common import DataSource
from src.tools.tavily_content_scraper import scrape_for_domain

logger = logging.getLogger(__name__)


class TavilyProvider:
    """Mines forums, social media, and web communities using Tavily Web Search."""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def source_type(self) -> DataSource:
        return DataSource.WEB

    def is_available(self) -> bool:
        """Check if Tavily API key is set."""
        return settings.tavily_enabled

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch pain points from web search."""
        if not self.is_available():
            logger.info("[TavilyProvider] skipped — TAVILY_API_KEY not set")
            return []

        try:
            comments = scrape_for_domain(domain, max_total_comments=limit)
            results: list[RawEvidence] = []
            for c in comments:
                results.append(
                    RawEvidence(
                        text=c.text,
                        url=c.url,
                        source=DataSource.WEB,
                        title=c.post_title,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"[TavilyProvider] fetch error: {e}")
            return []
