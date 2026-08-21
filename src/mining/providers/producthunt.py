"""
Product Hunt data mining provider.
"""

from __future__ import annotations

import logging

from src.config import get_settings
from src.mining.provider import RawEvidence
from src.models.common import DataSource
from src.tools.producthunt_scraper import scrape_for_domain

logger = logging.getLogger(__name__)


class ProductHuntProvider:
    """Mines Product Hunt comments and discussions."""

    @property
    def name(self) -> str:
        return "producthunt"

    @property
    def source_type(self) -> DataSource:
        return DataSource.PRODUCTHUNT

    def is_available(self) -> bool:
        """Check if Product Hunt API token is set."""
        return bool(get_settings().product_hunt_api_key)

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch pain point comments from Product Hunt."""
        if not self.is_available():
            logger.info("[ProductHuntProvider] skipped — PRODUCT_HUNT_API_KEY not set")
            return []

        try:
            comments = scrape_for_domain(domain, max_total_comments=limit)
            results: list[RawEvidence] = []
            for c in comments:
                results.append(
                    RawEvidence(
                        text=c.text,
                        url=c.url,
                        source=DataSource.PRODUCTHUNT,
                        title=c.post_title,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"[ProductHuntProvider] fetch error: {e}")
            return []
