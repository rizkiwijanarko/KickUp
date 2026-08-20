"""
Composite Data Miner — coordinates multi-source extraction with fallback cascades.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.mining.provider import RawEvidence, SourceProvider
from src.mining.providers.hackernews import HackerNewsProvider
from src.mining.providers.producthunt import ProductHuntProvider
from src.mining.providers.reddit import RedditProvider
from src.mining.providers.tavily import TavilyProvider
from src.mining.providers.youtube import YouTubeProvider

logger = logging.getLogger(__name__)


class CompositeDataMiner:
    """Orchestrates evidence extraction across all available data source providers."""

    def __init__(self, providers: Sequence[SourceProvider] | None = None) -> None:
        if providers is None:
            self.providers: list[SourceProvider] = [
                RedditProvider(),
                HackerNewsProvider(),
                ProductHuntProvider(),
                TavilyProvider(),
                YouTubeProvider(),
            ]
        else:
            self.providers = list(providers)

    def get_available_providers(self) -> list[SourceProvider]:
        """Return list of providers whose dependencies / API keys are configured."""
        return [p for p in self.providers if p.is_available()]

    def mine(
        self,
        domain: str,
        limit_per_source: int = 50,
        min_total_evidence: int = 15,
    ) -> list[RawEvidence]:
        """
        Extract grounded evidence across providers with automatic cascade.

        1. Queries high-signal community sources (Reddit, Hacker News, Product Hunt).
        2. If total evidence is below `min_total_evidence`, triggers web search (Tavily/YouTube).
        3. Deduplicates results by URL.
        """
        available = self.get_available_providers()
        logger.info(
            f"[CompositeDataMiner] Mining domain='{domain}' with {len(available)} available providers: "
            f"{[p.name for p in available]}"
        )

        all_evidence: list[RawEvidence] = []
        seen_urls: set[str] = set()

        # Phase 1: Community forums (HN, Reddit, Product Hunt)
        primary_providers = [p for p in available if p.name in ("reddit", "hackernews", "producthunt")]
        for provider in primary_providers:
            try:
                evidence_items = provider.fetch(domain, limit=limit_per_source)
                for item in evidence_items:
                    if item.url not in seen_urls and len(item.text.strip()) > 30:
                        seen_urls.add(item.url)
                        all_evidence.append(item)
            except Exception as e:
                logger.warning(f"[CompositeDataMiner] Error in primary provider '{provider.name}': {e}")

        # Phase 2: Web Search & Media Fallback if needed
        if len(all_evidence) < min_total_evidence:
            logger.info(
                f"[CompositeDataMiner] Evidence count ({len(all_evidence)}) below threshold ({min_total_evidence}). "
                f"Activating search fallback providers."
            )
            fallback_providers = [p for p in available if p.name in ("tavily", "youtube")]
            for provider in fallback_providers:
                try:
                    evidence_items = provider.fetch(domain, limit=limit_per_source)
                    for item in evidence_items:
                        if item.url not in seen_urls and len(item.text.strip()) > 30:
                            seen_urls.add(item.url)
                            all_evidence.append(item)
                except Exception as e:
                    logger.warning(f"[CompositeDataMiner] Error in fallback provider '{provider.name}': {e}")

        logger.info(f"[CompositeDataMiner] Extracted {len(all_evidence)} total evidence items for domain='{domain}'")
        return all_evidence

    @staticmethod
    def validate_quote(quote: str, evidence: list[RawEvidence]) -> RawEvidence | None:
        """Find the evidence item containing the given quote (case-insensitive substring)."""
        quote_clean = quote.lower().strip()
        if not quote_clean:
            return None

        for item in evidence:
            if quote_clean in item.text.lower():
                return item

        return None
