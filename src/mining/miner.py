"""
Composite Data Miner — coordinates concurrent multi-source extraction with SLA budget.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Sequence

from src.mining.provider import RawEvidence, SourceProvider
from src.mining.providers.hackernews import HackerNewsProvider
from src.mining.providers.producthunt import ProductHuntProvider
from src.mining.providers.reddit import RedditProvider
from src.mining.providers.tavily import TavilyProvider
from src.mining.providers.youtube import YouTubeProvider

logger = logging.getLogger(__name__)

DEFAULT_INGESTION_SLA_SECONDS = 5.0


class CompositeDataMiner:
    """Orchestrates evidence extraction across all available data source providers concurrently."""

    def __init__(
        self,
        providers: Sequence[SourceProvider] | None = None,
        sla_timeout_s: float = DEFAULT_INGESTION_SLA_SECONDS,
    ) -> None:
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
        self.sla_timeout_s = sla_timeout_s

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
        Extract grounded evidence across providers concurrently within the SLA timeout budget.

        1. Launches concurrent fetches across all available providers.
        2. Bounded by self.sla_timeout_s (default 5.0s).
        3. Deduplicates results by URL.
        """
        available = self.get_available_providers()
        if not available:
            logger.warning("[CompositeDataMiner] No data source providers are currently available.")
            return []

        logger.info(
            f"[CompositeDataMiner] Mining domain='{domain}' concurrently across {len(available)} providers "
            f"(SLA budget: {self.sla_timeout_s}s): {[p.name for p in available]}"
        )

        all_evidence: list[RawEvidence] = []
        seen_urls: set[str] = set()
        t0 = time.monotonic()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(available)) as executor:
            future_to_provider = {
                executor.submit(p.fetch, domain, limit=limit_per_source): p
                for p in available
            }

            try:
                for future in concurrent.futures.as_completed(future_to_provider, timeout=self.sla_timeout_s):
                    provider = future_to_provider[future]
                    try:
                        items = future.result()
                        new_count = 0
                        for item in items:
                            if item.url not in seen_urls and len(item.text.strip()) > 30:
                                seen_urls.add(item.url)
                                all_evidence.append(item)
                                new_count += 1
                        logger.info(f"[CompositeDataMiner] Provider '{provider.name}' returned {new_count} valid items.")
                    except Exception as exc:
                        logger.warning(f"[CompositeDataMiner] Provider '{provider.name}' raised an error: {exc}")
            except concurrent.futures.TimeoutError:
                elapsed = time.monotonic() - t0
                pending = [p.name for f, p in future_to_provider.items() if not f.done()]
                logger.warning(
                    f"[CompositeDataMiner] Ingestion SLA budget ({self.sla_timeout_s}s) exceeded at {elapsed:.2f}s. "
                    f"Aborting slow pending providers: {pending}"
                )

        elapsed = time.monotonic() - t0
        logger.info(
            f"[CompositeDataMiner] Finished in {elapsed:.2f}s: Extracted {len(all_evidence)} total "
            f"deduplicated evidence items for domain='{domain}'."
        )
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
