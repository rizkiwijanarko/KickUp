"""
Composite Data Miner — coordinates concurrent multi-source extraction with SLA budget.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Sequence

from src.mining.cache import SQLiteEvidenceCache
from src.mining.provider import RawEvidence, SourceProvider
from src.mining.providers.hackernews import HackerNewsProvider
from src.mining.providers.producthunt import ProductHuntProvider
from src.mining.providers.reddit import RedditProvider
from src.mining.providers.tavily import TavilyProvider
from src.mining.providers.youtube import YouTubeProvider

logger = logging.getLogger(__name__)

DEFAULT_INGESTION_SLA_SECONDS = 15.0


class CompositeDataMiner:
    """Orchestrates evidence extraction across all available data source providers concurrently."""

    def __init__(
        self,
        providers: Sequence[SourceProvider] | None = None,
        sla_timeout_s: float = DEFAULT_INGESTION_SLA_SECONDS,
        cache: SQLiteEvidenceCache | None = None,
        use_cache: bool = True,
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
        self.cache = cache if cache is not None else (SQLiteEvidenceCache() if use_cache else None)
        self.use_cache = use_cache

    def get_available_providers(self) -> list[SourceProvider]:
        """Return list of providers whose dependencies / API keys are configured."""
        return [p for p in self.providers if p.is_available()]

    def mine(
        self,
        domain: str,
        limit_per_source: int = 50,
        min_total_evidence: int = 15,
        force_refresh: bool = False,
    ) -> list[RawEvidence]:
        """
        Extract grounded evidence across providers concurrently within the SLA timeout budget.

        1. Checks SQLite evidence cache unless force_refresh is True.
        2. Launches concurrent fetches across all available providers bounded by SLA timeout.
        3. Deduplicates results by URL and ranks by composite engagement score.
        4. Saves final evidence to SQLite cache for future runs.
        """
        if self.cache and not force_refresh:
            cached = self.cache.get(domain)
            if cached:
                return cached

        available = self.get_available_providers()
        skipped = [p.name for p in self.providers if not p.is_available()]
        if skipped:
            logger.info(f"[CompositeDataMiner] Optional/unconfigured providers skipped: {skipped}")

        if not available:
            logger.warning(
                "[CompositeDataMiner] No data source providers are currently available. "
                "Skipping mining (synthetic fallback has been removed)."
            )
            return []

        logger.info(
            f"[CompositeDataMiner] Mining domain='{domain}' concurrently across {len(available)} providers "
            f"(SLA budget: {self.sla_timeout_s}s): {[p.name for p in available]}"
        )

        all_evidence: list[RawEvidence] = []
        seen_urls: set[str] = set()
        t0 = time.monotonic()

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(available),
            thread_name_prefix="miner_worker",
        )
        future_to_provider = {
            executor.submit(p.fetch, domain, limit=limit_per_source): p for p in available
        }

        try:
            for future in concurrent.futures.as_completed(
                future_to_provider, timeout=self.sla_timeout_s
            ):
                provider = future_to_provider[future]
                try:
                    items = future.result()
                    new_count = 0
                    for item in items:
                        if item.url not in seen_urls and len(item.text.strip()) > 30:
                            seen_urls.add(item.url)
                            all_evidence.append(item)
                            new_count += 1
                    logger.info(
                        f"[CompositeDataMiner] Provider '{provider.name}' returned {new_count} valid items."
                    )
                except Exception as exc:
                    logger.warning(
                        f"[CompositeDataMiner] Provider '{provider.name}' raised an error: {exc}"
                    )
        except concurrent.futures.TimeoutError:
            elapsed = time.monotonic() - t0
            pending = [p.name for f, p in future_to_provider.items() if not f.done()]
            logger.warning(
                f"[CompositeDataMiner] Ingestion SLA budget ({self.sla_timeout_s}s) exceeded at {elapsed:.2f}s. "
                f"Aborting slow pending providers: {pending}"
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        elapsed = time.monotonic() - t0

        # Sort by composite engagement score descending
        all_evidence.sort(key=lambda x: x.score, reverse=True)

        if self.cache and all_evidence:
            self.cache.set(domain, all_evidence)

        logger.info(
            f"[CompositeDataMiner] Finished in {elapsed:.2f}s: Extracted {len(all_evidence)} total "
            f"evidence items for domain='{domain}'."
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
