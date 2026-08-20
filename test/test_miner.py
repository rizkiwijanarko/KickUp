"""
Unit tests for CompositeDataMiner and SourceProviders.
"""

import pytest
from src.mining import CompositeDataMiner, RawEvidence, SourceProvider
from src.models import DataSource


class FakeSourceProvider:
    def __init__(self, name: str, source_type: DataSource, available: bool, items: list[RawEvidence]) -> None:
        self._name = name
        self._source_type = source_type
        self._available = available
        self._items = items

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_type(self) -> DataSource:
        return self._source_type

    def is_available(self) -> bool:
        return self._available

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        return self._items[:limit]


def test_composite_miner_with_mock_providers():
    p1 = FakeSourceProvider(
        name="hackernews",
        source_type=DataSource.HACKERNEWS,
        available=True,
        items=[
            RawEvidence(
                text="Debugging microservices across multiple clusters is excruciatingly slow.",
                url="https://news.ycombinator.com/item?id=101",
                source=DataSource.HACKERNEWS,
                title="Ask HN: Microservice pain points",
            ),
            RawEvidence(
                text="Duplicate URL test item that should be filtered out by miner deduplication.",
                url="https://news.ycombinator.com/item?id=101",
                source=DataSource.HACKERNEWS,
            ),
        ],
    )
    p2 = FakeSourceProvider(
        name="reddit",
        source_type=DataSource.REDDIT,
        available=False,
        items=[],
    )
    p3 = FakeSourceProvider(
        name="tavily",
        source_type=DataSource.WEB,
        available=True,
        items=[
            RawEvidence(
                text="Setting up local environment for Kubernetes microservices takes days.",
                url="https://dev.to/article/102",
                source=DataSource.WEB,
                title="Kubernetes dev pain",
            )
        ],
    )

    miner = CompositeDataMiner(providers=[p1, p2, p3])
    available = miner.get_available_providers()
    assert len(available) == 2
    assert [p.name for p in available] == ["hackernews", "tavily"]

    # Ingest: since p1 yields only 1 unique item (< min_total_evidence=10), p3 will be activated
    evidence = miner.mine("microservices", min_total_evidence=10)
    assert len(evidence) == 2
    urls = [e.url for e in evidence]
    assert "https://news.ycombinator.com/item?id=101" in urls
    assert "https://dev.to/article/102" in urls

    # Validate quote helper
    found = miner.validate_quote("excruciatingly slow", evidence)
    assert found is not None
    assert found.url == "https://news.ycombinator.com/item?id=101"

    not_found = miner.validate_quote("non existent phrase", evidence)
    assert not_found is None
