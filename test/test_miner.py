from src.mining import CompositeDataMiner, RawEvidence
from src.models import DataSource


class FakeSourceProvider:
    def __init__(
        self, name: str, source_type: DataSource, available: bool, items: list[RawEvidence]
    ) -> None:
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

    # Ingest: with 2 live items (< HYBRID_AUGMENTATION_THRESHOLD=8), hybrid augmentation yields 6 items
    evidence = miner.mine("microservices", min_total_evidence=10)
    assert len(evidence) == 6
    urls = [e.url for e in evidence]
    assert "https://news.ycombinator.com/item?id=101" in urls
    assert "https://dev.to/article/102" in urls
    # Verify synthetic items are present
    synthetic_count = sum(1 for e in evidence if e.metadata.get("synthetic"))
    assert synthetic_count == 4

    # Validate quote helper
    found = miner.validate_quote("excruciatingly slow", evidence)
    assert found is not None
    assert found.url == "https://news.ycombinator.com/item?id=101"

    not_found = miner.validate_quote("non existent phrase", evidence)
    assert not_found is None


def test_composite_miner_engagement_sorting_and_no_augmentation():
    # 10 items (> HYBRID_AUGMENTATION_THRESHOLD=8) with different scores
    items = [
        RawEvidence(
            text=f"Sample evidence text item {i} describing friction in developer tools.",
            url=f"https://news.ycombinator.com/item?id=20{i}",
            source=DataSource.HACKERNEWS,
            score=i * 10,
        )
        for i in range(10)
    ]
    p = FakeSourceProvider(
        name="hackernews", source_type=DataSource.HACKERNEWS, available=True, items=items
    )
    miner = CompositeDataMiner(providers=[p])
    evidence = miner.mine("developer tools")

    assert len(evidence) == 10
    # Verify no synthetic items
    assert all(not e.metadata.get("synthetic") for e in evidence)
    # Verify sorted descending by score
    scores = [e.score for e in evidence]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 90
    assert scores[-1] == 0
