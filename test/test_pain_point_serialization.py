"""
Unit tests for pain point miner evidence serialization and prompt building.
"""

from src.agents.pain_point_miner import _build_user_prompt, _serialize_evidence
from src.mining.provider import RawEvidence
from src.models.common import DataSource


def _evidence(text: str, url: str, source: DataSource, score: int) -> RawEvidence:
    return RawEvidence(
        text=text,
        url=url,
        source=source,
        title="Post title",
        score=score,
    )


def test_serialize_evidence_ranks_by_score_desc():
    items = [
        _evidence("Low engagement comment about developer tools.", "https://a/1", DataSource.HACKERNEWS, 5),
        _evidence("High engagement comment about developer tools.", "https://a/2", DataSource.HACKERNEWS, 99),
    ]
    serialized = _serialize_evidence(items, limit=10)
    assert [s["score"] for s in serialized] == [99, 5]


def test_serialize_evidence_applies_per_source_cap():
    items = []
    for i in range(12):  # 12 HN items, all with high scores
        items.append(
            _evidence(
                f"Comment {i} about developer tools with plenty of words.",
                f"https://news.ycombinator.com/item?id={i}",
                DataSource.HACKERNEWS,
                100 - i,
            )
        )
    for i in range(5):  # 5 YT items with lower scores
        items.append(
            _evidence(
                f"YouTube comment {i} about developer tooling frustrations.",
                f"https://youtube.com/watch?v=abc&lc={i}",
                DataSource.YOUTUBE,
                10 - i,
            )
        )

    serialized = _serialize_evidence(items, limit=15)
    # Per-source cap of 8 for HN, plus 5 YT => 13 total (within global 15)
    assert len(serialized) == 13
    hn_count = sum(1 for s in serialized if s["source"] == "hackernews")
    yt_count = sum(1 for s in serialized if s["source"] == "youtube")
    assert hn_count == 8
    assert yt_count == 5
    # Global cap respected
    assert len(_serialize_evidence(items, limit=10)) == 10


def test_serialize_evidence_global_limit_applies():
    items = [
        _evidence(
            f"Comment {i} about tools with sufficient words here.",
            f"https://a/{i}",
            DataSource.HACKERNEWS if i % 2 == 0 else DataSource.YOUTUBE,
            i,
        )
        for i in range(20)
    ]
    serialized = _serialize_evidence(items, limit=15)
    assert len(serialized) == 15
    assert serialized[0]["score"] == 19  # highest engagement first


def test_build_user_prompt_includes_engagement_score():
    serialized = [
        {
            "text": "This comment describes a real pain point.",
            "url": "https://news.ycombinator.com/item?id=1",
            "source": "hackernews",
            "post_title": "Ask HN: Tooling",
            "score": 42,
        }
    ]
    prompt = _build_user_prompt("developer tools", 5, serialized)
    assert "Engagement: 42" in prompt
    assert "https://news.ycombinator.com/item?id=1" in prompt
