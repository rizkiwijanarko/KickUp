"""Component-level test for the Pain Point Miner agent.

Fast, deterministic, offline by mocking the DataMiner + LLM.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.pain_point_miner import run as run_pain_point_miner
from src.mining import CompositeDataMiner, RawEvidence
from src.models import DataSource, PainPoint
from src.state.schema import VentureForgeState


logging.basicConfig(level=logging.INFO)


def _make_evidence() -> list[RawEvidence]:
    return [
        RawEvidence(
            text="I spend more time debugging docker-compose.yml than writing actual code.",
            url="https://www.reddit.com/r/docker/comments/abc123/comment/xyz",
            source=DataSource.REDDIT,
            title="Frustrated with docker compose",
        ),
        RawEvidence(
            text="Why does my test pass locally but fail in CI with the exact same Dockerfile?",
            url="https://www.reddit.com/r/devops/comments/def456/comment/qwe",
            source=DataSource.REDDIT,
            title="CI debugging is awful",
        ),
    ]


def _make_pp_dict(url: str) -> dict:
    return {
        "id": str(uuid4()),
        "title": "Docker Compose debugging pain",
        "description": "Developers struggle to manage multi-service setups and chase config errors.",
        "rubric": {
            "is_genuine_current_frustration": True,
            "has_verbatim_quote": True,
            "user_segment_specific": True,
        },
        "passes_rubric": True,
        "source_url": url,
        "raw_quote": "I spend more time debugging docker-compose.yml than writing actual code.",
        "source": DataSource.REDDIT.value,
    }


def test_no_evidence_returns_empty() -> None:
    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    with patch.object(CompositeDataMiner, "mine", return_value=[]):
        result = run_pain_point_miner(state)
    assert result["pain_points"] == []
    assert "events" in result


def test_wellformed_llm_response_extracts_pain_points() -> None:
    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    evidence = _make_evidence()
    mock_payload = [_make_pp_dict(evidence[0].url)]

    with patch.object(CompositeDataMiner, "mine", return_value=evidence):
        with patch("src.agents.pain_point_miner.get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_resp = MagicMock()
            fake_resp.content = json.dumps(mock_payload)
            fake_llm.invoke.return_value = fake_resp
            mock_get_llm.return_value = fake_llm

            result = run_pain_point_miner(state)

    pps: list[PainPoint] = result["pain_points"]
    assert len(pps) == 1
    assert pps[0].title == "Docker Compose debugging pain"
    assert pps[0].passes_rubric is True


def test_multi_evidence_clustering_and_quote_filtering() -> None:
    state = VentureForgeState(domain="developer tools", max_pain_points=5)
    evidence = _make_evidence()

    mock_payload = [
        {
            "id": str(uuid4()),
            "title": "CI and Docker Configuration Frustration",
            "description": "Engineers spend too much time debugging environment inconsistencies.",
            "rubric": {
                "is_genuine_current_frustration": True,
                "has_verbatim_quote": True,
                "user_segment_specific": True,
            },
            "passes_rubric": True,
            "evidence": [
                {
                    "source_url": evidence[0].url,
                    "raw_quote": "I spend more time debugging docker-compose.yml than writing actual code.",
                    "source": "reddit",
                },
                {
                    "source_url": evidence[1].url,
                    "raw_quote": "Why does my test pass locally but fail in CI with the exact same Dockerfile?",
                    "source": "reddit",
                },
                {
                    "source_url": "https://hallucinated.example.com",
                    "raw_quote": "This quote is completely fabricated and does not exist in evidence.",
                    "source": "web",
                },
            ],
        }
    ]

    with patch.object(CompositeDataMiner, "mine", return_value=evidence):
        with patch("src.agents.pain_point_miner.get_llm") as mock_get_llm:
            fake_llm = MagicMock()
            fake_resp = MagicMock()
            fake_resp.content = json.dumps(mock_payload)
            fake_llm.invoke.return_value = fake_resp
            mock_get_llm.return_value = fake_llm

            result = run_pain_point_miner(state)

    pps: list[PainPoint] = result["pain_points"]
    assert len(pps) == 1
    # 2 grounded quotes preserved, 1 hallucinated quote dropped
    assert len(pps[0].evidence) == 2
    assert (
        pps[0].evidence[0].raw_quote
        == "I spend more time debugging docker-compose.yml than writing actual code."
    )
    assert (
        pps[0].evidence[1].raw_quote
        == "Why does my test pass locally but fail in CI with the exact same Dockerfile?"
    )
