"""Component-level test for the Scorer agent.

Tests parsing edge cases and validation paths without hitting the real LLM.
Run with:
    uv run test_scorer_component.py
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.scorer import run as run_scorer
from src.state.schema import (
    DataSource,
    DemandRubric,
    FatalFlaw,
    FeasibilityRubric,
    Idea,
    NoveltyRubric,
    PainPoint,
    PainPointRubric,
    PipelineStage,
    ScoredIdea,
    VentureForgeState,
    Verdict,
)
from test.test_helpers import make_test_pain_point

logging.basicConfig(level=logging.INFO)


def _make_minimal_state() -> VentureForgeState:
    """Build a valid VentureForgeState with two ideas ready to score."""
    pp1 = make_test_pain_point(
        title="Docker Compose is hard",
        description="Developers struggle with complex multi-service local development setups.",
        source_url="https://reddit.com/r/docker/comments/abc123",
        raw_quote="I spend more time debugging docker-compose.yml than writing code.",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="CI debugging is painful",
        description="Developers waste hours reproducing CI failures locally.",
        source_url="https://reddit.com/r/devops/comments/def456",
        raw_quote="Why does my test pass locally but fail in CI?",
        source=DataSource.REDDIT,
    )

    idea_a = Idea(
        id=uuid4(),
        title="Docker Compose Simplifier",
        one_liner="Easily manage and debug Docker Compose files.",
        problem="Developers struggle with complex multi-service local development setups.",
        solution="A visual editor and debugger for Docker Compose files.",
        target_user="Solo developers and small teams",
        key_features=["Visual editor", "Error detection", "Auto-fix suggestions"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )
    idea_b = Idea(
        id=uuid4(),
        title="CI Debugger",
        one_liner="Reproduce CI failures locally in one click.",
        problem="CI pipelines take forever to debug.",
        solution="A tool that mirrors CI environment locally for instant debugging.",
        target_user="DevOps engineers",
        key_features=["CI env mirror", "One-click reproduce", "Log diff"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )

    return VentureForgeState(
        domain="developer tools",
        max_pain_points=10,
        ideas_per_run=3,
        top_n_pitches=2,
        pain_points=[pp1, pp2],
        ideas=[idea_a, idea_b],
    )


def _make_scored_response(
    idea_id: Any,
    *,
    f1: bool = True,
    f2: bool = True,
    f3: bool = True,
    d1: bool = True,
    d2: bool = True,
    d3: bool = True,
    n1: bool = True,
    n2: bool = True,
    verdict: str = "pursue",
    flaws: list[dict] | None = None,
) -> dict:
    """Return a single valid scored-idea dict with tunable rubric booleans."""
    yes_count = sum([f1, f2, f3, d1, d2, d3, n1, n2])
    return {
        "idea_id": str(idea_id),
        "reasoning_trace": "Strong demand signal from Reddit. Manual version is painful.",
        "feasibility_rubric": {
            "can_be_solved_manually_first": f1,
            "has_schlep_or_unsexy_advantage": f2,
            "can_2_3_person_team_build_mvp_in_6_months": f3,
        },
        "demand_rubric": {
            "addresses_at_least_2_pain_points": d1,
            "is_painkiller_not_vitamin": d2,
            "has_clear_vein_of_early_adopters": d3,
        },
        "novelty_rubric": {
            "differentiated_from_current_behavior": n1,
            "has_path_out_of_niche": n2,
        },
        "core_assumption": "Developers will adopt a tool that simplifies Docker Compose management.",
        "fatal_flaws": (flaws if flaws is not None else [{"flaw": "Incumbents may copy quickly.", "severity": "major"}]),
        "yes_count": yes_count,
        "total_checks": 8,
        "verdict": verdict,
        "one_risk": "Incumbents may copy quickly.",
        "rank": 1,
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_no_ideas_returns_empty() -> None:
    """If state has no ideas, scorer should return empty scored_ideas."""
    state = VentureForgeState(domain="test", ideas=[])
    result = run_scorer(state)
    assert result["scored_ideas"] == []
    assert result["current_stage"] == PipelineStage.SCORING
    assert result["next_node"] == "orchestrator"
    print("  PASS")


def test_wellformed_response_produces_scored_ideas() -> None:
    """Happy path: two ideas scored, ranked by yes_count."""
    state = _make_minimal_state()
    idea_a, idea_b = state.ideas

    with patch("src.agents.scorer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps([
            _make_scored_response(idea_a.id, f1=True, f2=True, f3=True, d1=True, d2=True, d3=True, n1=True, n2=True),
            _make_scored_response(idea_b.id, f1=True, f2=False, f3=True, d1=True, d2=False, d3=True, n1=False, n2=True),
        ])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_scorer(state)

    scored: list[ScoredIdea] = result["scored_ideas"]
    assert len(scored) == 2, f"Expected 2 scored ideas, got {len(scored)}"
    # Sorted by yes_count descending
    assert scored[0].idea_id == idea_a.id
    assert scored[0].yes_count == 8
    assert scored[0].verdict == Verdict.PURSUE
    assert scored[0].rank == 1
    assert scored[1].idea_id == idea_b.id
    assert scored[1].yes_count == 5  # True, False, True, True, False, True, False, True = 5
    assert scored[1].verdict == Verdict.EXPLORE
    assert scored[1].rank == 2
    print("  PASS")


def test_missing_idea_id_skips_item() -> None:
    """LLM forgets idea_id → scored idea is silently skipped."""
    state = _make_minimal_state()
    idea_a, idea_b = state.ideas

    with patch("src.agents.scorer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        # First item missing idea_id entirely — should be skipped
        bad = _make_scored_response(idea_a.id)
        bad.pop("idea_id")
        fake_response.content = json.dumps([bad])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_scorer(state)

    scored: list[ScoredIdea] = result["scored_ideas"]
    assert len(scored) == 0, f"Expected 0 scored ideas (missing id), got {len(scored)}"
    print("  PASS")


def test_rubric_yes_no_strings_coerced() -> None:
    """LLM returns 'yes'/'no' strings instead of booleans."""
    state = _make_minimal_state()
    idea_a = state.ideas[0]

    with patch("src.agents.scorer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        scored = _make_scored_response(idea_a.id)
        scored["feasibility_rubric"] = {
            "can_be_solved_manually_first": "yes",
            "has_schlep_or_unsexy_advantage": "no",
            "can_2_3_person_team_build_mvp_in_6_months": "yes",
        }
        scored["demand_rubric"] = {
            "addresses_at_least_2_pain_points": "yes",
            "is_painkiller_not_vitamin": "no",
            "has_clear_vein_of_early_adopters": "yes",
        }
        scored["novelty_rubric"] = {
            "differentiated_from_current_behavior": "yes",
            "has_path_out_of_niche": "yes",
        }
        fake_response.content = json.dumps([scored])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_scorer(state)

    scored_list: list[ScoredIdea] = result["scored_ideas"]
    assert len(scored_list) == 1
    s = scored_list[0]
    assert s.feasibility_rubric.can_be_solved_manually_first is True
    assert s.feasibility_rubric.has_schlep_or_unsexy_advantage is False
    assert s.demand_rubric.is_painkiller_not_vitamin is False
    assert s.yes_count == 6  # 3 True from feasibility + 2 True from demand + 2 True from novelty = 7? Let me count:
    # feasibility: yes, no, yes = 2
    # demand: yes, no, yes = 2
    # novelty: yes, yes = 2
    # total = 6
    assert s.yes_count == 6
    print("  PASS")


def test_fatal_flaw_overrides_verdict() -> None:
    """A fatal severity flaw should force verdict=park regardless of yes_count."""
    state = _make_minimal_state()
    idea_a = state.ideas[0]

    with patch("src.agents.scorer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        scored = _make_scored_response(idea_a.id, verdict="pursue")
        scored["fatal_flaws"] = [{"flaw": "Market is tiny and shrinking.", "severity": "fatal"}]
        fake_response.content = json.dumps([scored])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_scorer(state)

    scored_list: list[ScoredIdea] = result["scored_ideas"]
    assert len(scored_list) == 1
    assert scored_list[0].verdict == Verdict.PARK
    print("  PASS")


def test_live_llm_produces_valid_scored_ideas() -> None:
    """Uses real API call. Requires OPENAI_API_KEY in environment."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP (no OPENAI_API_KEY)")
        return

    state = _make_minimal_state()
    result = run_scorer(state)

    scored: list[ScoredIdea] = result["scored_ideas"]
    assert len(scored) > 0, f"Expected at least 1 scored idea, got {len(scored)}"
    for s in scored:
        assert isinstance(s.idea_id, type(state.ideas[0].id)), f"Invalid idea_id type: {type(s.idea_id)}"
        assert s.yes_count >= 0 and s.yes_count <= 8
        assert s.verdict in (Verdict.PURSUE, Verdict.EXPLORE, Verdict.PARK)
        assert s.rank is not None and s.rank >= 1
    print("  PASS")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

_TESTS = [
    ("No ideas returns empty", test_no_ideas_returns_empty),
    ("Well-formed response produces scored ideas", test_wellformed_response_produces_scored_ideas),
    ("Missing idea_id skips item", test_missing_idea_id_skips_item),
    ("Rubric yes/no strings coerced", test_rubric_yes_no_strings_coerced),
    ("Fatal flaw overrides verdict", test_fatal_flaw_overrides_verdict),
    ("Live LLM (slow)", test_live_llm_produces_valid_scored_ideas),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Scorer Component Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, fn in _TESTS:
        print(f"\n[{passed + failed + 1}] {name}...")
        try:
            fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        import sys
        sys.exit(1)
