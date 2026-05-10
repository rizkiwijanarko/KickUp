"""Component-level test for the Pitch Writer agent.

Tests parsing edge cases and validation paths without hitting the real LLM.
Run with:
    uv run test_pitch_writer_component.py
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.pitch_writer import run as run_pitch_writer
from src.state.schema import (
    DataSource,
    DemandRubric,
    FeasibilityRubric,
    Idea,
    NoveltyRubric,
    PainPoint,
    PainPointRubric,
    PitchBrief,
    PipelineStage,
    ScoredIdea,
    VentureForgeState,
    Verdict,
)
from test.test_helpers import make_test_pain_point

logging.basicConfig(level=logging.INFO)


def _make_minimal_state() -> VentureForgeState:
    """Build a valid VentureForgeState with one scored idea and brief."""
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

    idea = Idea(
        id=uuid4(),
        title="Docker Compose Simplifier",
        one_liner="Easily manage and debug Docker Compose files.",
        problem="Developers struggle with complex multi-service local development setups.",
        solution="A visual editor and debugger for Docker Compose files.",
        target_user="Solo developers and small teams",
        key_features=["Visual editor", "Error detection", "Auto-fix suggestions"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )

    scored = ScoredIdea(
        idea_id=idea.id,
        reasoning_trace="Strong demand signal from Reddit.",
        feasibility_rubric=FeasibilityRubric(
            can_be_solved_manually_first=True,
            has_schlep_or_unsexy_advantage=True,
            can_2_3_person_team_build_mvp_in_6_months=True,
        ),
        demand_rubric=DemandRubric(
            addresses_at_least_2_pain_points=True,
            is_painkiller_not_vitamin=True,
            has_clear_vein_of_early_adopters=True,
        ),
        novelty_rubric=NoveltyRubric(
            differentiated_from_current_behavior=True,
            has_path_out_of_niche=True,
        ),
        core_assumption="Developers will adopt a tool that simplifies Docker Compose management.",
        fatal_flaws=[],
        yes_count=8,
        verdict=Verdict.PURSUE,
        one_risk="Incumbents may copy quickly.",
        rank=1,
    )

    return VentureForgeState(
        domain="developer tools",
        max_pain_points=10,
        ideas_per_run=3,
        top_n_pitches=2,
        pain_points=[pp1, pp2],
        ideas=[idea],
        scored_ideas=[scored],
    )


def _make_pitch_brief_response(idea_id: Any) -> dict:
    """Return a single valid pitch-brief dict."""
    return {
        "idea_id": str(idea_id),
        "title": "Docker Compose Simplifier",
        "tagline": "Easily manage and debug Docker Compose files.",
        "problem": "Developers struggle with complex multi-service local development setups.",
        "solution": "A visual editor and debugger for Docker Compose files.",
        "target_user": "Solo developers and small teams",
        "market_opportunity": "Large developer tools market with growing Docker adoption.",
        "competitive_landscape": {
            "current_behavior": "Developers manually edit YAML files and debug via trial-and-error restarts",
            "direct_competitors": "Docker Desktop, VS Code extensions, and manual YAML editing",
            "real_enemy": "The habit of editing raw YAML without validation or visual feedback"
        },
        "differentiation": "Visual editor with real-time validation vs manual YAML editing",
        "validation_plan": {
            "discovery_questions": [
                "Walk me through the last time you debugged a Docker Compose issue",
                "How much time do you spend on Docker Compose configuration weekly?",
                "What frustrates you most about your current workflow?",
                "What would make you switch from your current approach?",
                "How do you currently validate your Docker Compose files?"
            ],
            "validation_criteria": "At least 7 out of 10 developers mention spending 2+ hours/week on Docker Compose debugging"
        },
        "business_model": "Monthly SaaS subscription model with freemium tier included.",
        "go_to_market": "Post on Reddit and Hacker News for early adopters.",
        "key_risk": "Incumbents may copy the feature quickly.",
        "next_steps": "Build MVP and recruit beta users from Reddit.",
        "evidence_links": ["https://reddit.com/r/docker/comments/abc123"],
        "markdown_content": (
            "# Docker Compose Simplifier\n\n"
            "## Problem\n"
            "Developers struggle with complex multi-service local development setups.\n\n"
            "## Solution\n"
            "A visual editor and debugger for Docker Compose files.\n"
        ),
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_no_scored_ideas_returns_empty() -> None:
    """If state has no scored ideas, return empty pitch_briefs."""
    state = VentureForgeState(domain="test", scored_ideas=[])
    result = run_pitch_writer(state)
    assert result["pitch_briefs"] == []
    assert result["current_stage"] == PipelineStage.WRITING
    print("  PASS")


def test_wellformed_response_produces_briefs() -> None:
    """Happy path: one scored idea → one pitch brief."""
    state = _make_minimal_state()
    idea = state.ideas[0]

    with patch("src.agents.pitch_writer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps([_make_pitch_brief_response(idea.id)])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 1, f"Expected 1 brief, got {len(briefs)}"
    assert briefs[0].idea_id == idea.id
    assert briefs[0].title == "Docker Compose Simplifier"
    assert briefs[0].revision_count == 0
    print("  PASS")


def test_missing_field_skips_brief() -> None:
    """LLM omits market_opportunity → brief is skipped (Pydantic validation fails)."""
    state = _make_minimal_state()
    idea = state.ideas[0]

    with patch("src.agents.pitch_writer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        bad = _make_pitch_brief_response(idea.id)
        bad.pop("market_opportunity")
        fake_response.content = json.dumps([bad])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 0, f"Expected 0 briefs (missing market_opportunity), got {len(briefs)}"
    print("  PASS")


def test_next_steps_list_coerced_to_string() -> None:
    """LLM returns next_steps as a list of strings → coerced to single string."""
    state = _make_minimal_state()
    idea = state.ideas[0]

    with patch("src.agents.pitch_writer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        payload = _make_pitch_brief_response(idea.id)
        payload["next_steps"] = [
            "Build a landing page.",
            "Recruit 10 beta users from Reddit.",
            "Set up analytics tracking.",
        ]
        fake_response.content = json.dumps([payload])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 1
    brief = briefs[0]
    assert isinstance(brief.next_steps, str), f"Expected str, got {type(brief.next_steps)}"
    assert "Build a landing page." in brief.next_steps
    assert "Recruit 10 beta users from Reddit." in brief.next_steps
    print("  PASS")


def test_revision_count_increments() -> None:
    """If state.revision_counts[idea_id] = 1, brief.revision_count should be 1."""
    state = _make_minimal_state()
    idea = state.ideas[0]
    state = state.model_copy(update={"revision_counts": {str(idea.id): 1}})

    with patch("src.agents.pitch_writer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps([_make_pitch_brief_response(idea.id)])
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 1
    assert briefs[0].revision_count == 1
    print("  PASS")


def test_live_llm_produces_valid_briefs() -> None:
    """Uses real API call. Requires OPENAI_API_KEY in environment."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP (no OPENAI_API_KEY)")
        return

    state = _make_minimal_state()
    result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) > 0, f"Expected at least 1 brief, got {len(briefs)}"
    for b in briefs:
        assert b.idea_id is not None
        assert len(b.title) > 0
        assert len(b.tagline) > 0
        assert len(b.problem) > 0
        assert len(b.solution) > 0
        assert len(b.market_opportunity) > 0
        assert len(b.business_model) > 0
        assert len(b.go_to_market) > 0
        assert len(b.markdown_content) >= 100
        # Phase 2: Check new fields
        assert b.competitive_landscape is not None
        assert len(b.competitive_landscape.current_behavior) > 0
        assert len(b.competitive_landscape.direct_competitors) > 0
        assert len(b.competitive_landscape.real_enemy) > 0
        assert len(b.differentiation) > 0
        assert b.validation_plan is not None
        assert len(b.validation_plan.discovery_questions) == 5
        assert len(b.validation_plan.validation_criteria) > 0
    print("  PASS")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

_TESTS = [
    ("No scored ideas returns empty", test_no_scored_ideas_returns_empty),
    ("Well-formed response produces briefs", test_wellformed_response_produces_briefs),
    ("Missing market_opportunity skips brief", test_missing_field_skips_brief),
    ("next_steps list coerced to string", test_next_steps_list_coerced_to_string),
    ("Revision count increments", test_revision_count_increments),
    ("Live LLM (slow)", test_live_llm_produces_valid_briefs),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Pitch Writer Component Tests")
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
