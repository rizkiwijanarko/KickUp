"""Component-level test for the Critic agent.

Tests three paths:
1. Real LLM call with synthetic state (slow, tests prompt/schema alignment).
2. Max-revisions auto-approve (fast, deterministic).
3. Malformed LLM response recovery (mocked).

Run with:
    uv run test_critic_component.py

For the live LLM test, set environment OPENAI_API_KEY.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.critic import _invoke_llm, run as run_critic
from src.state.schema import (
    CompetitiveLandscape,
    Critique,
    CritiqueRubric,
    DataSource,
    DemandRubric,
    FatalFlaw,
    FeasibilityRubric,
    Idea,
    NoveltyRubric,
    PainPoint,
    PainPointRubric,
    PitchBrief,
    PipelineStage,
    ScoredIdea,
    ValidationPlan,
    VentureForgeState,
)
from test.test_helpers import make_test_pain_point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------

def _make_minimal_state(
    revision_counts: dict | None = None,
    max_revisions: int = 2,
) -> VentureForgeState:
    """Build a valid VentureForgeState with exactly one pitch brief ready to critique."""
    pp1 = make_test_pain_point(
        title="Developers struggle with Docker Compose",
        description="Managing multi-service local dev setups is painful and error-prone.",
        source_url="https://reddit.com/r/docker/comments/abc123",
        raw_quote="I spend more time debugging docker-compose.yml than writing code.",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="CI pipelines take forever to debug",
        description="Developers waste hours reproducing CI failures locally.",
        source_url="https://reddit.com/r/devops/comments/def456",
        raw_quote="Why does my test pass locally but fail in CI with the same Dockerfile?",
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
        fatal_flaws=[FatalFlaw(flaw="Incumbents may copy the feature quickly.", severity="major")],
        yes_count=8,
        verdict="pursue",
        one_risk="Incumbents may copy the feature quickly.",
        rank=1,
    )

    brief = PitchBrief(
        idea_id=idea.id,
        title="Docker Compose Simplifier",
        tagline="Easily manage and debug Docker Compose files.",
        problem="Developers struggle with complex multi-service local development setups.",
        solution="A visual editor and debugger for Docker Compose files.",
        target_user="Solo developers and small teams",
        market_opportunity="Large developer tools market with growing Docker adoption.",
        competitive_landscape=CompetitiveLandscape(
            current_behavior="Developers manually edit YAML files and debug via trial-and-error restarts",
            direct_competitors="Docker Desktop, VS Code extensions, and manual YAML editing",
            real_enemy="The habit of editing raw YAML without validation or visual feedback"
        ),
        differentiation="Visual editor with real-time validation vs manual YAML editing",
        validation_plan=ValidationPlan(
            discovery_questions=[
                "Walk me through the last time you debugged a Docker Compose issue",
                "How much time do you spend on Docker Compose configuration weekly?",
                "What frustrates you most about your current workflow?",
                "What would make you switch from your current approach?",
                "How do you currently validate your Docker Compose files?"
            ],
            validation_criteria="At least 7 out of 10 developers mention spending 2+ hours/week on Docker Compose debugging"
        ),
        business_model="Monthly SaaS subscription model with freemium tier included.",
        go_to_market="Post on Reddit and Hacker News for early adopters.",
        key_risk="Incumbents may copy the feature quickly.",
        next_steps="Build MVP and recruit beta users from Reddit.",
        evidence_links=["https://reddit.com/r/docker/comments/abc123"],
        markdown_content=(
            "# Docker Compose Simplifier\n\n"
            "## Problem\n"
            "Developers struggle with complex multi-service local development setups.\n\n"
            "## Solution\n"
            "A visual editor and debugger for Docker Compose files.\n"
        ),
    )

    return VentureForgeState(
        domain="developer tools",
        max_pain_points=10,
        ideas_per_run=3,
        top_n_pitches=2,
        max_revisions=max_revisions,
        pain_points=[pp1, pp2],
        ideas=[idea],
        scored_ideas=[scored],
        pitch_briefs=[brief],
        revision_counts=(revision_counts or {}),
    )


def _make_wellformed_critic_response() -> str:
    """Complete, well-formed LLM response with current 8-field rubric (Phase 3)."""
    return json.dumps(
        {
            "reasoning_trace": (
                "The tagline is 7 words. The target user 'solo developers and small teams' "
                "is too broad - not a contained fire. The competition section mentions Docker Desktop "
                "but lacks a clear thesis on why they won't solve this for the target niche. "
                "Evidence has only 1 URL, needs at least 2. "
                "validation_plan has exactly 5 questions and they are all open-ended."
            ),
            "rubric": {
                "all_claims_evidence_backed": True,
                "no_hallucinated_source_urls": True,
                "tagline_under_12_words": True,
                "target_is_contained_fire": False,  # This fails
                "competition_embraced_with_thesis": False,  # This fails
                "minimum_evidence_sources": False,  # This fails
                "scorer_verdict_justified": True,
                "validation_plan_complete": True,  # Phase 3: new field
            },
            "all_pass": False,
            "approval_status": "revise",
            "failing_checks": [
                "target_is_contained_fire",
                "competition_embraced_with_thesis",
                "minimum_evidence_sources",
            ],
            "target_agent": "idea_generator",  # Positioning issues route to idea_generator
            "revision_feedback": (
                "Target user is too broad — replace 'solo developers and small teams' with "
                "a specific reachable community (e.g., 'React Native developers using Docker for mobile dev'). "
                "Competition thesis is weak — explain why Docker Desktop won't serve this niche (e.g., "
                "'Docker Desktop is optimized for backend devs, not mobile workflows'). "
                "Add at least one more distinct source URL to evidence_links."
            ),
        }
    )


def _make_malformed_critic_response() -> str:
    """LLM response missing required fields (target_agent, revision_feedback)."""
    return json.dumps(
        {
            "reasoning_trace": "Some reasoning here.",
            "rubric": {
                "all_claims_evidence_backed": True,
                "no_hallucinated_source_urls": True,
                "tagline_under_12_words": True,
                "target_is_contained_fire": True,
                "competition_embraced_with_thesis": True,
                "minimum_evidence_sources": True,
                "scorer_verdict_justified": True,
            },
            "all_pass": True,
            "approval_status": "approved",
            # Missing: target_agent, revision_feedback
        }
    )


# ------------------------------------------------------------------
# Plain test functions (no pytest / no class wrappers)
# ------------------------------------------------------------------

def test_auto_approve_at_max_revisions() -> None:
    """If revision_counts[idea_id] >= max_revisions, auto-approve with special status."""
    with patch("src.llm.client.get_llm") as mock_get_llm:
        # Mock LLM to return a failing critique with only positioning issues
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps({
            "reasoning_trace": "Pitch has positioning issues but max revisions reached.",
            "rubric": {
                "all_claims_evidence_backed": True,
                "no_hallucinated_source_urls": True,
                "tagline_under_12_words": True,
                "target_is_contained_fire": False,  # Positioning issue
                "competition_embraced_with_thesis": False,  # Positioning issue
                "minimum_evidence_sources": True,
                "scorer_verdict_justified": True,
                "validation_plan_complete": True,
            },
            "all_pass": False,
            "approval_status": "revise",
            "failing_checks": ["target_is_contained_fire", "competition_embraced_with_thesis"],
            "target_agent": "idea_generator",
            "revision_feedback": "Positioning issues but max revisions reached.",
        })
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        state = _make_minimal_state(max_revisions=2)
        idea_id = str(state.pitch_briefs[0].idea_id)
        state = state.model_copy(update={"revision_counts": {idea_id: 2}})

        result = run_critic(state)
        assert "critique" in result, "Expected critique in result"
        critique: Critique = result["critique"]
        # Max revisions reached: all_pass stays False but approval_status changes
        assert critique.all_pass is False, f"Expected all_pass=False (rubric still failed), got {critique.all_pass}"
        assert critique.approval_status == "max_revisions_reached", f"Expected max_revisions_reached, got {critique.approval_status}"
        # Target agent is determined by rubric priority (positioning issues → idea_generator)
        assert critique.target_agent == "idea_generator", f"Expected idea_generator (positioning issues), got {critique.target_agent}"
        assert "Max revisions reached" in critique.revision_feedback
        print("  PASS")


def test_empty_pitch_briefs_returns_no_critique() -> None:
    state = VentureForgeState(domain="test", pitch_briefs=[])
    result = run_critic(state)
    assert "critique" not in result, "Should not return critique when no briefs"
    assert result["current_stage"] == PipelineStage.CRITIQUING
    print("  PASS")


def test_wellformed_llm_response_parsed_correctly() -> None:
    """Mock a perfect LLM response and ensure we get a valid Critique."""
    with patch("src.agents.critic.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = _make_wellformed_critic_response()
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        state = _make_minimal_state()
        result = run_critic(state)

        assert "critique" in result, f"Expected critique, got keys={list(result.keys())}"
        critique: Critique = result["critique"]
        assert critique.approval_status == "revise", f"Got status={critique.approval_status}"
        assert critique.all_pass is False
        assert critique.target_agent == "idea_generator", f"Expected idea_generator, got {critique.target_agent}"
        assert len(critique.failing_checks) == 3, f"Expected 3 failing checks, got {critique.failing_checks}"
        assert "target_is_contained_fire" in critique.failing_checks
        assert "competition_embraced_with_thesis" in critique.failing_checks
        assert "minimum_evidence_sources" in critique.failing_checks
        assert critique.revision_feedback
        print("  PASS")


def test_malformed_response_returns_empty_dict() -> None:
    """Missing required fields (target_agent, revision_feedback) should be caught."""
    with patch("src.agents.critic.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = _make_malformed_critic_response()
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        state = _make_minimal_state()
        result = run_critic(state)

        assert "critique" not in result, "Should not return malformed critique"
        assert result["current_stage"] == PipelineStage.CRITIQUING
        print("  PASS")


def test_list_revision_feedback_coerced_to_string() -> None:
    """LLM sometimes returns revision_feedback as a list of strings."""
    with patch("src.agents.critic.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        payload = json.loads(_make_wellformed_critic_response())
        payload["revision_feedback"] = [
            "Target user is too broad.",
            "Name a specific community.",
        ]
        fake_response.content = json.dumps(payload)
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        state = _make_minimal_state()
        result = run_critic(state)

        assert "critique" in result
        critique: Critique = result["critique"]
        assert isinstance(critique.revision_feedback, str), f"Expected str, got {type(critique.revision_feedback)}"
        assert "Target user is too broad." in critique.revision_feedback
        assert "Name a specific community." in critique.revision_feedback
        print("  PASS")


def test_wrapped_critique_object_unwrapped() -> None:
    """If LLM returns {"critique": {...}}, extract inner object."""
    with patch("src.agents.critic.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = json.dumps(
            {"critique": json.loads(_make_wellformed_critic_response())}
        )
        fake_llm.invoke.return_value = fake_response
        mock_get_llm.return_value = fake_llm

        state = _make_minimal_state()
        result = run_critic(state)

        assert "critique" in result
        assert result["critique"].approval_status == "revise"
        assert result["critique"].target_agent == "idea_generator"
        print("  PASS")


def test_live_llm_produces_valid_critique() -> None:
    """Uses real API call. Requires OPENAI_API_KEY in environment."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP (no OPENAI_API_KEY)")
        return

    state = _make_minimal_state()
    result = run_critic(state)

    assert "critique" in result, f"Expected critique, got keys={list(result.keys())}"
    critique: Critique = result["critique"]
    assert isinstance(critique, Critique)
    assert critique.idea_id == state.pitch_briefs[0].idea_id
    assert critique.approval_status in ("approved", "revise")
    assert critique.target_agent in ("pain_point_miner", "idea_generator", "pitch_writer")
    assert isinstance(critique.rubric, CritiqueRubric)
    assert critique.revision_feedback
    logger.info(
        "Live critic: all_pass=%s status=%s target=%s checks=%s",
        critique.all_pass,
        critique.approval_status,
        critique.target_agent,
        critique.failing_checks,
    )
    print("  PASS")


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

_TESTS = [
    ("Auto-approve at max revisions", test_auto_approve_at_max_revisions),
    ("Empty pitch briefs", test_empty_pitch_briefs_returns_no_critique),
    ("Mocked well-formed LLM response", test_wellformed_llm_response_parsed_correctly),
    ("Mocked malformed LLM response", test_malformed_response_returns_empty_dict),
    ("List revision_feedback coercion", test_list_revision_feedback_coerced_to_string),
    ("Wrapped critique object unwrapped", test_wrapped_critique_object_unwrapped),
    ("Live LLM (slow)", test_live_llm_produces_valid_critique),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Critic Component Tests")
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
