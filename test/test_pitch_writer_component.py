"""Component-level test for the Pitch Writer agent.

Tests parsing edge cases and validation paths without hitting the real LLM.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.pitch_writer import run as run_pitch_writer
from src.state.schema import (
    CompetitiveLandscape,
    DataSource,
    DemandRubric,
    FeasibilityRubric,
    Idea,
    NoveltyRubric,
    PainPoint,
    PitchBrief,
    ScoredIdea,
    ValidationPlan,
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


def _make_pitch_brief_obj(idea_id: Any) -> PitchBrief:
    """Return a single valid PitchBrief instance."""
    return PitchBrief(
        idea_id=idea_id,
        title="Docker Compose Simplifier",
        tagline="Easily manage and debug Compose.",
        problem="Developers struggle with complex multi-service local development setups.",
        solution="A visual editor and debugger for Docker Compose files.",
        target_user="Solo developers and small teams",
        market_opportunity="Large developer tools market with growing Docker adoption.",
        competitive_landscape=CompetitiveLandscape(
            current_behavior="Developers manually edit YAML files and debug via trial-and-error restarts",
            direct_competitors="Docker Desktop, VS Code extensions, and manual YAML editing",
            real_enemy="The habit of editing raw YAML without validation or visual feedback",
        ),
        differentiation="Visual editor with real-time validation vs manual YAML editing",
        validation_plan=ValidationPlan(
            discovery_questions=[
                "Walk me through the last time you debugged a Docker Compose issue",
                "How much time do you spend on Docker Compose configuration weekly?",
                "What frustrates you most about your current workflow?",
                "What would make you switch from your current approach?",
                "How do you currently validate your Docker Compose files?",
            ],
            validation_criteria="At least 7 out of 10 developers mention spending 2+ hours/week on Docker Compose debugging",
        ),
        business_model="Subscription SaaS pricing with freemium tier.",
        go_to_market="Direct outreach to r/docker power users and open-source contributors.",
        key_risk="Incumbents like Docker Desktop may build native visual tooling.",
        next_steps="Build an interactive landing page and prototype MVP.",
        evidence_links=["https://reddit.com/r/docker/comments/abc123"],
        markdown_content=(
            "# Docker Compose Simplifier\n\n"
            "## Problem\n"
            "Developers struggle with complex multi-service local development setups and spend hours debugging.\n\n"
            "## Solution\n"
            "A visual editor and debugger for Docker Compose files with live validation and auto-fixing.\n"
        ),
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_no_scored_ideas_returns_empty() -> None:
    """If state has no scored ideas, return empty pitch_briefs."""
    state = VentureForgeState(domain="test", scored_ideas=[])
    result = run_pitch_writer(state)
    assert result["pitch_briefs"] == []
    assert "events" in result


def test_wellformed_response_produces_briefs() -> None:
    """Happy path: one scored idea → one pitch brief."""
    state = _make_minimal_state()
    idea = state.ideas[0]

    with patch("src.agents.pitch_writer.get_structured_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_pitch_brief_obj(idea.id)
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 1, f"Expected 1 brief, got {len(briefs)}"
    assert briefs[0].idea_id == idea.id
    assert briefs[0].title == "Docker Compose Simplifier"
    assert briefs[0].revision_count == 0


def test_missing_or_failed_llm_call_skips_brief() -> None:
    """LLM failure skips brief gracefully."""
    state = _make_minimal_state()

    with patch("src.agents.pitch_writer.get_structured_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = Exception("LLM call failed")
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 0


def test_revision_count_increments() -> None:
    """If state.revision_counts[idea_id] = 1, brief.revision_count should be 1."""
    state = _make_minimal_state()
    idea = state.ideas[0]
    state = state.model_copy(update={"revision_counts": {str(idea.id): 1}})

    with patch("src.agents.pitch_writer.get_structured_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_pitch_brief_obj(idea.id)
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 1
    assert briefs[0].revision_count == 1


def test_tagline_auto_trimming() -> None:
    """Test that taglines longer than 12 words are auto-trimmed cleanly."""
    long_tagline = "This is a very long and detailed tagline that has far more than twelve words in it"
    brief = PitchBrief(
        idea_id=uuid4(),
        title="Test App",
        tagline=long_tagline,
        problem="This is a sufficiently long problem description for testing purposes.",
        solution="This is a sufficiently long solution description for testing purposes.",
        target_user="Target users",
        market_opportunity="A very large market opportunity that passes validation.",
        competitive_landscape=CompetitiveLandscape(
            current_behavior="Developers manually manage configurations using raw YAML files.",
            direct_competitors="Existing legacy tools and manual bash scripts.",
            real_enemy="Habitual inertia and lack of real-time diagnostics.",
        ),
        differentiation="Much better and faster than existing alternatives with real-time feedback.",
        validation_plan=ValidationPlan(
            discovery_questions=["Question one for discovery?", "Question two for discovery?", "Question three for discovery?", "Question four for discovery?", "Question five for discovery?"],
            validation_criteria="Clear validation criteria that proves the customer pain point is acute.",
        ),
        business_model="SaaS subscription model.",
        go_to_market="Direct developer outreach.",
        key_risk="Adoption friction.",
        next_steps="Interview 5 users.",
        evidence_links=["https://example.com/1", "https://example.com/2"],
        markdown_content="# Test App\n\nFull pitch markdown content that is guaranteed to be well over one hundred characters in length to satisfy schema validation constraints.",
    )
    assert len(brief.tagline.split()) <= 12
    assert brief.tagline == "This is a very long and detailed tagline that has far more"


def test_best_effort_graduation_when_all_ideas_parked() -> None:
    """When top_scored_ideas is empty but scored_ideas exists (all parked), pitch_writer picks highest score."""
    state = _make_minimal_state()
    idea = state.ideas[0]
    parked_scored = state.scored_ideas[0].model_copy(update={"verdict": Verdict.PARK, "yes_count": 4})
    state = state.model_copy(update={"scored_ideas": [parked_scored]})

    assert len(state.top_scored_ideas) == 0

    with patch("src.agents.pitch_writer.get_structured_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = _make_pitch_brief_obj(idea.id)
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    briefs: list[PitchBrief] = result["pitch_briefs"]
    assert len(briefs) == 1
    assert briefs[0].idea_id == idea.id
