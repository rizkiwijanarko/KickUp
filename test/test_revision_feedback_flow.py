"""Tests for how Critic revision_feedback flows into Idea Generator and Pitch Writer.

These are fast, offline component-level tests that mock the LLMs to
focus purely on how revision_feedback is injected into prompts and how
agents respond.

Run with:
    uv run test_revision_feedback_flow.py
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.critic import run as run_critic
from src.agents.idea_generator import run as run_idea_generator
from src.agents.pitch_writer import run as run_pitch_writer
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
    PipelineStage,
    PitchBrief,
    ScoredIdea,
    ValidationPlan,
    VentureForgeState,
)
from test.test_helpers import make_test_pain_point


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_base_state() -> VentureForgeState:
    """Build a small end-to-end-ready state with one idea and brief.

    This mirrors the minimal state used in other component tests.
    """
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
        fatal_flaws=[FatalFlaw(flaw="Incumbents may copy quickly.", severity="major")],
        yes_count=8,
        verdict="pursue",
        one_risk="Incumbents may copy quickly.",
        rank=1,
    )

    brief = PitchBrief(
        idea_id=idea.id,
        title=idea.title,
        tagline=idea.one_liner,
        problem=idea.problem,
        solution=idea.solution,
        target_user=idea.target_user,
        market_opportunity="Large developer tools market.",
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
        business_model="SaaS subscription model.",
        go_to_market="Post on Reddit and Hacker News for early adopters.",
        key_risk=scored.one_risk,
        next_steps="Build MVP and recruit beta users from Reddit.",
        evidence_links=[pp1.source_url],
        markdown_content=(
            "# Docker Compose Simplifier\\n\\n"
            "## Problem\\n\\nDevelopers struggle with complex multi-service local development setups, "
            "leading to wasted time and brittle environments.\\n\\n"
            "## Solution\\n\\nA visual editor and debugger for Docker Compose files that helps developers "
            "quickly understand, modify, and validate their local dev setups."
        ),
    )

    return VentureForgeState(
        domain="developer tools",
        max_pain_points=10,
        ideas_per_run=3,
        top_n_pitches=2,
        max_revisions=2,
        pain_points=[pp1, pp2],
        ideas=[idea],
        scored_ideas=[scored],
        pitch_briefs=[brief],
    )


# ---------------------------------------------------------------------------
# Tests: Critic → Idea Generator
# ---------------------------------------------------------------------------


def test_revision_feedback_influences_idea_generator_prompt() -> None:
    """Critic feedback for positioning should be injected into idea_generator prompt.

    We patch get_llm in idea_generator to capture the prompt text and
    assert that the revision instructions appear when revision_feedback
    is set.
    """
    state = _make_base_state()

    # Fake a Critique that routes to idea_generator with specific feedback
    rubric = CritiqueRubric(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=False,
        competition_embraced_with_thesis=False,
        minimum_evidence_sources=True,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    critique = Critique(
        idea_id=state.ideas[0].id,
        reasoning_trace="Target user is a demographic, not a contained fire.",
        rubric=rubric,
        all_pass=False,
        approval_status="revise",
        failing_checks=[],  # will be recomputed
        target_agent="idea_generator",
        revision_feedback="Target user is too broad; define a specific community (e.g., Docker power users on r/docker).",
    )

    # Simulate that orchestrator has set revision_feedback on state
    state = state.model_copy(update={"revision_feedback": critique.revision_feedback})

    captured_prompt: dict[str, str] = {"text": ""}

    def _fake_invoke(messages):  # type: ignore[no-untyped-def]
        # messages[1] should be HumanMessage with content including revision block
        captured_prompt["text"] = messages[1].content
        # Return a minimal valid ideas payload so run() can complete
        payload = {
            "ideas": [
                {
                    "title": "Docker Power User Tool",
                    "one_liner": "Tool for Docker power users.",
                    "problem": "Long enough problem description for schema.",
                    "solution": "Long enough solution description for schema.",
                    "target_user": "Docker power users on r/docker",
                    "key_features": ["A", "B", "C"],
                    "addresses_pain_point_ids": [str(p.id) for p in state.pain_points],
                }
            ]
        }
        fake_resp = MagicMock()
        fake_resp.content = json.dumps(payload)
        return fake_resp

    with patch("src.agents.idea_generator.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = _fake_invoke
        mock_get_llm.return_value = fake_llm

        result = run_idea_generator(state)

    # Ensure the run completed and produced at least one idea
    assert result["ideas"], "Expected ideas from idea_generator"

    prompt_text = captured_prompt["text"]
    # Check that our revision feedback and positioning instructions were injected
    assert "THIS IS A REVISION ROUND" in prompt_text
    assert "Target user is too broad" in prompt_text
    assert "contained fire" in prompt_text
    print("  PASS")


# ---------------------------------------------------------------------------
# Tests: Critic → Pitch Writer
# ---------------------------------------------------------------------------


def test_revision_feedback_influences_pitch_writer_prompt() -> None:
    """Critic feedback for pitch writing should be injected into pitch_writer prompt."""
    state = _make_base_state()

    rubric = CritiqueRubric(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=False,  # too long tagline
        target_is_contained_fire=True,
        competition_embraced_with_thesis=True,
        minimum_evidence_sources=False,  # insufficient evidence
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    critique = Critique(
        idea_id=state.ideas[0].id,
        reasoning_trace="Tagline is too long and GTM is generic broadcast.",
        rubric=rubric,
        all_pass=False,
        approval_status="revise",
        failing_checks=[],  # will be recomputed
        target_agent="pitch_writer",
        revision_feedback=(
            "Shorten the tagline to under 12 words and rewrite go_to_market "
            "to describe a manual, unscalable acquisition strategy targeting a named community."
        ),
    )

    # Pretend we've already recorded this critique on state
    state = state.model_copy(update={"revision_feedback": critique.revision_feedback, "critiques": [critique]})

    captured_prompt: dict[str, str] = {"text": ""}

    def _fake_invoke(messages):  # type: ignore[no-untyped-def]
        captured_prompt["text"] = messages[1].content
        # Return a minimal valid brief so run() can complete
        idea = state.ideas[0]
        payload = [
            {
                "idea_id": str(idea.id),
                "title": idea.title,
                "tagline": "Docker compose debugger.",
                "problem": idea.problem,
                "solution": idea.solution,
                "target_user": idea.target_user,
                "market_opportunity": "Large developer tools market.",
                "business_model": "SaaS subscription.",
                "go_to_market": "Manually DM top 50 posters in r/docker and onboard them one by one.",
                "key_risk": "Incumbents may copy.",
                "next_steps": "Ship MVP and onboard first 10 users from r/docker.",
                "evidence_links": [state.pain_points[0].source_url],
                "markdown_content": "# Docker Compose Simplifier\n\nFull brief...",
            }
        ]
        fake_resp = MagicMock()
        fake_resp.content = json.dumps(payload)
        return fake_resp

    with patch("src.agents.pitch_writer.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = _fake_invoke
        mock_get_llm.return_value = fake_llm

        result = run_pitch_writer(state)

    # Even if parsing fails, we still care about the prompt text
    prompt_text = captured_prompt["text"]
    assert "THIS IS A REVISION ROUND for the pitch briefs" in prompt_text
    assert "Critic failing checks" in prompt_text
    assert "Do NOT change the underlying idea" in prompt_text
    print("  PASS")

    prompt_text = captured_prompt["text"]
    assert "THIS IS A REVISION ROUND for the pitch briefs" in prompt_text
    assert "Critic failing checks" in prompt_text
    assert "tagline_under_12_words" in prompt_text or "failing checks" in prompt_text
    assert "Do NOT change the underlying idea" in prompt_text
    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    ("Revision feedback flows into idea_generator prompt", test_revision_feedback_influences_idea_generator_prompt),
    ("Revision feedback flows into pitch_writer prompt", test_revision_feedback_influences_pitch_writer_prompt),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Revision Feedback Flow Tests")
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
