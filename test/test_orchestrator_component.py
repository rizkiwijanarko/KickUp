"""Component-level test for the Orchestrator (routing + reflection loop).

Fast, deterministic, offline (no LLM).
Run with:
    uv run test_orchestrator_component.py
"""
from __future__ import annotations

import json
import logging
from uuid import uuid4

from src.agents.orchestrator import orchestrator
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


def _make_full_state_with_critique(*, all_pass: bool, max_revisions: int = 2, revision_count: int = 0) -> VentureForgeState:
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
        reasoning_trace="Strong demand signal.",
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
        core_assumption="Developers will adopt this.",
        fatal_flaws=[FatalFlaw(flaw="Incumbents may copy quickly.", severity="major")],
        yes_count=8,
        verdict="pursue",
        one_risk="Incumbents may copy quickly.",
        rank=1,
    )
    brief = PitchBrief(
        idea_id=idea.id,
        title=idea.title,
        tagline="Easily manage Compose files.",
        problem=idea.problem,
        solution=idea.solution,
        target_user=idea.target_user,
        market_opportunity="Large dev tools market with growing Docker adoption.",
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
        business_model="Monthly subscription with freemium tier included.",
        go_to_market="Direct outreach to r/docker power users and small teams.",
        key_risk="Incumbents may copy quickly.",
        next_steps="Build MVP and recruit beta users.",
        evidence_links=[pp1.source_url],
        markdown_content=(
            "# Pitch\\n\\nThis markdown content is intentionally long enough to pass schema.\\n"
            "It should be well over one hundred characters to satisfy validation.\\n"
            "Problem, solution, market, and GTM details live here.\\n"
        ),
        revision_count=revision_count,
    )
    rubric = CritiqueRubric(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=all_pass,  # Fail this if all_pass=False
        competition_embraced_with_thesis=True,
        minimum_evidence_sources=True,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    critique = Critique(
        idea_id=idea.id,
        reasoning_trace="ok" if all_pass else "Target user is too broad - not a contained fire.",
        rubric=rubric,
        all_pass=all_pass,
        approval_status="approved" if all_pass else "revise",
        target_agent="idea_generator",  # Positioning issues route to idea_generator
        revision_feedback=(
            "Target user 'Solo developers and small teams' is too broad. "
            "Replace with a specific reachable community (e.g., 'React Native developers using Docker')."
            if not all_pass
            else "All good — approved as-is."
        ),
    )
    state = VentureForgeState(
        domain="developer tools",
        max_revisions=max_revisions,
        pain_points=[pp1, pp2],
        ideas=[idea],
        scored_ideas=[scored],
        pitch_briefs=[brief],
        critique=critique,
        revision_counts={str(idea.id): revision_count} if revision_count else {},
        current_stage=PipelineStage.CRITIQUING,
    )
    return state


def test_routes_to_mining_when_no_pain_points() -> None:
    state = VentureForgeState(domain="test", pain_points=[])
    patch = orchestrator(state)
    assert patch["current_stage"] == PipelineStage.MINING
    assert patch["next_node"] == "pain_point_miner"
    print("  PASS")


def test_routes_to_generator_when_no_ideas() -> None:
    pp1 = make_test_pain_point(
        title="Some pain",
        description="A sufficiently long description for schema validation.",
        source_url="https://reddit.com/r/x/comments/1",
        raw_quote="This is a real quote that is long enough.",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="Another pain",
        description="Another sufficiently long description for schema validation.",
        source_url="https://reddit.com/r/x/comments/2",
        raw_quote="This is another real quote that is long enough.",
        source=DataSource.REDDIT,
    )
    # Need at least 2 pain points to pass the MIN_PAIN_POINTS_FOR_IDEAS quality gate
    state = VentureForgeState(domain="test", pain_points=[pp1, pp2], ideas=[])
    patch = orchestrator(state)
    assert patch["current_stage"] == PipelineStage.GENERATING
    assert patch["next_node"] == "idea_generator"
    print("  PASS")


def test_reflection_loop_bumps_revision_and_resets_downstream() -> None:
    state = _make_full_state_with_critique(all_pass=False, max_revisions=2, revision_count=0)
    idea_id = str(state.critique.idea_id) if state.critique else ""
    patch = orchestrator(state)

    assert patch["current_stage"] == PipelineStage.REVISING
    assert patch["next_node"] == "idea_generator"  # Positioning issues route to idea_generator
    assert patch["revision_counts"][idea_id] == 1
    # For idea_generator revisions, only the specific idea is cleared (Issue #1 fix)
    assert len(patch["ideas"]) == 0 or all(str(i.id) != idea_id for i in patch.get("ideas", []))
    assert patch["critique"] is None
    assert patch["revision_feedback"]
    print("  PASS")


def test_marks_completed_when_critique_passes() -> None:
    state = _make_full_state_with_critique(all_pass=True, max_revisions=2, revision_count=0)
    patch = orchestrator(state)
    assert patch["current_stage"] == PipelineStage.COMPLETED
    assert patch["next_node"] == "__end__"
    assert patch["completed_at"] is not None
    print("  PASS")


def test_marks_completed_when_cannot_revise() -> None:
    state = _make_full_state_with_critique(all_pass=False, max_revisions=1, revision_count=1)
    patch = orchestrator(state)
    assert patch["current_stage"] == PipelineStage.COMPLETED
    assert patch["next_node"] == "__end__"
    print("  PASS")


_TESTS = [
    ("Routes to mining when no pain points", test_routes_to_mining_when_no_pain_points),
    ("Routes to generator when no ideas", test_routes_to_generator_when_no_ideas),
    ("Reflection loop bumps revision + resets downstream", test_reflection_loop_bumps_revision_and_resets_downstream),
    ("Marks completed when critique passes", test_marks_completed_when_critique_passes),
    ("Marks completed when cannot revise", test_marks_completed_when_cannot_revise),
]


if __name__ == "__main__":
    print("=" * 60)
    print("Orchestrator Component Tests")
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

