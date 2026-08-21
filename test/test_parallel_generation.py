"""
Unit and Component tests for Concurrent Idea & Pitch Generation.
"""

from __future__ import annotations

import time
from unittest.mock import patch
from uuid import uuid4


from src.agents.idea_generator import run as run_idea_generator
from src.agents.pitch_writer import run as run_pitch_writer
from src.models.common import DataSource
from src.state.schema import (
    CompetitiveLandscape,
    DemandRubric,
    FeasibilityRubric,
    Idea,
    NoveltyRubric,
    PitchBrief,
    ScoredIdea,
    ValidationPlan,
    VentureForgeState,
)
from test.test_helpers import make_test_pain_point


def test_idea_generator_parallel_execution() -> None:
    """Test that idea_generator generates 3 ideas concurrently across threads."""
    pp1 = make_test_pain_point(
        title="Docker builds are slow",
        description="Docker build caching is poorly configured for multi-stage builds.",
        source_url="https://reddit.com/r/dev/1",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="CI pipelines fail intermittently",
        description="Flaky integration tests cause builds to fail unpredictably.",
        source_url="https://reddit.com/r/dev/2",
        source=DataSource.REDDIT,
    )

    state = VentureForgeState(
        domain="developer tools",
        ideas_per_run=3,
        pain_points=[pp1, pp2],
        filtered_pain_points=[pp1, pp2],
    )

    call_times = []

    def mock_invoke_llm(state, idea_number, total_ideas, theme_angle=None, retry_count=0):
        call_times.append(time.monotonic())
        # Sleep slightly to verify concurrent overlap
        time.sleep(0.05)
        return {
            "title": f"Idea {idea_number}",
            "one_liner": f"One liner for idea {idea_number}",
            "problem": "Problem description with enough detail for testing.",
            "solution": "Solution description with enough detail for testing.",
            "target_user": "Senior Platform Engineers",
            "key_features": ["Feature 1", "Feature 2", "Feature 3"],
            "addresses_pain_point_ids": [str(pp1.id), str(pp2.id)],
        }

    with patch("src.agents.idea_generator.invoke_llm_single", side_effect=mock_invoke_llm):
        start_time = time.monotonic()
        patch_result = run_idea_generator(state)
        total_duration = time.monotonic() - start_time

    ideas = patch_result["ideas"]
    assert len(ideas) == 3
    # If ran sequentially, 3 * 0.05s = ~0.15s. In parallel, it finishes in < 0.14s.
    assert total_duration < 0.14
    assert patch_result["next_node"] == "orchestrator"


def test_idea_generator_partial_failure_resilience() -> None:
    """Test that if 1 parallel worker fails, the valid ideas from other workers are retained."""
    pp1 = make_test_pain_point(
        title="Docker builds are slow",
        description="Docker build caching is poorly configured for multi-stage builds.",
        source_url="https://reddit.com/r/dev/1",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="CI pipelines fail intermittently",
        description="Flaky integration tests cause builds to fail unpredictably.",
        source_url="https://reddit.com/r/dev/2",
        source=DataSource.REDDIT,
    )

    state = VentureForgeState(
        domain="developer tools",
        ideas_per_run=3,
        pain_points=[pp1, pp2],
        filtered_pain_points=[pp1, pp2],
    )

    def mock_invoke_llm(state, idea_number, total_ideas, theme_angle=None, retry_count=0):
        if idea_number == 2:
            return None  # Worker 2 fails
        return {
            "title": f"Idea {idea_number}",
            "one_liner": f"One liner for idea {idea_number}",
            "problem": "Problem description with enough detail for testing.",
            "solution": "Solution description with enough detail for testing.",
            "target_user": "Senior Platform Engineers",
            "key_features": ["Feature 1", "Feature 2", "Feature 3"],
            "addresses_pain_point_ids": [str(pp1.id), str(pp2.id)],
        }

    with patch("src.agents.idea_generator.invoke_llm_single", side_effect=mock_invoke_llm):
        patch_result = run_idea_generator(state)

    ideas = patch_result["ideas"]
    assert len(ideas) == 2  # 2 of 3 succeeded
    assert {i.title for i in ideas} == {"Idea 1", "Idea 3"}


def test_pitch_writer_parallel_execution() -> None:
    """Test that pitch_writer generates pitch briefs concurrently across target ideas."""
    pp1 = make_test_pain_point(
        title="Docker builds are slow",
        description="Docker build caching is poorly configured for multi-stage builds.",
        source_url="https://reddit.com/r/dev/1",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="CI pipelines fail intermittently",
        description="Flaky integration tests cause builds to fail unpredictably.",
        source_url="https://reddit.com/r/dev/2",
        source=DataSource.REDDIT,
    )

    idea1 = Idea(
        id=uuid4(),
        title="Idea Alpha",
        one_liner="One liner alpha",
        problem="Problem alpha with sufficient detail for schema validation.",
        solution="Solution alpha with sufficient detail for schema validation.",
        target_user="Target platform engineers",
        key_features=["Feature 1", "Feature 2", "Feature 3"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )
    idea2 = Idea(
        id=uuid4(),
        title="Idea Beta",
        one_liner="One liner beta",
        problem="Problem beta with sufficient detail for schema validation.",
        solution="Solution beta with sufficient detail for schema validation.",
        target_user="Target DevOps engineers",
        key_features=["Feature 4", "Feature 5", "Feature 6"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )

    scored1 = ScoredIdea(
        idea_id=idea1.id,
        reasoning_trace="Trace 1",
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
        core_assumption="Assumption 1",
        fatal_flaws=[],
        yes_count=8,
        verdict="pursue",
        one_risk="Risk 1",
        rank=1,
    )
    scored2 = ScoredIdea(
        idea_id=idea2.id,
        reasoning_trace="Trace 2",
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
        core_assumption="Assumption 2",
        fatal_flaws=[],
        yes_count=7,
        verdict="pursue",
        one_risk="Risk 2",
        rank=2,
    )

    state = VentureForgeState(
        domain="developer tools",
        top_n_pitches=2,
        pain_points=[pp1],
        ideas=[idea1, idea2],
        scored_ideas=[scored1, scored2],
    )

    def mock_gen_pitch(state, scored_idea):
        time.sleep(0.05)
        return {"title": f"Pitch for {scored_idea.idea_id}"}

    def mock_convert_pitch(pitch_dict, idea_id, state):
        return PitchBrief(
            idea_id=idea_id,
            title=f"Brief for {idea_id}",
            tagline="A concise tagline for brief",
            problem="Problem with sufficient detail",
            solution="Solution with sufficient detail",
            target_user="Target users in tech",
            market_opportunity="A very large enterprise developer tools market with significant budget.",
            competitive_landscape=CompetitiveLandscape(
                current_behavior="Current behavior of writing custom shell scripts and manual configs.",
                direct_competitors="Direct competitor",
                real_enemy="Habitual inertia",
            ),
            differentiation="Unique visual debugging workflow that eliminates YAML syntax errors.",
            validation_plan=ValidationPlan(
                discovery_questions=["Q1", "Q2", "Q3", "Q4", "Q5"],
                validation_criteria="At least 7 out of 10 developers validate this solution.",
            ),
            business_model="Business model with tiered monthly subscriptions for teams.",
            go_to_market="Direct outreach to top active posters and moderators in targeted developer subreddits.",
            key_risk="Significant platform competition and fast replication risk.",
            next_steps="Ship initial MVP and recruit 10 pilot engineering teams.",
            evidence_links=[pp1.source_url],
            markdown_content="# Brief\n\nThis is a full one-page pitch brief describing problem, solution, market, and business model in comprehensive detail.",
        )

    with (
        patch("src.agents.pitch_writer.generate_pitch_with_retry", side_effect=mock_gen_pitch),
        patch("src.agents.pitch_writer.convert_to_pitch_brief", side_effect=mock_convert_pitch),
    ):
        start_time = time.monotonic()
        patch_result = run_pitch_writer(state)
        total_duration = time.monotonic() - start_time

    briefs = patch_result["pitch_briefs"]
    assert len(briefs) == 2
    # Preserves rank order
    assert briefs[0].idea_id == idea1.id
    assert briefs[1].idea_id == idea2.id
    assert total_duration < 0.09  # Parallel execution completed
