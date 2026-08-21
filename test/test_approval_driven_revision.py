"""
Tests for Approval-Driven Critique Dispatch, In-Place Revision Refinement, and SQLite Evidence Caching.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4


from src.agents.critic import run as run_critic
from src.agents.orchestrator import orchestrator
from src.agents.pitch_writer import run as run_pitch_writer
from src.mining.cache import SQLiteEvidenceCache
from src.mining.provider import RawEvidence
from src.models.common import (
    DataSource,
    PipelineStage,
)
from src.state.schema import (
    CompetitiveLandscape,
    Critique,
    CritiqueRubric,
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


def _make_multi_pitch_state() -> VentureForgeState:
    """Create a state with 2 ideas, 2 scored ideas, and 2 pitch briefs."""
    pp1 = make_test_pain_point(
        title="Pain point 1",
        description="Description for pain point 1 that is long enough.",
        source_url="https://reddit.com/r/dev/1",
        raw_quote="Quote 1 for pain point 1",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="Pain point 2",
        description="Description for pain point 2 that is long enough.",
        source_url="https://reddit.com/r/dev/2",
        raw_quote="Quote 2 for pain point 2",
        source=DataSource.REDDIT,
    )

    idea1 = Idea(
        id=uuid4(),
        title="Idea Alpha",
        one_liner="One liner for alpha",
        problem="Problem description for alpha",
        solution="Solution description for alpha",
        target_user="Target users for alpha",
        key_features=["Feature A", "Feature B", "Feature C"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )
    idea2 = Idea(
        id=uuid4(),
        title="Idea Beta",
        one_liner="One liner for beta",
        problem="Problem description for beta",
        solution="Solution description for beta",
        target_user="Target users for beta",
        key_features=["Feature D", "Feature E", "Feature F"],
        addresses_pain_point_ids=[pp1.id, pp2.id],
    )

    scored1 = ScoredIdea(
        idea_id=idea1.id,
        reasoning_trace="Reasoning for alpha",
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
        core_assumption="Assumption alpha",
        fatal_flaws=[],
        yes_count=8,
        verdict="pursue",
        one_risk="Risk alpha",
        rank=1,
    )

    scored2 = ScoredIdea(
        idea_id=idea2.id,
        reasoning_trace="Reasoning for beta",
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
        core_assumption="Assumption beta",
        fatal_flaws=[],
        yes_count=7,
        verdict="pursue",
        one_risk="Risk beta",
        rank=2,
    )

    brief1 = PitchBrief(
        idea_id=idea1.id,
        title=idea1.title,
        tagline=idea1.one_liner,
        problem=idea1.problem,
        solution=idea1.solution,
        target_user=idea1.target_user,
        market_opportunity="A very large enterprise developer tools market with significant budget.",
        competitive_landscape=CompetitiveLandscape(
            current_behavior="Current behavior alpha",
            direct_competitors="Competitor alpha",
            real_enemy="Habit alpha",
        ),
        differentiation="Unique visual debugging workflow that eliminates YAML syntax errors.",
        validation_plan=ValidationPlan(
            discovery_questions=["Q1", "Q2", "Q3", "Q4", "Q5"],
            validation_criteria="At least 7 out of 10 developers validate this solution.",
        ),
        business_model="Business model alpha with tiered monthly subscriptions for teams.",
        go_to_market="Direct outreach to top active posters and moderators in targeted developer subreddits.",
        key_risk="Significant platform competition and fast replication risk.",
        next_steps="Ship initial MVP and recruit 10 pilot engineering teams.",
        evidence_links=[pp1.source_url],
        markdown_content="# Brief Alpha\n\nThis is a full one-page pitch brief describing problem, solution, market, and business model in comprehensive detail.",
    )

    brief2 = PitchBrief(
        idea_id=idea2.id,
        title=idea2.title,
        tagline=idea2.one_liner,
        problem=idea2.problem,
        solution=idea2.solution,
        target_user=idea2.target_user,
        market_opportunity="A very large enterprise developer tools market with significant budget.",
        competitive_landscape=CompetitiveLandscape(
            current_behavior="Current behavior beta",
            direct_competitors="Competitor beta",
            real_enemy="Habit beta",
        ),
        differentiation="Unique visual debugging workflow that eliminates YAML syntax errors.",
        validation_plan=ValidationPlan(
            discovery_questions=["Q1", "Q2", "Q3", "Q4", "Q5"],
            validation_criteria="At least 7 out of 10 developers validate this solution.",
        ),
        business_model="Business model beta with tiered monthly subscriptions for teams.",
        go_to_market="Direct outreach to top active posters and moderators in targeted developer subreddits.",
        key_risk="Significant platform competition and fast replication risk.",
        next_steps="Ship initial MVP and recruit 10 pilot engineering teams.",
        evidence_links=[pp2.source_url],
        markdown_content="# Brief Beta\n\nThis is a full one-page pitch brief describing problem, solution, market, and business model in comprehensive detail.",
    )

    return VentureForgeState(
        domain="developer tools",
        max_revisions=2,
        top_n_pitches=2,
        pain_points=[pp1, pp2],
        ideas=[idea1, idea2],
        scored_ideas=[scored1, scored2],
        pitch_briefs=[brief1, brief2],
    )


def test_pitch_writer_merges_revised_brief_without_dropping_others() -> None:
    """When pitch_writer revises one brief, it must merge back into state preserving all other briefs."""
    state = _make_multi_pitch_state()
    target_idea_id = state.ideas[1].id

    # Set revision mode for Idea Beta
    state = state.model_copy(
        update={
            "current_revision_idea_id": target_idea_id,
            "revision_feedback": "Shorten tagline for Idea Beta",
        }
    )

    revised_brief = state.pitch_briefs[1].model_copy(update={"tagline": "Short tagline"})

    with (
        patch("src.agents.pitch_writer.generate_pitch_with_retry") as mock_gen,
        patch("src.agents.pitch_writer.convert_to_pitch_brief") as mock_convert,
    ):
        mock_gen.return_value = {"title": "Idea Beta"}
        mock_convert.return_value = revised_brief

        patch_result = run_pitch_writer(state)

    result_briefs = patch_result["pitch_briefs"]
    assert len(result_briefs) == 2, f"Expected 2 briefs, got {len(result_briefs)}"
    assert result_briefs[0].idea_id == state.ideas[0].id
    assert result_briefs[1].idea_id == target_idea_id
    assert result_briefs[1].tagline == "Short tagline"


def test_approval_driven_critique_routing_skips_approved_briefs() -> None:
    """Critic review should select next pending brief and orchestrator should route accordingly."""
    state = _make_multi_pitch_state()
    idea1_id = state.ideas[0].id
    idea2_id = state.ideas[1].id

    # Simulate Idea Alpha passed critique
    critique1 = Critique(
        idea_id=idea1_id,
        reasoning_trace="Alpha is great",
        rubric=CritiqueRubric(
            all_claims_evidence_backed=True,
            no_hallucinated_source_urls=True,
            tagline_under_12_words=True,
            target_is_contained_fire=True,
            competition_embraced_with_thesis=True,
            minimum_evidence_sources=True,
            scorer_verdict_justified=True,
            validation_plan_complete=True,
        ),
        all_pass=True,
        approval_status="approved",
        target_agent="pitch_writer",
        revision_feedback="All checks passed successfully.",
    )

    # Record critique1 on state
    state = state.model_copy(
        update={
            "critique": critique1,
            "critiques": [critique1],
        }
    )

    # Orchestrator should see Idea Alpha is approved and route to Critic for Idea Beta
    patch_orch = orchestrator(state)
    assert patch_orch["current_stage"] == PipelineStage.CRITIQUING
    assert patch_orch["next_node"] == "critic"

    # When Critic runs on this updated state, it should pick Idea Beta
    state_for_critic = state.model_copy(update=patch_orch)
    with patch("src.agents.critic._invoke_llm") as mock_invoke:
        mock_invoke.return_value = Critique(
            idea_id=idea2_id,
            reasoning_trace="Beta is also great",
            rubric=CritiqueRubric(
                all_claims_evidence_backed=True,
                no_hallucinated_source_urls=True,
                tagline_under_12_words=True,
                target_is_contained_fire=True,
                competition_embraced_with_thesis=True,
                minimum_evidence_sources=True,
                scorer_verdict_justified=True,
                validation_plan_complete=True,
            ),
            all_pass=True,
            approval_status="approved",
            target_agent="pitch_writer",
            revision_feedback="All checks passed successfully.",
        )

        critic_patch = run_critic(state_for_critic)

    assert critic_patch["critique"].idea_id == idea2_id

    # Now both ideas are approved; orchestrator should mark pipeline completed
    state_all_approved = state_for_critic.model_copy(
        update={
            "critique": critic_patch["critique"],
            "critiques": [critique1, critic_patch["critique"]],
        }
    )
    final_orch_patch = orchestrator(state_all_approved)
    assert final_orch_patch["current_stage"] == PipelineStage.COMPLETED
    assert final_orch_patch["next_node"] == "__end__"


def test_sqlite_evidence_cache(tmp_path) -> None:
    """Test SQLite evidence cache insertion, retrieval, and TTL."""
    db_path = str(tmp_path / "test_cache.db")
    cache = SQLiteEvidenceCache(db_path=db_path)

    items = [
        RawEvidence(
            text="Evidence quote about dev tools",
            url="https://news.ycombinator.com/item?id=12345",
            source=DataSource.HACKERNEWS,
            title="Dev tools discussion",
            author="tester",
            score=50,
        )
    ]

    # Cache should initially be empty
    assert cache.get("developer tools") is None

    # Store in cache
    cache.set("developer tools", items)

    # Retrieve from cache
    cached_items = cache.get("developer tools")
    assert cached_items is not None
    assert len(cached_items) == 1
    assert cached_items[0].url == "https://news.ycombinator.com/item?id=12345"
    assert cached_items[0].source == DataSource.HACKERNEWS

    # Expired cache test
    assert cache.get("developer tools", max_age_seconds=-1) is None

    # Clear cache test
    cache.clear("developer tools")
    assert cache.get("developer tools") is None
