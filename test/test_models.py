"""
Unit tests for domain models and rubrics.
"""

from uuid import uuid4
import pytest
from pydantic import ValidationError

from src.models import (
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
    PainPointEvidence,
    PainPointRubric,
    PitchBrief,
    ScoredIdea,
    ValidationPlan,
    Verdict,
)


def test_pain_point_model_and_rubric():
    evidence = [
        PainPointEvidence(
            source_url="https://news.ycombinator.com/item?id=123",
            raw_quote="Debugging flaky async tests in CI takes 4 hours every week.",
            source=DataSource.HACKERNEWS,
        )
    ]
    rubric = PainPointRubric(
        is_genuine_current_frustration=True,
        has_verbatim_quote=True,
        user_segment_specific=True,
    )
    assert rubric.all_pass is True

    pp = PainPoint(
        title="Flaky async CI test debugging",
        description="Developers waste half a day triaging non-deterministic test failures in CI.",
        rubric=rubric,
        passes_rubric=True,
        evidence=evidence,
    )

    assert pp.source_url == "https://news.ycombinator.com/item?id=123"
    assert pp.evidence_count == 1
    assert pp.source == DataSource.HACKERNEWS


def test_scored_idea_verdict_derivation():
    # Yes count = 7, no fatal flaws -> Pursue
    scored_pursue = ScoredIdea(
        idea_id=uuid4(),
        reasoning_trace="Strong demand and feasibility",
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
            has_path_out_of_niche=False,
        ),
        core_assumption="CI developers will install our deterministic runner",
        fatal_flaws=[],
        yes_count=7,
        verdict="pursue",
        one_risk="High competition in CI space",
    )
    assert scored_pursue.verdict == "pursue"

    # Fatal flaw forces park even with high yes count
    scored_park = ScoredIdea(
        idea_id=uuid4(),
        reasoning_trace="Fatal dependency flaw",
        feasibility_rubric=FeasibilityRubric(
            can_be_solved_manually_first=True,
            has_schlep_or_unsexy_advantage=True,
            can_2_3_team_build_mvp_in_6_months=True,
        ) if False else FeasibilityRubric(
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
        core_assumption="Platform allows third party runners",
        fatal_flaws=[FatalFlaw(flaw="Apple banned this category", severity="fatal")],
        yes_count=7,
        verdict="pursue",  # Will be overridden by validator
        one_risk="Platform ban",
    )
    assert scored_park.verdict == "park"


def test_pitch_brief_tagline_validation():
    landscape = CompetitiveLandscape(
        current_behavior="Engineers spend hours reading log files manually",
        direct_competitors="Datadog, Sentry",
        real_enemy="Grep and log print statements",
    )
    validation = ValidationPlan(
        discovery_questions=[
            "How do you currently triage test failures?",
            "What was the last bug that blocked your deployment?",
            "Who on the team handles flakiness?",
            "How much CI spend goes to retries?",
            "What happens when a build fails?",
        ],
        validation_criteria="5 engineering teams adopt within 2 weeks",
    )

    # Valid tagline (< 12 words)
    brief = PitchBrief(
        idea_id=uuid4(),
        title="CI Flakiness Buster",
        tagline="Deterministic test execution for modern engineering teams",
        problem="Flaky tests cost engineering teams millions of wasted hours every year.",
        solution="Isolated sandbox container reruns that capture deterministic execution traces.",
        target_user="DevOps leads and staff software engineers",
        market_opportunity="Developer tooling market growing at 20% CAGR.",
        competitive_landscape=landscape,
        differentiation="Zero code change drop-in runner",
        validation_plan=validation,
        business_model="Seat-based SaaS with compute tiering",
        go_to_market="Bottom-up GitHub marketplace adoption",
        key_risk="Cloud compute costs",
        next_steps="Build MVP GitHub Action",
        evidence_links=["https://news.ycombinator.com/item?id=123"],
        markdown_content="# CI Flakiness Buster\n\nFull pitch deck markdown here. This product solves flaky tests with deterministic execution tracing and sandboxed reruns.",
    )
    assert brief.title == "CI Flakiness Buster"

    # Invalid tagline (> 12 words)
    with pytest.raises(ValidationError):
        PitchBrief(
            idea_id=uuid4(),
            title="CI Flakiness Buster",
            tagline="This is a very long tagline that definitely exceeds the twelve word limit set by the prompt and rubric",
            problem="Problem description...",
            solution="Solution description...",
            target_user="Engineers",
            market_opportunity="Huge opportunity...",
            competitive_landscape=landscape,
            differentiation="Differentiation...",
            validation_plan=validation,
            business_model="SaaS...",
            go_to_market="Direct sales...",
            key_risk="Risk...",
            next_steps="Next steps...",
            evidence_links=["https://news.ycombinator.com/item?id=123"],
            markdown_content="# Content...",
        )
