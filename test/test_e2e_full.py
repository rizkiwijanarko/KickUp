"""
Comprehensive End-to-End Test for VentureForge
===============================================

This test validates the entire system from start to finish:
1. Pain Point Miner extracts real pain points
2. Idea Generator creates startup ideas
3. Scorer evaluates ideas with binary rubrics
4. Pitch Writer produces investor briefs
5. Critic reviews and triggers reflection loop if needed
6. Full LangGraph orchestration with state persistence

Requirements:
- LLM_API_KEY or OPENAI_API_KEY must be set
- TAVILY_API_KEY recommended for fallback community discovery
- Internet connection for Reddit/HN scraping

Run with:
    pytest test/test_e2e_full.py -v -s
    # or directly:
    python test/test_e2e_full.py
"""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.config import settings
from src.graph import build_graph
from src.state.schema import (
    DataSource,
    PainPoint,
    PainPointRubric,
    PipelineStage,
    Verdict,
    VentureForgeState,
)
from test.test_helpers import make_test_pain_point


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def test_domain() -> str:
    """Domain for testing - developer tools has reliable pain points."""
    return "developer tools"


@pytest.fixture
def synthetic_pain_points() -> list[PainPoint]:
    """Pre-built pain points for fast synthetic testing."""
    return [
        make_test_pain_point(
            title="Docker Compose complexity nightmare",
            description="Developers struggle with multi-service local dev setups.",
            source_url="https://reddit.com/r/programming/comments/test1",
            raw_quote="I spend more time debugging docker-compose.yml than coding",
            source=DataSource.REDDIT,
        ),
        make_test_pain_point(
            title="Local LLM management is painful",
            description="No simple tool for running local AI models.",
            source_url="https://reddit.com/r/LocalLLaMA/comments/test2",
            raw_quote="I wish there was a simple local LLM runner that just works",
            source=DataSource.REDDIT,
        ),
        make_test_pain_point(
            title="API docs always out of date",
            description="Teams can't keep OpenAPI specs in sync with code.",
            source_url="https://news.ycombinator.com/item?id=test3",
            raw_quote="Every time we ship a feature the API docs are already wrong",
            source=DataSource.HACKERNEWS,
        ),
        make_test_pain_point(
            title="CI pipeline debugging takes forever",
            description="Developers waste hours reproducing CI failures locally.",
            source_url="https://reddit.com/r/devops/comments/test4",
            raw_quote="Why does my test pass locally but fail in CI with same Dockerfile",
            source=DataSource.REDDIT,
        ),
    ]


@pytest.fixture
def initial_state(test_domain: str) -> VentureForgeState:
    """Create initial state for a new run."""
    return VentureForgeState(
        domain=test_domain,
        max_pain_points=5,  # Small for fast testing
        ideas_per_run=3,
        top_n_pitches=2,
        max_revisions=2,
    )


@pytest.fixture
def synthetic_state(test_domain: str, synthetic_pain_points: list[PainPoint]) -> VentureForgeState:
    """State pre-loaded with synthetic pain points for faster testing."""
    return VentureForgeState(
        domain=test_domain,
        max_pain_points=5,
        ideas_per_run=3,
        top_n_pitches=2,
        max_revisions=2,
        pain_points=synthetic_pain_points,
    )


# =============================================================================
# COMPONENT TESTS (Fast, no LangGraph)
# =============================================================================

class TestComponents:
    """Test individual agent components in isolation."""

    def test_pain_point_miner(self, initial_state: VentureForgeState):
        """Test Pain Point Miner extracts valid pain points."""
        from src.agents.pain_point_miner import run as run_pain_point_miner

        result = run_pain_point_miner(initial_state)
        pain_points = result["pain_points"]

        assert isinstance(pain_points, list), "Should return list of pain points"
        assert len(pain_points) > 0, "Should extract at least one pain point"

        # Validate structure
        for pp in pain_points:
            assert isinstance(pp, PainPoint), f"Expected PainPoint, got {type(pp)}"
            assert pp.source_url, "Pain point must have source URL"
            assert pp.raw_quote, "Pain point must have verbatim quote"
            assert pp.rubric, "Pain point must have rubric"
            assert pp.passes_rubric, "Only passing pain points should be returned"

        print(f"\n✓ Pain Point Miner extracted {len(pain_points)} valid pain points")

    def test_idea_generator(self, synthetic_state: VentureForgeState):
        """Test Idea Generator creates valid startup ideas."""
        from src.agents.idea_generator import run as run_idea_generator

        result = run_idea_generator(synthetic_state)
        ideas = result["ideas"]

        assert isinstance(ideas, list), "Should return list of ideas"
        assert len(ideas) > 0, "Should generate at least one idea"

        # Validate structure
        for idea in ideas:
            assert idea.title, "Idea must have title"
            assert idea.one_liner, "Idea must have one-liner"
            assert len(idea.one_liner.split()) <= 15, "One-liner should be under 15 words"
            assert idea.problem, "Idea must describe problem"
            assert idea.solution, "Idea must describe solution"
            assert idea.target_user, "Idea must identify target user"
            assert len(idea.addresses_pain_point_ids) >= 2, "Idea must address at least 2 pain points"

        print(f"\n✓ Idea Generator created {len(ideas)} valid ideas")

    def test_scorer(self, synthetic_state: VentureForgeState):
        """Test Scorer evaluates ideas with binary rubrics."""
        from src.agents.idea_generator import run as run_idea_generator
        from src.agents.scorer import run as run_scorer

        # Generate ideas first
        result = run_idea_generator(synthetic_state)
        state_with_ideas = synthetic_state.model_copy(update=result)

        # Score them
        result = run_scorer(state_with_ideas)
        scored_ideas = result["scored_ideas"]

        assert isinstance(scored_ideas, list), "Should return list of scored ideas"
        assert len(scored_ideas) > 0, "Should score at least one idea"

        # Validate scoring structure
        for si in scored_ideas:
            assert si.idea_id, "Scored idea must reference idea_id"
            assert si.reasoning_trace, "Must include reasoning trace"
            assert si.feasibility_rubric, "Must have feasibility rubric"
            assert si.demand_rubric, "Must have demand rubric"
            assert si.novelty_rubric, "Must have novelty rubric"
            assert 0 <= si.yes_count <= 8, f"yes_count must be 0-8, got {si.yes_count}"
            assert si.verdict in [Verdict.PURSUE, Verdict.EXPLORE, Verdict.PARK], \
                f"Invalid verdict: {si.verdict}"
            assert si.one_risk, "Must identify one key risk"
            assert si.core_assumption, "Must state core assumption"

            # Validate verdict logic
            has_fatal = any(f.severity == "fatal" for f in si.fatal_flaws)
            if si.yes_count >= 6 and not has_fatal:
                assert si.verdict == Verdict.PURSUE, \
                    f"6+ yes with no fatal flaws should be PURSUE, got {si.verdict}"
            elif si.yes_count <= 2 or has_fatal:
                assert si.verdict == Verdict.PARK, \
                    f"≤2 yes or fatal flaw should be PARK, got {si.verdict}"

        print(f"\n✓ Scorer evaluated {len(scored_ideas)} ideas with binary rubrics")

    def test_pitch_writer(self, synthetic_state: VentureForgeState):
        """Test Pitch Writer produces valid investor briefs."""
        from src.agents.idea_generator import run as run_idea_generator
        from src.agents.scorer import run as run_scorer
        from src.agents.pitch_writer import run as run_pitch_writer

        # Generate and score ideas
        result = run_idea_generator(synthetic_state)
        state_with_ideas = synthetic_state.model_copy(update=result)
        result = run_scorer(state_with_ideas)
        state_with_scores = state_with_ideas.model_copy(update=result)

        # Write pitches
        result = run_pitch_writer(state_with_scores)
        pitch_briefs = result["pitch_briefs"]

        assert isinstance(pitch_briefs, list), "Should return list of pitch briefs"
        assert len(pitch_briefs) > 0, "Should write at least one pitch"

        # Validate pitch structure
        for brief in pitch_briefs:
            assert brief.idea_id, "Pitch must reference idea_id"
            assert brief.title, "Pitch must have title"
            assert brief.tagline, "Pitch must have tagline"
            assert len(brief.tagline.split()) <= 12, \
                f"Tagline should be ≤12 words, got {len(brief.tagline.split())}"
            assert brief.problem, "Pitch must describe problem"
            assert brief.solution, "Pitch must describe solution"
            assert brief.target_user, "Pitch must identify target user"
            assert brief.market_opportunity, "Pitch must describe market"
            assert brief.business_model, "Pitch must describe business model"
            assert brief.go_to_market, "Pitch must describe GTM strategy"
            assert brief.key_risk, "Pitch must identify key risk"
            assert brief.next_steps, "Pitch must outline next steps"
            assert brief.evidence_links, "Pitch must include evidence links"
            assert brief.markdown_content, "Pitch must have markdown content"

        print(f"\n✓ Pitch Writer produced {len(pitch_briefs)} investor-ready briefs")

    def test_critic(self, synthetic_state: VentureForgeState):
        """Test Critic reviews pitches with binary rubric."""
        from src.agents.idea_generator import run as run_idea_generator
        from src.agents.scorer import run as run_scorer
        from src.agents.pitch_writer import run as run_pitch_writer
        from src.agents.critic import run as run_critic

        # Generate full pipeline output
        result = run_idea_generator(synthetic_state)
        state_with_ideas = synthetic_state.model_copy(update=result)
        result = run_scorer(state_with_ideas)
        state_with_scores = state_with_ideas.model_copy(update=result)
        result = run_pitch_writer(state_with_scores)
        state_with_pitches = state_with_scores.model_copy(update=result)

        # Critique
        result = run_critic(state_with_pitches)
        critique = result.get("critique")

        assert critique is not None, "Critic must produce critique"
        assert critique.idea_id, "Critique must reference idea_id"
        assert critique.reasoning_trace, "Critique must include reasoning"
        assert critique.rubric, "Critique must have rubric"
        assert isinstance(critique.rubric.all_claims_evidence_backed, bool)
        assert isinstance(critique.rubric.no_hallucinated_source_urls, bool)
        assert isinstance(critique.rubric.tagline_under_12_words, bool)
        assert isinstance(critique.rubric.target_is_contained_fire, bool)
        assert isinstance(critique.rubric.competition_embraced_with_thesis, bool)
        assert isinstance(critique.rubric.unscalable_acquisition_concrete, bool)
        assert isinstance(critique.rubric.gtm_leads_with_manual_recruitment, bool)

        assert critique.approval_status in ["approved", "revise"], \
            f"Invalid approval status: {critique.approval_status}"

        if not critique.all_pass:
            assert critique.target_agent, "Must specify target agent for revision"
            assert critique.revision_feedback, "Must provide revision feedback"
            assert critique.failing_checks, "Must list failing checks"

        print(f"\n✓ Critic reviewed pitch: {critique.approval_status}")


# =============================================================================
# INTEGRATION TESTS (LangGraph orchestration)
# =============================================================================

class TestOrchestration:
    """Test full LangGraph orchestration with reflection loop."""

    @pytest.mark.slow
    def test_full_pipeline_synthetic(self, synthetic_state: VentureForgeState):
        """Test complete pipeline with synthetic data (fast)."""
        graph = build_graph()

        # Run the graph
        final_state = graph.invoke(
            synthetic_state,
            config={
                "recursion_limit": 80,
                "configurable": {"thread_id": synthetic_state.run_id},
            },
        )

        # Validate final state
        assert final_state.pain_points, "Should have pain points"
        assert final_state.ideas, "Should have generated ideas"
        assert final_state.scored_ideas, "Should have scored ideas"
        assert final_state.pitch_briefs, "Should have pitch briefs"
        assert final_state.current_stage in [
            PipelineStage.COMPLETED,
            PipelineStage.CRITIQUING,
        ], f"Unexpected stage: {final_state.current_stage}"

        # Check reflection loop was considered
        if final_state.critique:
            print(f"\n✓ Critique produced: {final_state.critique.approval_status}")
            if not final_state.critique.all_pass:
                print(f"  Failing checks: {final_state.critique.failing_checks}")
                print(f"  Target agent: {final_state.critique.target_agent}")

        print(f"\n✓ Full pipeline completed successfully")
        print(f"  Pain points: {len(final_state.pain_points)}")
        print(f"  Ideas: {len(final_state.ideas)}")
        print(f"  Scored ideas: {len(final_state.scored_ideas)}")
        print(f"  Pitch briefs: {len(final_state.pitch_briefs)}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_pipeline_real_data(self, initial_state: VentureForgeState):
        """Test complete pipeline with real Reddit/HN scraping (slow)."""
        graph = build_graph()

        # Run the graph
        final_state = graph.invoke(
            initial_state,
            config={
                "recursion_limit": 80,
                "configurable": {"thread_id": initial_state.run_id},
            },
        )

        # Validate final state
        assert final_state.pain_points, "Should extract real pain points"
        assert all(pp.source_url.startswith("http") for pp in final_state.pain_points), \
            "All pain points must have real URLs"
        assert final_state.ideas, "Should generate ideas from real pain points"
        assert final_state.scored_ideas, "Should score ideas"
        assert final_state.pitch_briefs, "Should write pitches"

        # Validate no hallucinations
        for pp in final_state.pain_points:
            assert pp.source_url, "Pain point must have source URL"
            assert pp.raw_quote, "Pain point must have verbatim quote"

        for brief in final_state.pitch_briefs:
            assert brief.evidence_links, "Pitch must cite evidence"
            for link in brief.evidence_links:
                assert link.startswith("http"), f"Invalid evidence link: {link}"

        print(f"\n✓ Full pipeline with real data completed")
        print(f"  Pain points: {len(final_state.pain_points)}")
        print(f"  Ideas: {len(final_state.ideas)}")
        print(f"  Pitch briefs: {len(final_state.pitch_briefs)}")

    def test_reflection_loop_triggers(self, synthetic_state: VentureForgeState):
        """Test that reflection loop can trigger and route correctly."""
        from src.agents.critic import run as run_critic
        from src.agents.idea_generator import run as run_idea_generator
        from src.agents.scorer import run as run_scorer
        from src.agents.pitch_writer import run as run_pitch_writer

        # Build state with pitches
        result = run_idea_generator(synthetic_state)
        state = synthetic_state.model_copy(update=result)
        result = run_scorer(state)
        state = state.model_copy(update=result)
        result = run_pitch_writer(state)
        state = state.model_copy(update=result)

        # Run critic
        result = run_critic(state)
        state = state.model_copy(update=result)

        if state.critique and not state.critique.all_pass:
            # Test revision logic
            assert state.can_revise, "Should be able to revise on first critique"
            assert state.critique.target_agent, "Must specify target for revision"
            assert state.critique.revision_feedback, "Must provide feedback"

            # Test bump_revision
            patch = state.bump_revision(state.critique)
            idea_id_str = str(state.critique.idea_id)
            assert idea_id_str in patch["revision_counts"]
            assert patch["revision_counts"][idea_id_str] == 1
            assert "next_node" in patch, "bump_revision should set next_node"
            assert patch["next_node"] == state.critique.target_agent

            # Test reset_for_revision (clears downstream data AND critique)
            reset_patch = state.reset_for_revision(state.critique.target_agent)
            assert "critique" in reset_patch, "Should clear critique"
            assert reset_patch["critique"] is None
            assert "current_stage" in reset_patch
            assert reset_patch["current_stage"] == PipelineStage.REVISING
            # Verify appropriate fields are cleared based on target agent
            if state.critique.target_agent == "pain_point_miner":
                assert "pain_points" in reset_patch
            elif state.critique.target_agent == "idea_generator":
                assert "ideas" in reset_patch
            elif state.critique.target_agent == "pitch_writer":
                assert "pitch_briefs" in reset_patch

            print(f"\n✓ Reflection loop logic validated")
            print(f"  Target agent: {state.critique.target_agent}")
            print(f"  Failing checks: {state.critique.failing_checks}")
        else:
            print(f"\n✓ Critique passed on first attempt (no revision needed)")

    def test_max_revisions_enforced(self, synthetic_state: VentureForgeState):
        """Test that max_revisions cap is enforced."""
        from src.agents.critic import run as run_critic

        # Simulate reaching max revisions
        test_idea_id = str(uuid4())
        state = synthetic_state.model_copy(
            update={
                "revision_counts": {test_idea_id: 2},  # Already at max
                "max_revisions": 2,
            }
        )

        # Check can_revise property
        assert not state.can_revise, "Should not allow revision at max_revisions"

        print(f"\n✓ Max revisions cap enforced correctly")


# =============================================================================
# SUCCESS CRITERIA VALIDATION
# =============================================================================

class TestSuccessCriteria:
    """Validate all success criteria from orchestration.json."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_all_success_criteria(self, initial_state: VentureForgeState):
        """Validate all success criteria in one comprehensive test."""
        graph = build_graph()

        # Run pipeline
        final_state = graph.invoke(
            initial_state,
            config={
                "recursion_limit": 80,
                "configurable": {"thread_id": initial_state.run_id},
            },
        )

        # ✓ Pipeline runs end-to-end without crashing
        assert final_state is not None, "Pipeline must complete"

        # ✓ All pain points have real, verifiable source URLs
        for pp in final_state.pain_points:
            assert pp.source_url.startswith("http"), \
                f"Pain point must have real URL: {pp.source_url}"
            assert pp.raw_quote, "Pain point must have verbatim quote"

        # ✓ Critic reflection loop triggers (or passes on first attempt)
        if final_state.critique:
            print(f"\n✓ Critic evaluation: {final_state.critique.approval_status}")
            if not final_state.critique.all_pass:
                print(f"  Reflection loop triggered for: {final_state.critique.target_agent}")

        # ✓ Binary rubrics used throughout
        for si in final_state.scored_ideas:
            assert 0 <= si.yes_count <= 8, "Scorer must use binary rubric"
            assert si.verdict in [Verdict.PURSUE, Verdict.EXPLORE, Verdict.PARK]

        # ✓ No hallucinations in pitch briefs
        for brief in final_state.pitch_briefs:
            assert brief.evidence_links, "Pitch must cite evidence"
            for link in brief.evidence_links:
                assert link.startswith("http"), f"Evidence link must be real URL: {link}"

        # ✓ State is immutable (agents return patches, not mutations)
        # This is enforced by Pydantic and the agent design pattern

        print(f"\n" + "=" * 60)
        print("ALL SUCCESS CRITERIA VALIDATED ✓")
        print("=" * 60)
        print(f"Pain points extracted: {len(final_state.pain_points)}")
        print(f"Ideas generated: {len(final_state.ideas)}")
        print(f"Ideas scored: {len(final_state.scored_ideas)}")
        print(f"Pitch briefs written: {len(final_state.pitch_briefs)}")
        print(f"Final stage: {final_state.current_stage}")


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    # Check environment
    if not (settings.llm_api_key or settings.fast_llm_api_key or os.getenv("OPENAI_API_KEY")):
        print("ERROR: No LLM API key configured")
        print("Set LLM_API_KEY, FAST_LLM_API_KEY, or OPENAI_API_KEY in .env")
        sys.exit(1)

    print("=" * 60)
    print("VentureForge End-to-End Test Suite")
    print("=" * 60)
    print(f"LLM Provider: {settings.llm_base_url}")
    print(f"Model: {settings.llm_model}")
    print("=" * 60)

    # Run pytest programmatically
    exit_code = pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "not slow",  # Skip slow tests by default
    ])

    sys.exit(exit_code)
