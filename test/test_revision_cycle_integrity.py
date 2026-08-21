"""
Revision Cycle Integrity Tests
================================
Fast, offline tests that cover the gaps missed by existing component tests.

These tests target failure modes that ONLY manifest in multi-step sequences:
  1. RevisionLedger accumulation correctness after a full critique → revise → re-critique cycle.
  2. `_should_critique` routing predicate edge cases (critique=None + pending brief present).
  3. `critiques` list deduplication: bump_revision vs _build_success_patch don't double-accumulate.
  4. `pending_briefs` remains empty (pipeline completion) after all briefs are resolved.
  5. `quarantined_idea_ids` correctly identifies max-revision ideas even when `critique` is stale.

Run with:
    uv run pytest test/test_revision_cycle_integrity.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.agents.critic import _build_success_patch, run as run_critic
from src.agents.orchestrator import orchestrator
from src.agents.routing import _should_critique
from src.state.graph_state import VentureForgeState
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
    PipelineStage,
    PitchBrief,
    ScoredIdea,
    ValidationPlan,
)
from test.test_helpers import make_test_pain_point


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_rubric(all_pass: bool = True) -> CritiqueRubric:
    return CritiqueRubric(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=all_pass,
        competition_embraced_with_thesis=True,
        minimum_evidence_sources=True,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )


def _make_critique(idea_id, *, all_pass: bool, target_agent: str = "idea_generator") -> Critique:
    return Critique(
        idea_id=idea_id,
        reasoning_trace="Test reasoning",
        rubric=_make_rubric(all_pass=all_pass),
        all_pass=all_pass,
        approval_status="approved" if all_pass else "revise",
        failing_checks=[] if all_pass else ["target_is_contained_fire"],
        target_agent=target_agent,
        revision_feedback="Everything looks good." if all_pass else "Target is too broad.",
    )


def _make_minimal_state(max_revisions: int = 2) -> tuple[VentureForgeState, Idea, ScoredIdea, PitchBrief]:
    pp1 = make_test_pain_point(
        title="Docker pain",
        description="Developers struggle with Docker Compose setups.",
        source_url="https://reddit.com/r/docker/comments/abc",
        raw_quote="docker compose is so painful.",
        source=DataSource.REDDIT,
    )
    pp2 = make_test_pain_point(
        title="CI pain",
        description="CI debugging is extremely time-consuming for teams.",
        source_url="https://reddit.com/r/devops/comments/def",
        raw_quote="CI failures are impossible to reproduce locally.",
        source=DataSource.REDDIT,
    )
    idea = Idea(
        id=uuid4(),
        title="Docker Compose Debugger",
        one_liner="Debug Docker Compose visually.",
        problem="Docker Compose setups are painful to debug.",
        solution="A visual editor and debugger for Docker Compose files.",
        target_user="Backend developers",
        key_features=["Visual editor", "Error hints", "Auto-fix"],
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
        core_assumption="Developers want better tooling.",
        fatal_flaws=[FatalFlaw(flaw="Competition risk.", severity="minor")],
        yes_count=8,
        verdict="pursue",
        one_risk="Incumbents may copy.",
        rank=1,
    )
    brief = PitchBrief(
        idea_id=idea.id,
        title="Docker Compose Debugger",
        tagline="Debug Docker Compose visually.",
        problem=idea.problem,
        solution=idea.solution,
        target_user=idea.target_user,
        market_opportunity="Large developer tools market with $5B TAM.",
        competitive_landscape=CompetitiveLandscape(
            current_behavior="Manual YAML editing and trial-and-error debugging.",
            direct_competitors="Docker Desktop, VS Code YAML extension.",
            real_enemy="The habit of editing raw YAML without tooling.",
        ),
        differentiation="Visual-first experience vs raw text editing.",
        validation_plan=ValidationPlan(
            discovery_questions=[
                "When did you last debug a Docker Compose issue?",
                "How long did it take to resolve?",
                "What tool do you use currently?",
                "What would make you switch?",
                "Would you pay for a visual debugger?",
            ],
            validation_criteria="At least 7/10 developers cite 2+ hours/week on Docker debugging.",
        ),
        business_model="Monthly SaaS subscription with a freemium tier.",
        go_to_market="DM top 50 r/docker posters and onboard one by one.",
        key_risk="Incumbents may copy.",
        next_steps="Ship MVP and recruit 10 beta users from r/docker.",
        evidence_links=[pp1.source_url, pp2.source_url],
        markdown_content=(
            "# Docker Compose Debugger\n\n"
            "## Problem\nDocker Compose setups are painful to debug.\n\n"
            "## Solution\nA visual editor and debugger for Docker Compose files.\n"
        ),
    )
    state = VentureForgeState(
        domain="developer tools",
        max_revisions=max_revisions,
        pain_points=[pp1, pp2],
        ideas=[idea],
        scored_ideas=[scored],
        pitch_briefs=[brief],
    )
    return state, idea, scored, brief


# ---------------------------------------------------------------------------
# Test 1: RevisionLedger accumulation correctness
# ---------------------------------------------------------------------------


def test_revision_ledger_all_critiques_deduplicates_correctly() -> None:
    """RevisionLedger.all_critiques must return the latest critique per idea_id,
    not double-count when critique (singular) matches an id already in critiques list."""
    state, idea, _, _ = _make_minimal_state()
    idea_id = idea.id

    critique_v1 = _make_critique(idea_id, all_pass=False)
    critique_v2 = _make_critique(idea_id, all_pass=True)  # second version, same idea

    # Simulate a stale plural entry plus the active latest critique.
    state = state.model_copy(update={
        "critiques": [critique_v1],
        "critique": critique_v2,
    })

    ledger = state.revisions
    all_c = ledger.all_critiques

    assert all_c == [critique_v2], "all_critiques must contain one latest entry per idea"

    latest = ledger._latest_by_idea()
    assert latest[idea_id].all_pass is True, (
        f"Latest critique for idea should be v2 (all_pass=True), got all_pass={latest[idea_id].all_pass}"
    )

    # approved_idea_ids should include this idea since the latest critique passes
    assert idea_id in ledger.approved_idea_ids, "Idea should be in approved_idea_ids"
    assert idea_id not in ledger.quarantined_idea_ids, "Idea should NOT be quarantined"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 2: _should_critique routing predicate after revision reset
# ---------------------------------------------------------------------------


def test_should_critique_is_true_when_critique_is_none_and_pending_briefs_exist() -> None:
    """After reset_for_revision sets critique=None, _should_critique must re-trigger.

    This is the most critical routing edge case: after orchestrator calls
    reset_for_revision (which sets critique=None), the next orchestrator call
    must route back to critic, not to __end__.
    """
    state, idea, _, brief = _make_minimal_state()

    # State after orchestrator has reset for revision: critique=None, brief still present
    state = state.model_copy(update={
        "critique": None,
        "critiques": [],  # clean slate
        "current_stage": PipelineStage.REVISING,
    })

    result = _should_critique(state)
    assert result is True, (
        "_should_critique should return True when critique=None and pending briefs exist"
    )
    print("  PASS")


def test_should_critique_is_false_when_current_critique_matches_pending_brief() -> None:
    """If critique.idea_id IS in pending briefs, we're mid-review—don't re-trigger.

    This prevents the orchestrator from sending the same brief back to critic
    before the current critique result has been processed.
    """
    state, idea, _, brief = _make_minimal_state()
    critique = _make_critique(idea.id, all_pass=False)

    # critique exists and its idea IS in pending_briefs (not yet approved)
    state = state.model_copy(update={
        "critique": critique,
        "critiques": [],
    })

    # pending_briefs will include this brief since it's not approved
    pending = state.revisions.pending_briefs(state.pitch_briefs)
    assert any(b.idea_id == idea.id for b in pending), "Brief must be pending for this test"

    result = _should_critique(state)
    # critique.idea_id IS in pending → predicate line 83 evaluates to False
    assert result is False, (
        "_should_critique should be False when critique.idea_id is already in pending briefs "
        "(we already have a critique for it; orchestrator should route to revision, not re-critique)"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 3: critiques list deduplication after full revise cycle
# ---------------------------------------------------------------------------


def test_critiques_list_not_doubled_after_bump_and_rebuild() -> None:
    """After critique → bump_revision → re-critique, critiques list must not
    accumulate duplicate entries for the same idea_id.

    Revision metadata updates must not compete with the critic's canonical
    latest-per-idea critique snapshot.
    """
    state, idea, _, brief = _make_minimal_state()
    idea_id = idea.id

    critique_v1 = _make_critique(idea_id, all_pass=False)

    # Step 1: Simulate orchestrator's bump_revision call
    bump_patch = state.bump_revision(critique_v1)
    state_after_bump = state.model_copy(update=bump_patch)

    assert "critiques" not in bump_patch
    assert state_after_bump.critiques == state.critiques

    # Step 2: Exercise the production critic patch builder.
    critique_v2 = _make_critique(idea_id, all_pass=True)
    success_patch = _build_success_patch(state_after_bump, critique_v2, at_max_revisions=False)
    state_after_rereview = state_after_bump.model_copy(update=success_patch)

    # Should have exactly 1 critique for this idea (v2 replaced v1)
    idea_critiques = [c for c in state_after_rereview.critiques if c.idea_id == idea_id]
    assert len(idea_critiques) == 1, (
        f"Expected exactly 1 critique for idea after revise cycle, got {len(idea_critiques)}"
    )
    assert idea_critiques[0].all_pass is True, "Surviving critique should be v2 (all_pass=True)"

    # RevisionLedger should correctly report this as approved
    ledger = state_after_rereview.revisions
    assert idea_id in ledger.approved_idea_ids, "Idea should be approved after passing re-critique"
    print("  PASS")


def test_repeated_failed_revisions_keep_only_latest_critique() -> None:
    """Repeated failures followed by approval retain one latest critique."""
    state, idea, scored, brief = _make_minimal_state(max_revisions=3)
    idea_id = idea.id

    for critique in (
        _make_critique(idea_id, all_pass=False),
        _make_critique(idea_id, all_pass=False),
    ):
        state = state.model_copy(update={"critique": critique})
        revision_patch = orchestrator(state)
        assert revision_patch["next_node"] == "idea_generator"
        state = state.model_copy(update=revision_patch)
        assert "critiques" not in revision_patch
        state = state.model_copy(update={
            "critique": None,
            "pitch_briefs": [brief],
            "ideas": [idea],
            "scored_ideas": [scored],
        })

    final_critique = _make_critique(idea_id, all_pass=True)
    state = state.model_copy(update={
        "critique": final_critique,
        "critiques": [final_critique],
    })

    assert state.revisions.count(idea_id) == 2
    assert state.revisions.all_critiques == [final_critique]
    assert idea_id in state.revisions.approved_idea_ids
    assert state.revisions.pending_briefs(state.pitch_briefs) == []


# ---------------------------------------------------------------------------
# Test 4: Pipeline completion when all briefs are resolved
# ---------------------------------------------------------------------------


def test_pending_briefs_empty_after_all_approved_triggers_completion() -> None:
    """When all briefs are approved, pending_briefs returns [] and orchestrator
    must route to __end__, not back to critic."""
    state, idea, _, brief = _make_minimal_state()
    idea_id = idea.id

    approved_critique = _make_critique(idea_id, all_pass=True)

    state = state.model_copy(update={
        "critique": approved_critique,
        "critiques": [approved_critique],
        "current_stage": PipelineStage.CRITIQUING,
    })

    # Confirm pending_briefs is empty
    pending = state.revisions.pending_briefs(state.pitch_briefs)
    assert pending == [], f"Expected no pending briefs after approval, got {pending}"

    # Orchestrator must route to __end__
    patch = orchestrator(state)
    assert patch["current_stage"] == PipelineStage.COMPLETED, (
        f"Expected COMPLETED, got {patch['current_stage']}"
    )
    assert patch["next_node"] == "__end__", (
        f"Expected __end__, got {patch['next_node']}"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 5: quarantined_idea_ids uses revision_counts correctly
# ---------------------------------------------------------------------------


def test_quarantined_ids_detected_via_revision_counts_not_just_approval_status() -> None:
    """quarantined_idea_ids must catch ideas that hit max_revisions even if
    approval_status is 'revise' (not 'max_revisions_reached').

    The ledger counts via `self.count(id_) >= self._max_revisions` as a fallback,
    meaning an idea can be quarantined even if the critic didn't explicitly set
    max_revisions_reached.
    """
    state, idea, _, brief = _make_minimal_state(max_revisions=2)
    idea_id = idea.id

    # Critique with approval_status='revise' (critic didn't set max_revisions_reached)
    # but revision_counts says we're at max
    critique = _make_critique(idea_id, all_pass=False)
    # Don't set approval_status='max_revisions_reached' — use raw count fallback

    state = state.model_copy(update={
        "critique": critique,
        "critiques": [critique],
        "revision_counts": {str(idea_id): 2},  # at max
    })

    ledger = state.revisions
    quarantined = ledger.quarantined_idea_ids

    assert idea_id in quarantined, (
        f"Expected idea_id to be quarantined via revision_counts fallback, "
        f"but quarantined={quarantined}"
    )
    # Should NOT appear in approved (all_pass=False)
    assert idea_id not in ledger.approved_idea_ids, "Quarantined idea should not be approved"
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 6: Critic correctly locks idea_id to the reviewed brief
# ---------------------------------------------------------------------------


def test_critic_idea_id_locked_to_reviewed_brief_not_llm_output() -> None:
    """Critic must override critique.idea_id with the actual brief's idea_id,
    not use whatever the LLM hallucinated.

    This is the fix from our previous session. Regression test.
    """
    state, idea, _, brief = _make_minimal_state()
    wrong_idea_id = uuid4()  # LLM returns a wrong/hallucinated UUID

    with patch("src.agents.critic.get_structured_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = Critique(
            idea_id=wrong_idea_id,  # WRONG: LLM hallucinated a different id
            reasoning_trace="Pitch looks good.",
            rubric=_make_rubric(all_pass=True),
            all_pass=True,
            approval_status="approved",
            failing_checks=[],
            target_agent="pitch_writer",
            revision_feedback="Everything looks good.",
        )
        mock_get_llm.return_value = fake_llm

        result = run_critic(state)

    assert "critique" in result, "Expected critique in result"
    critique = result["critique"]
    assert critique.idea_id == brief.idea_id, (
        f"Critic must lock idea_id to reviewed brief ({brief.idea_id}), "
        f"not use LLM output ({wrong_idea_id})"
    )
    print("  PASS")


# ---------------------------------------------------------------------------
# Test 7: Full orchestrator routing after a complete revise-then-approve cycle
# ---------------------------------------------------------------------------


def test_orchestrator_routes_correctly_through_full_critique_revise_approve_cycle() -> None:
    """Simulate the multi-step sequence that E2E exercises:
    1. Orchestrator has pending briefs → routes to critic
    2. Critic returns failing critique → orchestrator routes to revision
    3. After revision, critique=None + pending brief → routes back to critic
    4. Critic returns passing critique → routes to completion
    """
    state, idea, scored, brief = _make_minimal_state()
    idea_id = idea.id

    # Step 1: No critique yet → should route to critic
    patch1 = orchestrator(state)
    assert patch1["next_node"] == "critic", f"Step 1: Expected critic, got {patch1['next_node']}"
    state = state.model_copy(update=patch1)

    # Step 2: Critic produces failing result
    failing_critique = _make_critique(idea_id, all_pass=False, target_agent="idea_generator")
    state = state.model_copy(update={"critique": failing_critique, "critiques": []})

    # Orchestrator sees failing critique → should route to revision
    patch2 = orchestrator(state)
    assert patch2["next_node"] == "idea_generator", (
        f"Step 2: Expected idea_generator for revision, got {patch2['next_node']}"
    )
    assert patch2["current_stage"] == PipelineStage.REVISING
    state = state.model_copy(update=patch2)

    # Step 3: After revision, critique=None (reset), brief still pending (regenerated)
    # Simulate that idea_generator put a fresh brief back (same idea_id for simplicity)
    state = state.model_copy(update={
        "critique": None,
        "pitch_briefs": [brief],  # brief is back after revision
        "ideas": [idea],
        "scored_ideas": [scored],
    })
    patch3 = orchestrator(state)
    # With critique=None + pending brief → should go back to critic
    assert patch3["next_node"] in ("critic", "idea_generator", "scorer", "pitch_writer"), (
        f"Step 3: Got unexpected node {patch3['next_node']}"
    )
    # Specifically it should not complete yet
    assert patch3.get("current_stage") != PipelineStage.COMPLETED, (
        "Step 3: Should NOT complete when briefs still pending"
    )
    state = state.model_copy(update=patch3)

    # Step 4: Critic now approves
    approved_critique = _make_critique(idea_id, all_pass=True)
    state = state.model_copy(update={
        "critique": approved_critique,
        "critiques": [approved_critique],
        "pitch_briefs": [brief],
    })
    patch4 = orchestrator(state)
    assert patch4["current_stage"] == PipelineStage.COMPLETED, (
        f"Step 4: Expected COMPLETED after approval, got {patch4['current_stage']}"
    )
    assert patch4["next_node"] == "__end__"
    print("  PASS")


def test_mining_failure_exhausts_budget_and_fails_not_loops() -> None:
    """Persistent mining failure must terminate in a FAILED state, not loop."""
    state, _, _, brief = _make_minimal_state()
    exhausted_budget = state.model_copy(update={
        "pain_points": [],
        "ideas": [],
        "pain_point_miner_revision_count": 2,
        "pitch_briefs": [brief],
    })

    patch = orchestrator(exhausted_budget)
    assert patch["next_node"] == "__end__"
    assert patch["current_stage"] == PipelineStage.FAILED


# ---------------------------------------------------------------------------
# Test 8: Computed target_agent routing matrix
# ---------------------------------------------------------------------------


def test_target_agent_computed_from_rubric_routing_matrix() -> None:
    """target_agent must be derived from the rubric, never from the LLM.

    - positioning-only failures -> idea_generator
    - any evidence/claims failure -> pitch_writer (even alongside positioning)
    - pure pitch failures (tagline, validation) -> pitch_writer
    """
    from src.models.critique import CritiqueRubric as CR

    def make(rubric: CR) -> str:
        c = Critique(
            idea_id=uuid4(),
            reasoning_trace="test",
            rubric=rubric,
            all_pass=False,
            approval_status="revise",
            revision_feedback="Fix the pitch.",
        )
        return c.target_agent

    positioning_only = CR(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=False,
        competition_embraced_with_thesis=False,
        minimum_evidence_sources=True,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    assert make(positioning_only) == "idea_generator"

    positioning_plus_claims = CR(
        all_claims_evidence_backed=False,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=False,
        competition_embraced_with_thesis=False,
        minimum_evidence_sources=True,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    assert make(positioning_plus_claims) == "pitch_writer"

    evidence_only = CR(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=False,
        tagline_under_12_words=True,
        target_is_contained_fire=True,
        competition_embraced_with_thesis=True,
        minimum_evidence_sources=False,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    assert make(evidence_only) == "pitch_writer"

    scorer_issue_only = CR(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=True,
        target_is_contained_fire=True,
        competition_embraced_with_thesis=True,
        minimum_evidence_sources=True,
        scorer_verdict_justified=False,
        validation_plan_complete=True,
    )
    assert make(scorer_issue_only) == "pitch_writer"

    tagline_only = CR(
        all_claims_evidence_backed=True,
        no_hallucinated_source_urls=True,
        tagline_under_12_words=False,
        target_is_contained_fire=True,
        competition_embraced_with_thesis=True,
        minimum_evidence_sources=True,
        scorer_verdict_justified=True,
        validation_plan_complete=True,
    )
    assert make(tagline_only) == "pitch_writer"
    print("  PASS")


def test_idea_generator_revision_preserves_idea_id() -> None:
    """In revision mode, idea_generator must keep the target idea's ID
    (identity-preserving revision), not mint a fresh UUID."""
    from src.agents.idea_generator import run as run_idea_generator

    state, idea, scored, brief = _make_minimal_state()

    # Simulate orchestrator routing to idea_generator for a positioning revision
    state = state.model_copy(
        update={
            "current_revision_idea_id": idea.id,
            "revision_feedback": "Target user is too broad; define a specific reachable community.",
        }
    )

    payload = {
        "title": "Docker Compose Debugger Refined",
        "one_liner": "Visual Docker Compose debugging.",
        "problem": "Long enough problem statement for schema validation purposes.",
        "solution": "Long enough solution description for schema validation purposes.",
        "target_user": "r/docker power users",
        "key_features": ["A", "B", "C"],
        "addresses_pain_point_ids": [str(pp.id) for pp in state.filtered_pain_points],
    }

    def _fake_invoke(messages):  # type: ignore[no-untyped-def]
        fake_resp = MagicMock()
        fake_resp.content = json.dumps(payload)
        return fake_resp

    with patch("src.agents.idea_generator.get_llm") as mock_get_llm:
        fake_llm = MagicMock()
        fake_llm.invoke.side_effect = _fake_invoke
        mock_get_llm.return_value = fake_llm

        result = run_idea_generator(state)

    assert result["ideas"], "Expected refined idea from idea_generator"
    refined = result["ideas"][0]
    assert refined.id == idea.id, (
        f"Revision must preserve idea ID: expected {idea.id}, got {refined.id}"
    )
    assert refined.title == "Docker Compose Debugger Refined"
    print("  PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_TESTS = [
    ("RevisionLedger deduplication correctness", test_revision_ledger_all_critiques_deduplicates_correctly),
    ("_should_critique=True when critique=None + pending", test_should_critique_is_true_when_critique_is_none_and_pending_briefs_exist),
    ("_should_critique=False when critique matches pending", test_should_critique_is_false_when_current_critique_matches_pending_brief),
    ("critiques list not doubled after bump+rebuild", test_critiques_list_not_doubled_after_bump_and_rebuild),
    ("mining failure exhausted budget -> FAILED", test_mining_failure_exhausts_budget_and_fails_not_loops),
    ("pending_briefs empty → orchestrator completes", test_pending_briefs_empty_after_all_approved_triggers_completion),
    ("quarantined via revision_counts fallback", test_quarantined_ids_detected_via_revision_counts_not_just_approval_status),
    ("critic locks idea_id to reviewed brief", test_critic_idea_id_locked_to_reviewed_brief_not_llm_output),
    ("full critique→revise→approve cycle", test_orchestrator_routes_correctly_through_full_critique_revise_approve_cycle),
    ("computed target_agent routing matrix", test_target_agent_computed_from_rubric_routing_matrix),
    ("idea_generator revision preserves idea ID", test_idea_generator_revision_preserves_idea_id),
]


if __name__ == "__main__":
    print("=" * 70)
    print("Revision Cycle Integrity Tests")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed:
        import sys
        sys.exit(1)
