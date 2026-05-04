"""Critic — adversarial reviewer evaluating pitch briefs with binary rubric."""
from __future__ import annotations

from src.state.schema import (
    CriticRubric,
    CriticStatus,
    Critique,
    PipelineStage,
    TargetAgent,
    VentureForgeState,
)


def run(state: VentureForgeState) -> dict:
    """
    Stub: critique pitch briefs.
    Auto-approve on max revisions; otherwise revise.
    """
    if state.revision_count >= state.max_revisions:
        # Auto-approve at max revisions
        rubric = CriticRubric(
            all_claims_evidence_backed=True,
            no_hallucinated_source_urls=True,
            target_user_specific_not_generic=True,
            honest_risk_disclosure=True,
            no_buzzword_filler=True,
            tagline_under_12_words=True,
            go_to_market_concrete=True,
        )
        critique = Critique(
            rubric=rubric,
            all_pass=True,
            approval_status=CriticStatus.APPROVED,
            target_agent=TargetAgent.PITCH_WRITER,
            revision_feedback="Max revisions reached — approved by default.",
        )
        return {
            "critique": critique,
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }

    # Default: flag for revision to demonstrate the loop
    rubric = CriticRubric(
        all_claims_evidence_backed=False,
        no_hallucinated_source_urls=True,
        target_user_specific_not_generic=False,
        honest_risk_disclosure=True,
        no_buzzword_filler=True,
        tagline_under_12_words=True,
        go_to_market_concrete=False,
    )
    critique = Critique(
        rubric=rubric,
        all_pass=False,
        approval_status=CriticStatus.REVISE,
        target_agent=TargetAgent.PITCH_WRITER,
        revision_feedback="Increase evidence-backed claims and concretize go-to-market. Specify target user more precisely.",
    )
    return {
        "critique": critique,
        "current_stage": PipelineStage.CRITIQUING,
        "next_node": "orchestrator",
    }
