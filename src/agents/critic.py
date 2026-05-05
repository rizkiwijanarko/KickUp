"""Critic — adversarial reviewer evaluating pitch briefs with binary rubric."""
from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import coerce_rubric_bools, extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    Critique,
    CritiqueRubric,
    PipelineStage,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return get_prompt("critic")


def _build_user_prompt(state: VentureForgeState) -> str:
    # Critique the pitch brief corresponding to the highest-scoring idea
    # (top_scored_ideas[0]) when possible; fall back to the first brief.
    if not state.pitch_briefs:
        return "No pitch briefs to review."

    brief = state.pitch_briefs[0]
    if state.scored_ideas:
        top_ids = [s.idea_id for s in state.top_scored_ideas]
        for idea_id in top_ids:
            match = next((b for b in state.pitch_briefs if b.idea_id == idea_id), None)
            if match is not None:
                brief = match
                break

    revision_count = state.get_revision_count(brief.idea_id)

    # Look up the Scorer output for this pitch so the Critic can cross-reference
    scored_idea = None
    for s in state.scored_ideas:
        if s.idea_id == brief.idea_id:
            scored_idea = s.model_dump(mode="json")
            break

    user_text = (
        f"Domain: {state.domain}\n"
        f"Current Revision Count: {revision_count}\n\n"
        f"PITCH BRIEF TO REVIEW:\n{brief.markdown_content}\n\n"
        f"SCORER OUTPUT FOR THIS IDEA:\n{json.dumps(scored_idea, indent=2) if scored_idea else 'Not found'}\n\n"
        "Provide a brutal, honest critique using the binary rubric. "
        "If it fails any check, specify which worker should fix it. "
        "Ensure to only return a JSON object."
    )
    return user_text


def _invoke_llm(state: VentureForgeState) -> dict:
    llm = get_llm(temperature=0.2, max_tokens=2048, reasoning=True)
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[critic] LLM invocation failed: {e}")
        return {}

    logger.info(f"[critic] LLM responded in {time.monotonic()-start:.1f}s")

    parsed = extract_json(content)
    if parsed is None:
        logger.error("[critic] JSON extraction failed")
        return {}
    return parsed


def run(state: VentureForgeState) -> dict:
    if not state.pitch_briefs:
        logger.warning("[critic] no pitch briefs to critique")
        patch = {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="critic",
                stage=PipelineStage.CRITIQUING,
                kind="warning",
                message="No pitch briefs available for critique.",
            )
        )
        return patch

    # Auto-approve if max revisions reached for this pitch. We apply the
    # same brief-selection logic as in _build_user_prompt: prefer the
    # pitch for the highest-scoring idea when available.
    brief = state.pitch_briefs[0]
    if state.scored_ideas:
        top_ids = [s.idea_id for s in state.top_scored_ideas]
        for idea_id in top_ids:
            match = next((b for b in state.pitch_briefs if b.idea_id == idea_id), None)
            if match is not None:
                brief = match
                break

    if state.get_revision_count(brief.idea_id) >= state.max_revisions:
        rubric = CritiqueRubric(
            all_claims_evidence_backed=True,
            no_hallucinated_source_urls=True,
            tagline_under_12_words=True,
            target_is_contained_fire=True,
            competition_embraced_with_thesis=True,
            unscalable_acquisition_concrete=True,
            gtm_leads_with_manual_recruitment=True,
        )
        critique = Critique(
            idea_id=brief.idea_id,
            reasoning_trace="Max revisions reached — approved by default.",
            rubric=rubric,
            all_pass=True,
            approval_status="approved",
            target_agent="pitch_writer",
            revision_feedback="Max revisions reached — approved by default.",
        )
        patch = {
            "critique": critique,
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="critic",
                stage=PipelineStage.CRITIQUING,
                kind="info",
                message=(
                    f"Auto-approved pitch for idea {brief.idea_id} after "
                    f"reaching max revisions ({state.max_revisions})."
                ),
                idea_id=brief.idea_id,
            )
        )
        return patch

    raw = _invoke_llm(state)
    if not raw:
        # Fallback to simple revision if LLM fails
        patch = {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="critic",
                stage=PipelineStage.CRITIQUING,
                kind="warning",
                message="Critic LLM invocation failed; keeping previous state.",
            )
        )
        return patch

    try:
        # Unwrap if LLM returned {"critique": {...}}
        if "critique" in raw and isinstance(raw["critique"], dict):
            raw = raw["critique"]

        rubric = CritiqueRubric(**coerce_rubric_bools(raw["rubric"]))
        critique = Critique(
            idea_id=brief.idea_id,
            reasoning_trace=raw.get("reasoning_trace", ""),
            rubric=rubric,
            all_pass=raw["all_pass"],
            approval_status=raw["approval_status"],
            failing_checks=raw.get("failing_checks", []),
            target_agent=raw["target_agent"],
            revision_feedback="\n".join(raw["revision_feedback"]) if isinstance(raw["revision_feedback"], list) else raw["revision_feedback"],
        )
    except Exception as e:
        logger.error(f"[critic] malformed critique: {e}")
        patch = {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="critic",
                stage=PipelineStage.CRITIQUING,
                kind="error",
                message="Critic produced a malformed critique; keeping previous state.",
            )
        )
        return patch

    # Normal successful critique
    message = (
        "Approved pitch" if critique.all_pass else
        f"Requested revision for idea {critique.idea_id} → target_agent={critique.target_agent}"
    )
    patch = {
        "critique": critique,
        "current_stage": PipelineStage.CRITIQUING,
        "next_node": "orchestrator",
    }
    patch.update(
        state.add_event(
            agent="critic",
            stage=PipelineStage.CRITIQUING,
            kind="info",
            message=message,
            idea_id=critique.idea_id,
        )
    )
    return patch
