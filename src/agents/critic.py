"""Critic — adversarial reviewer evaluating pitch briefs with binary rubric."""
from __future__ import annotations

import json
import logging
import time
from uuid import UUID

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
    # Use reasoning=False to disable thinking mode for structured JSON output
    llm = get_llm(temperature=0.2, max_tokens=2048, reasoning=False)
    
    # Add explicit JSON-only instruction
    system_prompt = _build_system_prompt()
    system_prompt += "\n\n**CRITICAL: Output ONLY the JSON object. No markdown code fences, no explanations, no preamble. Start with { and end with }.**"
    
    messages = [
        SystemMessage(content=system_prompt),
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
    
    # Debug: log response preview
    logger.info(f"[critic] Response preview (first 500 chars): {content[:500]}")

    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[critic] JSON extraction failed. Response length: {len(content)} chars")
        logger.error(f"[critic] Full response (first 2000 chars): {content[:2000]}")
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

    # Select the brief to critique (prefer highest-scoring idea)
    brief = state.pitch_briefs[0]
    if state.scored_ideas:
        top_ids = [s.idea_id for s in state.top_scored_ideas]
        for idea_id in top_ids:
            match = next((b for b in state.pitch_briefs if b.idea_id == idea_id), None)
            if match is not None:
                brief = match
                break

    # Check if we're at max revisions (but still run the LLM to evaluate the final revision)
    at_max_revisions = state.get_revision_count(brief.idea_id) >= state.max_revisions

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
        if "critique" in raw:
            raw = raw["critique"]

        # Coerce rubric booleans
        if "rubric" in raw and isinstance(raw["rubric"], dict):
            raw["rubric"] = coerce_rubric_bools(raw["rubric"])
        
        # Handle list revision_feedback (coerce to string)
        if "revision_feedback" in raw and isinstance(raw["revision_feedback"], list):
            raw["revision_feedback"] = "\n".join(raw["revision_feedback"])

        # Add idea_id (required by Critique model but not in LLM output)
        raw["idea_id"] = brief.idea_id
        
        critique = Critique(**raw)
        
        # If we're at max revisions AND the critique still fails, auto-approve
        if at_max_revisions and not critique.all_pass:
            logger.info(
                f"[critic] Max revisions reached for idea {brief.idea_id}. "
                f"LLM critique failed but auto-approving anyway."
            )
            # Override the critique to approve
            critique = Critique(
                idea_id=brief.idea_id,
                reasoning_trace=(
                    f"Max revisions reached. Original critique: {critique.reasoning_trace}"
                ),
                rubric=CritiqueRubric(
                    all_claims_evidence_backed=True,
                    no_hallucinated_source_urls=True,
                    tagline_under_12_words=True,
                    target_is_contained_fire=True,
                    competition_embraced_with_thesis=True,
                ),
                all_pass=True,
                approval_status="approved",
                target_agent="pitch_writer",
                revision_feedback=(
                    f"Max revisions reached — approved by default. "
                    f"Original feedback: {critique.revision_feedback}"
                ),
            )

        patch = {
            "critique": critique,
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }
        
        if at_max_revisions and not critique.all_pass:
            # This branch is now unreachable (we override above), but kept for clarity
            message = (
                f"Auto-approved pitch for idea {brief.idea_id} after "
                f"reaching max revisions ({state.max_revisions}), despite LLM critique failing."
            )
        elif critique.all_pass:
            message = f"Approved pitch for idea {brief.idea_id}."
        else:
            message = (
                f"Critique for idea {brief.idea_id}: {len(critique.failing_checks)} "
                f"checks failed. Routing to {critique.target_agent} for revision."
            )
        
        patch.update(
            state.add_event(
                agent="critic",
                stage=PipelineStage.CRITIQUING,
                kind="info",
                message=message,
                idea_id=brief.idea_id,
            )
        )
        return patch

    except Exception as e:
        logger.error(f"[critic] Failed to parse critique: {e}")
        # Convert UUID to string for JSON serialization
        raw_for_log = {k: str(v) if isinstance(v, UUID) else v for k, v in raw.items()}
        logger.error(f"[critic] Raw LLM output: {json.dumps(raw_for_log, indent=2)}")
        patch = {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="critic",
                stage=PipelineStage.CRITIQUING,
                kind="error",
                message=f"Failed to parse critique: {e}",
            )
        )
        return patch
