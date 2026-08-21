"""Critic — adversarial reviewer evaluating pitch briefs with binary rubric."""

from __future__ import annotations

import json
import logging
import time
from typing import Literal
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from src.constants import (
    CRITIC_LLM_TEMPERATURE,
    ERROR_CRITIC_LLM_INVOCATION_FAILED,
    WARNING_CRITIC_NO_BRIEFS,
)
from src.exceptions import LLMError, ValidationError
from src.llm.client import get_structured_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    Critique,
    PipelineStage,
    PitchBrief,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Prompt Building
# ============================================================================


def _build_system_prompt() -> str:
    """Build the system prompt for the Critic agent.

    Returns:
        System prompt
    """
    base_prompt = get_prompt("critic")
    # No JSON instruction needed - structured output handles this
    return base_prompt


def _get_brief_to_review(state: VentureForgeState) -> tuple[int, PitchBrief]:
    """Get the next pending pitch brief to review.

    Args:
        state: Current pipeline state

    Returns:
        Tuple of (index, brief)
    """
    if not state.pitch_briefs:
        raise ValidationError(WARNING_CRITIC_NO_BRIEFS)

    pending = state.revisions.pending_briefs(state.pitch_briefs)
    if not pending:
        return len(state.pitch_briefs) - 1, state.pitch_briefs[-1]

    brief = pending[0]
    try:
        index = state.pitch_briefs.index(brief)
    except ValueError:
        index = 0
    return index, brief


def _get_scored_idea(state: VentureForgeState, idea_id: UUID) -> dict | None:
    """Look up the Scorer output for a given idea.

    Args:
        state: Current pipeline state
        idea_id: ID of the idea to find

    Returns:
        Scored idea dict or None if not found
    """
    for scored in state.scored_ideas:
        if scored.idea_id == idea_id:
            return scored.model_dump(mode="json")
    return None


def _serialize_brief(brief: PitchBrief) -> dict:
    """Serialize a pitch brief for JSON output.

    Args:
        brief: Pitch brief to serialize

    Returns:
        Serialized brief dict with UUID converted to string
    """
    brief_dict = brief.model_dump(mode="json")
    brief_dict["idea_id"] = str(brief_dict["idea_id"])
    return brief_dict


def _build_user_prompt(state: VentureForgeState) -> str:
    """Build the user prompt for the Critic agent.

    Args:
        state: Current pipeline state

    Returns:
        User prompt with brief details and context
    """
    index, brief = _get_brief_to_review(state)
    revision_count = state.revisions.count(brief.idea_id)
    scored_idea = _get_scored_idea(state, brief.idea_id)
    brief_dict = _serialize_brief(brief)

    # Get relevant pain points for this idea to verify evidence_links
    idea = next((i for i in state.ideas if i.id == brief.idea_id), None)
    relevant_pain_points = []
    if idea:
        for pp_id in idea.addresses_pain_point_ids:
            pp = next((p for p in state.filtered_pain_points if p.id == pp_id), None)
            if pp:
                pp_dict = {
                    "id": str(pp.id),
                    "title": pp.title,
                    "evidence_urls": [ev.source_url for ev in pp.evidence],
                }
                relevant_pain_points.append(pp_dict)

    user_text = (
        f"Domain: {state.domain}\n"
        f"Current Revision Count: {revision_count}\n"
        f"Reviewing brief {index + 1} of {len(state.pitch_briefs)}\n\n"
        f"PITCH BRIEF TO REVIEW (structured):\n{json.dumps(brief_dict, indent=2)}\n\n"
        f"PITCH BRIEF MARKDOWN:\n{brief.markdown_content}\n\n"
        f"SCORER OUTPUT FOR THIS IDEA:\n{json.dumps(scored_idea, indent=2) if scored_idea else 'Not found'}\n\n"
        f"RELEVANT PAIN POINTS (for evidence verification):\n{json.dumps(relevant_pain_points, indent=2)}\n\n"
        "Provide a brutal, honest critique using the binary rubric. "
        "If it fails any check, specify which worker should fix it."
    )
    return user_text


# ============================================================================
# LLM Interaction
# ============================================================================


def _invoke_llm(state: VentureForgeState) -> Critique:
    """Invoke the LLM to generate a critique using structured output.

    Args:
        state: Current pipeline state

    Returns:
        Critique object from structured output

    Raises:
        LLMError: If LLM invocation fails
    """
    llm = get_structured_llm(
        Critique,
        temperature=CRITIC_LLM_TEMPERATURE,
        reasoning=False,
    )

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    start = time.monotonic()
    try:
        critique = llm.invoke(messages)
    except Exception as e:
        logger.error(f"[critic] LLM invocation failed: {e}")
        raise LLMError(ERROR_CRITIC_LLM_INVOCATION_FAILED.format(error=str(e)))

    elapsed = time.monotonic() - start
    logger.info(f"[critic] LLM responded in {elapsed:.1f}s")

    return critique


# ============================================================================
# Revision Logic
# ============================================================================


def _is_at_max_revisions(state: VentureForgeState, idea_id: UUID) -> bool:
    """Check if an idea has reached max revisions.

    Args:
        state: Current pipeline state
        idea_id: ID of the idea to check

    Returns:
        True if at max revisions, False otherwise
    """
    return not state.revisions.can_revise(idea_id)


def _handle_max_revisions(critique: Critique, state: VentureForgeState) -> Critique:
    """Handle critique when max revisions reached.

    If the critique still fails at max revisions, mark as 'max_revisions_reached'
    instead of forcing approval.

    Args:
        critique: The critique object
        state: Current pipeline state

    Returns:
        Modified critique with max_revisions_reached status
    """
    if not critique.all_pass:
        logger.warning(
            f"[critic] Max revisions reached for idea {critique.idea_id}. "
            f"LLM critique failed but cannot revise further. "
            f"Marking as 'max_revisions_reached' instead of 'approved'."
        )
        critique.approval_status = "max_revisions_reached"
        critique.revision_feedback = (
            f"Max revisions reached ({state.max_revisions}). Cannot revise further. "
            f"Original feedback: {critique.revision_feedback}"
        )
    return critique


# ============================================================================
# Event & Patch Building
# ============================================================================


def _build_critique_message(critique: Critique, at_max_revisions: bool, max_revisions: int) -> str:
    """Build the event message for a critique.

    Args:
        critique: The critique object
        at_max_revisions: Whether at max revisions
        max_revisions: Max revisions allowed

    Returns:
        Event message string
    """
    if at_max_revisions and not critique.all_pass:
        return (
            f"Auto-approved pitch for idea {critique.idea_id} after "
            f"reaching max revisions ({max_revisions}), despite LLM critique failing."
        )
    elif critique.all_pass:
        return f"Approved pitch for idea {critique.idea_id}."
    else:
        return (
            f"Critique for idea {critique.idea_id}: {len(critique.failing_checks)} "
            f"checks failed. Routing to {critique.target_agent} for revision."
        )


def _build_success_patch(
    state: VentureForgeState, critique: Critique, at_max_revisions: bool
) -> dict:
    """Build the patch dict for a successful critique.

    Args:
        state: Current pipeline state
        critique: The critique object
        at_max_revisions: Whether at max revisions

    Returns:
        Patch dict for state update
    """
    message = _build_critique_message(critique, at_max_revisions, state.max_revisions)

    # Replace the latest critique for this idea in the canonical snapshot.
    updated_critiques = [c for c in state.critiques if c.idea_id != critique.idea_id] + [critique]

    patch = {
        "critique": critique,
        "critiques": updated_critiques,
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


def _build_error_patch(
    state: VentureForgeState,
    error_message: str,
    kind: Literal["info", "warning", "error"] = "error",
) -> dict:
    """Build the patch dict for an error case.

    Args:
        state: Current pipeline state
        error_message: Error message to log
        kind: Event kind (error, warning)

    Returns:
        Patch dict for state update
    """
    patch = {
        "current_stage": PipelineStage.CRITIQUING,
        "next_node": "orchestrator",
    }

    patch.update(
        state.add_event(
            agent="critic",
            stage=PipelineStage.CRITIQUING,
            kind=kind,
            message=error_message,
        )
    )

    return patch


# ============================================================================
# Main Entry Point
# ============================================================================


def run(state: VentureForgeState) -> dict:
    """Run the Critic agent to evaluate a pitch brief.

    The Critic performs adversarial review using a binary rubric. If the brief
    fails any checks, it routes back to the appropriate worker for revision.
    At max revisions, the brief is marked as 'max_revisions_reached' rather
    than forcing approval.

    Args:
        state: Current pipeline state

    Returns:
        Patch dict to update state with critique results
    """
    # Validate we have briefs to critique
    try:
        index, brief = _get_brief_to_review(state)
    except ValidationError as e:
        logger.warning(f"[critic] {e}")
        return _build_error_patch(state, str(e), kind="warning")

    # Check revision status
    at_max_revisions = _is_at_max_revisions(state, brief.idea_id)

    # Invoke LLM
    try:
        critique = _invoke_llm(state)
    except (LLMError, ValidationError) as e:
        logger.error(f"[critic] {e}")
        return _build_error_patch(
            state, "Critic LLM invocation failed; keeping previous state.", kind="warning"
        )

    # Guarantee idea_id matches the reviewed brief
    critique.idea_id = brief.idea_id

    # Handle max revisions case
    if at_max_revisions:
        critique = _handle_max_revisions(critique, state)

    return _build_success_patch(state, critique, at_max_revisions)
