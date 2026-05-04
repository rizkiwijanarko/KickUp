"""Critic — adversarial reviewer evaluating pitch briefs with binary rubric."""
from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    CriticRubric,
    CriticStatus,
    Critique,
    PipelineStage,
    TargetAgent,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return get_prompt("critic")


def _build_user_prompt(state: VentureForgeState) -> str:
    # Critique the first pitch brief that isn't approved yet (simplified for now)
    # In a full multi-pitch system, we'd critique all or the best one.
    if not state.pitch_briefs:
        return "No pitch briefs to review."

    brief = state.pitch_briefs[0]
    
    user_text = (
        f"Domain: {state.domain}\n"
        f"Current Revision Count: {state.revision_count}\n\n"
        f"PITCH BRIEF TO REVIEW:\n{brief.markdown_content}\n\n"
        "Provide a brutal, honest critique using the binary rubric. "
        "If it fails any check, specify which worker should fix it."
    )
    return user_text


def _invoke_llm(state: VentureForgeState) -> dict:
    llm = get_llm(temperature=0.2, max_tokens=2048)
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

    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"[critic] JSON parse error: {e}")
        return {}


def run(state: VentureForgeState) -> dict:
    if not state.pitch_briefs:
        logger.warning("[critic] no pitch briefs to critique")
        return {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }

    # Auto-approve if max revisions reached
    if state.revision_count >= state.max_revisions:
        rubric = CriticRubric(
            all_claims_evidence_backed=True,
            no_hallucinated_source_urls=True,
            target_user_is_specific_person=True,
            competitive_analysis_includes_behavior=True,
            honest_risk_disclosure=True,
            discovery_questions_are_open_ended=True,
            tagline_under_12_words=True,
            go_to_market_concrete=True,
        )
        critique = Critique(
            idea_id=state.pitch_briefs[0].idea_id,
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

    raw = _invoke_llm(state)
    if not raw:
        # Fallback to simple revision if LLM fails
        return {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }

    try:
        rubric = CriticRubric(**raw["rubric"])
        critique = Critique(
            idea_id=state.pitch_briefs[0].idea_id,
            rubric=rubric,
            all_pass=raw["all_pass"],
            approval_status=raw["approval_status"],
            failing_checks=raw.get("failing_checks", []),
            target_agent=TargetAgent(raw["target_agent"]),
            revision_feedback=raw["revision_feedback"],
        )
    except Exception as e:
        logger.error(f"[critic] malformed critique: {e}")
        return {
            "current_stage": PipelineStage.CRITIQUING,
            "next_node": "orchestrator",
        }

    return {
        "critique": critique,
        "current_stage": PipelineStage.CRITIQUING,
        "next_node": "orchestrator",
    }
