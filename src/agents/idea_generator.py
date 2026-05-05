"""Idea Generator — clusters pain points into themes and generates distinct
startup ideas.

Pipeline flow
=============
1. Receive filtered pain points from state.
2. Cap to a manageable number (~50) to fit context window.
3. LLM brainstorms ideas grouped by themes, returns structured JSON.
4. Code validates that every ``addresses_pain_point_ids`` references a real
   pain point UUID that was present in the input.
5. Return validated Idea objects.
"""
from __future__ import annotations

import json
import logging
import time
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import Idea, PipelineStage, VentureForgeState

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Tunables
# ------------------------------------------------------------------
_MAX_PAIN_POINTS_CONTEXT: int = 50  # avoid blowing context window
_IDEAS_PER_RUN_DEFAULT: int = 5


def _build_system_prompt() -> str:
    return get_prompt("idea_generator")


def _build_user_prompt(state: VentureForgeState) -> str:
    # Cap pain points for context window safety
    pps = state.filtered_pain_points[:_MAX_PAIN_POINTS_CONTEXT]
    domain = state.domain
    count = state.ideas_per_run or _IDEAS_PER_RUN_DEFAULT
    feedback = state.revision_feedback or "None"

    # Serialize pain points
    pp_blobs: list[dict] = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
            "raw_quote": pp.raw_quote,
            "source_url": pp.source_url,
        }
        for pp in pps
    ]

    # If revision feedback exists, this run is part of the reflection
    # loop (typically triggered by the Critic for positioning issues
    # such as target_is_contained_fire or competition_embraced_with_thesis).
    # Make that explicit in the prompt so the LLM focuses on fixing
    # those weaknesses first.
    revision_block = ""
    if state.revision_feedback:
        revision_block = (
            "THIS IS A REVISION ROUND. The critic flagged weaknesses in "
            "positioning (e.g., target user not a contained community, "
            "or weak competitive thesis). You MUST address the following "
            "feedback before generating ideas:\n"  # noqa: E501
            f"- Critic feedback: {feedback}\n\n"
            "In your new ideas, make the target_user a specific, named, "
            "reachable community (a 'contained fire') and make the "
            "competition thesis explicit: what are users doing today and "
            "what incumbents are afraid to do.\n\n"
        )

    user_text = (
        f"Domain: {domain}\n"
        f"Ideas to generate: {count}\n\n"
        f"PAIN POINTS ({len(pps)} provided):\n"
        f"{json.dumps(pp_blobs, indent=2)}\n\n"
        f"{revision_block}"
        "Generate the exact number of ideas requested. "
        "Each idea must reference at least 2 pain point UUIDs from the list above. "
        "Return JSON: {\"ideas\": [ ... ]}. "
        "Do not invent UUIDs — only use the ids in the pain points array."
    )
    return user_text


def _invoke_llm(state: VentureForgeState) -> list[dict]:
    """Call LLM, parse JSON, return raw idea dicts."""
    llm = get_llm(temperature=0.7, max_tokens=4096, reasoning=False)
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[idea_generator] LLM invocation failed after {time.monotonic()-start:.1f}s: {e}")
        return []

    logger.info(f"[idea_generator] LLM responded in {time.monotonic()-start:.1f}s")

    parsed = extract_json(content)
    if parsed is None:
        logger.error("[idea_generator] JSON extraction failed")
        return []

    ideas = parsed.get("ideas") if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
    if not isinstance(ideas, list):
        logger.warning("[idea_generator] LLM did not return an array of ideas")
        return []
    return ideas


def _validate_idea(raw: dict, valid_ids: set[UUID]) -> Idea | None:
    """Return an Idea if it references real pain points, else None."""
    # Parse UUID references
    raw_ids = raw.get("addresses_pain_point_ids", [])
    resolved: list[UUID] = []
    for rid in raw_ids:
        try:
            uid = UUID(str(rid))
            if uid in valid_ids:
                resolved.append(uid)
        except (ValueError, TypeError):
            continue

    if len(resolved) < 2:
        logger.debug(
            f"[idea_generator] REJECTED — idea '{raw.get('title', '?')}' "
            f"references only {len(resolved)} valid pain point(s)"
        )
        return None

    try:
        idea = Idea(
            id=uuid4(),
            title=raw["title"],
            one_liner=raw["one_liner"],
            problem=raw["problem"],
            solution=raw["solution"],
            target_user=raw["target_user"],
            key_features=raw.get("key_features", []),
            addresses_pain_point_ids=resolved,
        )
        return idea
    except Exception as e:
        logger.debug(f"[idea_generator] REJECTED — malformed idea: {e}")
        return None


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def run(state: VentureForgeState) -> dict:
    pps = state.filtered_pain_points
    if not pps:
        logger.warning("[idea_generator] no pain points available — returning empty")
        patch = {
            "ideas": [],
            "current_stage": PipelineStage.GENERATING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="idea_generator",
                stage=PipelineStage.GENERATING,
                kind="warning",
                message="No pain points available to generate ideas from.",
            )
        )
        return patch

    valid_ids = {pp.id for pp in pps}
    raw_ideas = _invoke_llm(state)
    logger.info(f"[idea_generator] LLM produced {len(raw_ideas)} raw ideas")

    validated: list[Idea] = []
    for raw in raw_ideas:
        idea = _validate_idea(raw, valid_ids)
        if idea:
            validated.append(idea)

    count = state.ideas_per_run or _IDEAS_PER_RUN_DEFAULT
    final = validated[:count]

    logger.info(
        f"[idea_generator] {len(final)}/{len(raw_ideas)} ideas validated "
        f"(required <= {count} with >=2 real pain point refs each)"
    )

    patch = {
        "ideas": final,
        "current_stage": PipelineStage.GENERATING,
        "next_node": "orchestrator",
    }
    patch.update(
        state.add_event(
            agent="idea_generator",
            stage=PipelineStage.GENERATING,
            kind="info",
            message=(
                f"Generated {len(final)} ideas (requested {count}) "
                f"addressing ≥2 validated pain points each."
            ),
        )
    )
    return patch
