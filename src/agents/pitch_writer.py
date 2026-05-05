"""Pitch Writer — writes investor-ready one-page pitch briefs for top ideas."""
from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    PipelineStage,
    PitchBrief,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return get_prompt("pitch_writer")


def _build_user_prompt(state: VentureForgeState) -> str:
    top_ideas = state.top_scored_ideas
    ideas_map = {str(idea.id): idea for idea in state.ideas}
    
    scored_blobs = []
    for s in top_ideas:
        idea = ideas_map.get(str(s.idea_id))
        if not idea:
            continue
        scored_blobs.append({
            "idea_id": str(s.idea_id),
            "title": idea.title,
            "one_liner": idea.one_liner,
            "problem": idea.problem,
            "solution": idea.solution,
            "target_user": idea.target_user,
            "key_features": idea.key_features,
            "yes_count": s.yes_count,
            "core_assumption": s.core_assumption,
            "fatal_flaws": [f.model_dump() for f in s.fatal_flaws],
            "one_risk": s.one_risk,
        })

    pps = state.filtered_pain_points
    pp_blobs = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
            "source_url": pp.source_url,
        }
        for pp in pps
    ]

    user_text = (
        f"Domain: {state.domain}\n\n"
        f"SCORED IDEAS (Top {len(scored_blobs)}):\n{json.dumps(scored_blobs, indent=2)}\n\n"
        f"SUPPORTING PAIN POINTS:\n{json.dumps(pp_blobs, indent=2)}\n\n"
        f"Revision feedback (if any): {state.revision_feedback or 'None'}\n\n"
        "Write full pitch briefs for these ideas. Return a JSON array of pitch briefs."
    )
    return user_text


def _invoke_llm(state: VentureForgeState) -> list[dict]:
    llm = get_llm(temperature=0.4, max_tokens=4096, reasoning=False)
    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[pitch_writer] LLM invocation failed: {e}")
        return []

    logger.info(f"[pitch_writer] LLM responded in {time.monotonic()-start:.1f}s")

    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[pitch_writer] JSON extraction failed (content len={len(content)}, first 80={content[:80]!r})")
        return []

    if isinstance(parsed, dict) and "pitch_briefs" in parsed:
        return parsed["pitch_briefs"]
    return parsed if isinstance(parsed, list) else []


def run(state: VentureForgeState) -> dict:
    if not state.scored_ideas:
        logger.warning("[pitch_writer] no scored ideas to write briefs for")
        patch = {
            "pitch_briefs": [],
            "current_stage": PipelineStage.WRITING,
            "next_node": "orchestrator",
        }
        patch.update(
            state.add_event(
                agent="pitch_writer",
                stage=PipelineStage.WRITING,
                kind="warning",
                message="No scored ideas available for writing pitch briefs.",
            )
        )
        return patch

    raw_briefs = _invoke_llm(state)
    briefs: list[PitchBrief] = []

    for raw in raw_briefs:
        try:
            brief = PitchBrief(
                idea_id=raw["idea_id"],
                title=raw["title"],
                tagline=raw["tagline"],
                problem=raw["problem"],
                solution=raw["solution"],
                target_user=raw["target_user"],
                market_opportunity=raw["market_opportunity"],
                business_model=raw["business_model"],
                go_to_market=raw["go_to_market"],
                key_risk=raw["key_risk"],
                next_steps="\n".join(raw["next_steps"]) if isinstance(raw["next_steps"], list) else raw["next_steps"],
                evidence_links=raw.get("evidence_links", []),
                markdown_content=raw["markdown_content"],
                revision_count=state.get_revision_count(raw["idea_id"]),
            )
            briefs.append(brief)
        except Exception as e:
            logger.debug(f"[pitch_writer] skipping malformed pitch brief: {e}")
            continue

    patch = {
        "pitch_briefs": briefs,
        "current_stage": PipelineStage.WRITING,
        "next_node": "orchestrator",
    }
    patch.update(
        state.add_event(
            agent="pitch_writer",
            stage=PipelineStage.WRITING,
            kind="info",
            message=f"Wrote {len(briefs)} pitch briefs for top scored ideas.",
        )
    )
    return patch
