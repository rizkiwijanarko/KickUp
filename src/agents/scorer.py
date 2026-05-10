"""Scorer — evaluates ideas with a binary yes/no rubric."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import coerce_rubric_bools, extract_json, get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    DemandRubric,
    FatalFlaw,
    FeasibilityRubric,
    NoveltyRubric,
    PipelineStage,
    ScoredIdea,
    VentureForgeState,
)

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return get_prompt("scorer")


def _build_user_prompt(state: VentureForgeState) -> str:
    # If in revision mode, only score the specific idea being revised
    if state.current_revision_idea_id:
        ideas_to_score = [idea for idea in state.ideas if idea.id == state.current_revision_idea_id]
        if not ideas_to_score:
            # Idea was removed, return empty prompt
            return "No ideas to score."
    else:
        # Initial scoring: score all ideas
        ideas_to_score = state.ideas
    
    ideas_blobs = [
        {
            "id": str(idea.id),
            "title": idea.title,
            "one_liner": idea.one_liner,
            "problem": idea.problem,
            "solution": idea.solution,
            "target_user": idea.target_user,
        }
        for idea in ideas_to_score
    ]
    
    # Sort pain points by evidence count (descending) to prioritize well-validated pain points
    sorted_pps = sorted(
        state.filtered_pain_points,
        key=lambda pp: len(pp.evidence),
        reverse=True
    )
    pp_blobs = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
        }
        for pp in sorted_pps
    ]

    user_text = (
        f"Domain: {state.domain}\n\n"
        f"IDEAS TO SCORE:\n{json.dumps(ideas_blobs, indent=2)}\n\n"
        f"SUPPORTING PAIN POINTS:\n{json.dumps(pp_blobs, indent=2)}\n\n"
        "Evaluate each idea according to the binary rubric. "
        "Return a JSON array of scored ideas."
    )
    return user_text


def _invoke_llm(state: VentureForgeState) -> list[dict]:
    # Use reasoning=False to disable thinking mode for structured JSON output
    llm = get_llm(temperature=0.1, max_tokens=16384, reasoning=False)
    
    # Add explicit JSON-only instruction
    system_prompt = _build_system_prompt()
    system_prompt += "\n\n**CRITICAL: Output ONLY the JSON array. No markdown code fences, no explanations, no preamble. Start with [ and end with ].**"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_user_prompt(state)),
    ]

    start = time.monotonic()
    try:
        raw = llm.invoke(messages)
        content = raw.content if hasattr(raw, "content") else str(raw)
    except Exception as e:
        logger.error(f"[scorer] LLM invocation failed: {e}")
        return []

    logger.info(f"[scorer] LLM responded in {time.monotonic()-start:.1f}s")
    
    # Debug: log response preview
    logger.info(f"[scorer] Response preview (first 500 chars): {content[:500]}")

    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[scorer] JSON extraction failed. Response length: {len(content)} chars")
        logger.error(f"[scorer] Full response (first 2000 chars): {content[:2000]}")
        return []
    if isinstance(parsed, dict) and "scored_ideas" in parsed:
        return parsed["scored_ideas"]
    return parsed if isinstance(parsed, list) else []


def run(state: VentureForgeState) -> dict:
    if not state.ideas:
        logger.warning("[scorer] no ideas to score")
        return {
            "scored_ideas": [],
            "current_stage": PipelineStage.SCORING,
            "next_node": "orchestrator",
        }

    raw_scores = _invoke_llm(state)
    scored_ideas: list[ScoredIdea] = []

    for raw in raw_scores:
        try:
            # Coerce "yes"/"no" strings to bools and re-calculate yes_count
            f_rubric = FeasibilityRubric(**coerce_rubric_bools(raw["feasibility_rubric"]))
            d_rubric = DemandRubric(**coerce_rubric_bools(raw["demand_rubric"]))
            n_rubric = NoveltyRubric(**coerce_rubric_bools(raw["novelty_rubric"]))

            yes_count = sum([
                f_rubric.can_be_solved_manually_first,
                f_rubric.has_schlep_or_unsexy_advantage,
                f_rubric.can_2_3_person_team_build_mvp_in_6_months,
                d_rubric.addresses_at_least_2_pain_points,
                d_rubric.is_painkiller_not_vitamin,
                d_rubric.has_clear_vein_of_early_adopters,
                n_rubric.differentiated_from_current_behavior,
                n_rubric.has_path_out_of_niche,
            ])

            raw_flaws = raw.get("fatal_flaws", [])
            fatal_flaws = [FatalFlaw(**f) for f in raw_flaws if isinstance(f, dict)]

            # LLM may return either 'id' (echoing input) or 'idea_id'
            idea_id = raw.get("idea_id") or raw.get("id")
            if not idea_id:
                continue

            scored = ScoredIdea(
                idea_id=idea_id,
                reasoning_trace=raw.get("reasoning_trace", ""),
                feasibility_rubric=f_rubric,
                demand_rubric=d_rubric,
                novelty_rubric=n_rubric,
                core_assumption=raw["core_assumption"],
                fatal_flaws=fatal_flaws,
                yes_count=yes_count,
                verdict=raw["verdict"],
                one_risk=raw["one_risk"],
            )
            scored_ideas.append(scored)
        except Exception as e:
            logger.debug(f"[scorer] skipping malformed scored idea: {e}")
            continue

    # Rank ideas
    scored_ideas.sort(key=lambda s: s.yes_count, reverse=True)
    for i, s in enumerate(scored_ideas):
        s.rank = i + 1

    # Merge with existing scores if in revision mode
    if state.current_revision_idea_id:
        # Remove old score for the revised idea
        existing_scores = [s for s in state.scored_ideas if s.idea_id != state.current_revision_idea_id]
        # Add new score
        all_scores = existing_scores + scored_ideas
        # Re-rank all scores
        all_scores.sort(key=lambda s: s.yes_count, reverse=True)
        for i, s in enumerate(all_scores):
            s.rank = i + 1
        
        logger.info(
            f"[scorer] Revision mode: re-scored idea {state.current_revision_idea_id}. "
            f"Total scores: {len(all_scores)}"
        )
    else:
        # Initial scoring: use new scores
        all_scores = scored_ideas

    # Verdict counts for logging
    pursue = sum(1 for s in all_scores if s.verdict == "pursue")
    explore = sum(1 for s in all_scores if s.verdict == "explore")
    park = sum(1 for s in all_scores if s.verdict == "park")

    patch = {
        "scored_ideas": all_scores,
        "scorer_attempts": state.scorer_attempts + 1,
        "current_revision_idea_id": None,  # Clear revision flag
        "next_node": "orchestrator",
    }
    patch.update(
        state.add_event(
            agent="scorer",
            stage=PipelineStage.SCORING,
            kind="info",
            message=(
                f"Scored {len(scored_ideas)} ideas → "
                f"{pursue} pursue / {explore} explore / {park} park."
            ),
        )
    )
    return patch
