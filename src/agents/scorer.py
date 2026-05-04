"""Scorer — evaluates ideas with a binary yes/no rubric."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.client import get_llm
from src.llm.prompts import get_prompt
from src.state.schema import (
    DemandRubric,
    FeasibilityRubric,
    NoveltyRubric,
    PipelineStage,
    ScoredIdea,
    VentureForgeState,
    Verdict,
)

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    return get_prompt("scorer")


def _build_user_prompt(state: VentureForgeState) -> str:
    ideas_blobs = [
        {
            "id": str(idea.id),
            "title": idea.title,
            "one_liner": idea.one_liner,
            "problem": idea.problem,
            "solution": idea.solution,
            "target_user": idea.target_user,
        }
        for idea in state.ideas
    ]
    
    # Also provide pain points for demand validation
    pps = state.filtered_pain_points
    pp_blobs = [
        {
            "id": str(pp.id),
            "title": pp.title,
            "description": pp.description,
        }
        for pp in pps
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
    llm = get_llm(temperature=0.1, max_tokens=4096)
    messages = [
        SystemMessage(content=_build_system_prompt()),
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

    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "scored_ideas" in parsed:
            return parsed["scored_ideas"]
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError as e:
        logger.error(f"[scorer] JSON parse error: {e}")
        return []


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
            # Re-calculate yes_count to ensure it's accurate
            f_rubric = FeasibilityRubric(**raw["feasibility_rubric"])
            d_rubric = DemandRubric(**raw["demand_rubric"])
            n_rubric = NoveltyRubric(**raw["novelty_rubric"])
            
            yes_count = sum([
                f_rubric.can_2_3_person_team_build_mvp_in_6_months,
                f_rubric.uses_only_existing_proven_tech,
                f_rubric.no_special_regulatory_requirements,
                d_rubric.addresses_at_least_2_pain_points,
                d_rubric.is_painkiller_not_vitamin,
                d_rubric.target_user_clearly_defined,
                n_rubric.differentiated_from_current_behavior,
                n_rubric.leverages_unique_insight,
            ])

            scored = ScoredIdea(
                idea_id=raw["idea_id"],
                feasibility_rubric=f_rubric,
                demand_rubric=d_rubric,
                novelty_rubric=n_rubric,
                core_assumption=raw["core_assumption"],
                fatal_flaws=raw.get("fatal_flaws", []),
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

    return {
        "scored_ideas": scored_ideas,
        "current_stage": PipelineStage.SCORING,
        "next_node": "orchestrator",
    }
