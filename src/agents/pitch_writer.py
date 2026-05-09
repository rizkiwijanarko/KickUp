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
    # If we're in revision mode for a specific idea, only write that pitch
    if state.current_revision_idea_id:
        # Find the specific scored idea being revised
        target_scored = next(
            (s for s in state.scored_ideas if s.idea_id == state.current_revision_idea_id),
            None
        )
        if not target_scored:
            # Fallback to top ideas if we can't find the specific one
            top_ideas = state.top_scored_ideas
        else:
            top_ideas = [target_scored]
    else:
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
            "evidence": [
                {
                    "source_url": ev.source_url,
                    "raw_quote": ev.raw_quote,
                    "source": ev.source.value,
                }
                for ev in pp.evidence
            ],
            "evidence_count": len(pp.evidence),
        }
        for pp in sorted_pps
    ]

    feedback = state.revision_feedback or "None"

    # If revision feedback exists and the Critic targeted the
    # pitch_writer, we want the LLM to explicitly fix the failing
    # rubric checks (tagline length, unscalable_acquisition_concrete,
    # gtm_leads_with_manual_recruitment) without changing the core
    # idea or evidence.
    revision_block = ""
    if state.revision_feedback:
        # Optionally, look at the most recent critique for extra
        # context, if available.
        last_crit = state.critiques[-1] if state.critiques else None
        failing = ", ".join(last_crit.failing_checks) if last_crit else "(see feedback)"
        revision_block = (
            "THIS IS A REVISION ROUND for the pitch briefs. The critic "
            "flagged issues in the pitch writing (e.g., tagline length, "
            "unscalable acquisition, or go-to-market style). You MUST "
            "fix the following before returning new briefs:\n"  # noqa: E501
            f"- Critic failing checks: {failing}\n"
            f"- Critic feedback: {feedback}\n\n"
            "Do NOT change the underlying idea, evidence_links, or core "
            "assumptions. Only rewrite the pitch fields (tagline, "
            "go_to_market, business_model, etc.) so that they satisfy the "
            "rubric while staying truthful to the evidence.\n\n"
        )

    user_text = (
        f"Domain: {state.domain}\n\n"
        f"SCORED IDEAS (Top {len(scored_blobs)}):\n{json.dumps(scored_blobs, indent=2)}\n\n"
        f"SUPPORTING PAIN POINTS:\n{json.dumps(pp_blobs, indent=2)}\n\n"
        f"{revision_block}"
        "Write full pitch briefs for these ideas. Return a JSON array of pitch briefs."
    )
    return user_text


def _invoke_llm(state: VentureForgeState) -> list[dict]:
    # Pitch briefs are long (~6K tokens per brief x 3 briefs = ~18K tokens)
    # Set max_tokens to 8192 to avoid truncation
    llm = get_llm(temperature=0.4, max_tokens=8192, reasoning=False)
    
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
        logger.error(f"[pitch_writer] LLM invocation failed: {e}")
        return []

    logger.info(f"[pitch_writer] LLM responded in {time.monotonic()-start:.1f}s")
    
    # Debug: log response preview and check for truncation
    logger.info(f"[pitch_writer] Response preview (first 500 chars): {content[:500]}")
    logger.info(f"[pitch_writer] Response length: {len(content)} chars")
    
    # Warn if response looks truncated (doesn't end with ] or })
    if content and not content.rstrip().endswith((']', '}')):
        logger.warning(
            f"[pitch_writer] Response may be truncated (doesn't end with ] or }}). "
            f"Last 100 chars: {content[-100:]}"
        )

    parsed = extract_json(content)
    if parsed is None:
        logger.error(f"[pitch_writer] JSON extraction failed. Response length: {len(content)} chars")
        logger.error(f"[pitch_writer] Full response (first 2000 chars): {content[:2000]}")
        return []

    if isinstance(parsed, dict) and "pitch_briefs" in parsed:
        return parsed["pitch_briefs"]
    return parsed if isinstance(parsed, list) else []


def _collect_evidence_urls(idea_id: str, state: VentureForgeState) -> list[str]:
    """
    Collect all evidence URLs from pain points referenced by this idea.
    Fallback for when LLM fails to provide evidence_links.
    """
    urls = []
    idea = next((i for i in state.ideas if str(i.id) == str(idea_id)), None)
    if not idea:
        return urls
    
    for pp_id in idea.addresses_pain_point_ids:
        pp = next((p for p in state.pain_points if str(p.id) == str(pp_id)), None)
        if pp and hasattr(pp, 'evidence') and pp.evidence:
            for ev in pp.evidence:
                if ev.source_url and ev.source_url not in urls:
                    urls.append(ev.source_url)
    
    return urls


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
            
            # Validate and fix evidence_links
            if not brief.evidence_links or len(brief.evidence_links) < 2:
                logger.warning(
                    f"[pitch_writer] LLM provided {len(brief.evidence_links)} evidence links for idea {brief.idea_id}, "
                    "collecting from pain points"
                )
                collected_urls = _collect_evidence_urls(brief.idea_id, state)
                if collected_urls:
                    brief.evidence_links = collected_urls
                    logger.info(
                        f"[pitch_writer] Collected {len(collected_urls)} evidence URLs from pain points for idea {brief.idea_id}"
                    )
            
            briefs.append(brief)
        except Exception as e:
            logger.debug(f"[pitch_writer] skipping malformed pitch brief: {e}")
            continue

    # Merge with existing briefs if in revision mode
    if state.current_revision_idea_id:
        # In revision mode: merge the new brief with existing briefs
        all_briefs = state.pitch_briefs + briefs
    else:
        # Initial generation: replace all briefs
        all_briefs = briefs

    patch = {
        "pitch_briefs": all_briefs,
        "current_revision_idea_id": None,  # Clear revision flag
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
