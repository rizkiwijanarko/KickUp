"""
VentureForge LLM Prompts
========================
Loads agent prompts from PROMPTS.md (markdown-prompt convention).

Usage:
    from src.llm.prompts import get_prompt
    prompt = get_prompt("pain_point_miner")

PROMPTS.md format (each prompt is an H2 block):
    ## pain_point_miner
    You are a Pain Point Miner. Your job is to...
    ...

    ## idea_generator
    ...
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.prompts import ChatPromptTemplate


#
# In-memory cache loaded once on first access.
#
_PROMPTS: dict[str, str] = {}

# Known agent prompt sections (H2 headings that are actual agent IDs)
_AGENT_IDS: set[str] = {
    "pain_point_miner",
    "idea_generator",
    "scorer",
    "pitch_writer",
    "critic",
}


def _load_prompts() -> dict[str, str]:
    """Parse PROMPTS.md into {section_name: prompt_text}."""
    # Walk up from this file to find PROMPTS.md
    search_dir = Path(__file__).resolve().parent.parent.parent
    prompts_file = search_dir / "PROMPTS.md"

    if not prompts_file.exists():
        return {}

    text = prompts_file.read_text(encoding="utf-8")
    # Build a regex that only matches H2 headings for known agent IDs.
    # This avoids splitting on sub-headings like ## Input inside a prompt.
    agent_pattern = "|".join(re.escape(a) for a in _AGENT_IDS)
    pattern = re.compile(rf"^##\s+({agent_pattern})\s*\n", re.MULTILINE)
    parts = pattern.split(text)
    # parts[0] = preamble (before first ##), parts[1] = name, parts[2] = body, ...
    prompts: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip().lower()
        body = parts[i + 1].strip()
        prompts[name] = body
    return prompts


def get_prompt(name: str) -> str:
    """Return raw prompt text for the given agent name."""
    if not _PROMPTS:
        _PROMPTS.update(_load_prompts())
    return _PROMPTS.get(name, f"# {name}\nNo prompt configured.")


def all_prompt_names() -> list[str]:
    """Return list of loaded prompt names."""
    if not _PROMPTS:
        _PROMPTS.update(_load_prompts())
    return sorted(_PROMPTS.keys())
