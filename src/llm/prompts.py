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


def _load_prompts() -> dict[str, str]:
    """Parse PROMPTS.md into {section_name: prompt_text}."""
    # Walk up from this file to find PROMPTS.md
    search_dir = Path(__file__).resolve().parent.parent.parent
    prompts_file = search_dir / "PROMPTS.md"

    if not prompts_file.exists():
        return {}

    text = prompts_file.read_text(encoding="utf-8")
    # Split on H2 headers
    pattern = re.compile(r"^##\s+(\w[\w_\-]*)\s*\n", re.MULTILINE)
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
