"""
VentureForge LLM Client
========================
Provider-agnostic OpenAI-compatible factory.
Switch LLM provider by changing LLM_BASE_URL / LLM_API_KEY in env.

Usage:
    from src.llm.client import get_llm
    llm = get_llm(temperature=0.1)
    response = llm.invoke("Hello")
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import settings


def get_llm(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """
    Return a configured LLM instance.

    Args:
        temperature: Override default temperature (0.0–2.0).
        max_tokens: Override default max_tokens.
        model: Override default model name.
    """
    config = settings.effective_llm_config
    return ChatOpenAI(
        base_url=config["base_url"],
        api_key=config["api_key"] or "sk-dummy",  # prevent KeyError if empty
        model=model or config["model"],
        temperature=temperature if temperature is not None else settings.default_temperature,
        max_tokens=max_tokens or settings.max_tokens,
        timeout=config["timeout"],
    )


@lru_cache(maxsize=32)
def get_structured_llm(
    output_schema: type,
    *,
    temperature: float | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """
    Return an LLM configured with a Pydantic output schema for structured generation.

    Args:
        output_schema: A Pydantic v2 BaseModel subclass describing the desired output.
        temperature: Override default temperature.
        model: Override default model name.
    """
    base = get_llm(temperature=temperature, model=model)
    return base.with_structured_output(output_schema)
