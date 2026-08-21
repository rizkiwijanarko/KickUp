"""
VentureForge LLM Provider Adapters
====================================
Each adapter encapsulates the extra parameters needed to talk to one
provider.  The LLM factory in ``client.py`` selects the first adapter
whose ``matches()`` returns True for the configured base URL.

Adding a new provider = adding one new class here.  No edits to the
factory or to any agent.

Glossary (from codebase-design vocabulary)
    adapter  – a concrete thing that satisfies the interface at a seam
    seam     – ``get_llm()`` in client.py; the boundary callers cross
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProviderAdapter(ABC):
    """Interface that every provider adapter must satisfy.

    ``matches(base_url)`` — decides ownership.
    ``extra_params(base_params, reasoning)`` — returns provider-specific
    dict to merge into the base ChatOpenAI kwargs.  Must never mutate
    ``base_params`` in-place.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging."""
        ...

    @abstractmethod
    def matches(self, base_url: str) -> bool:
        """Return True if this adapter handles the given base URL."""
        ...

    @abstractmethod
    def extra_params(
        self,
        base_params: dict[str, Any],
        reasoning: bool,
    ) -> dict[str, Any]:
        """Return provider-specific parameter overrides.

        The caller will ``base_params | extra_params(...)`` — so only
        return the keys that differ from the base defaults.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class OpenAIAdapter(LLMProviderAdapter):
    """Standard OpenAI API — no extra parameters required."""

    @property
    def name(self) -> str:
        return "openai"

    def matches(self, base_url: str) -> bool:
        return "openai.com" in base_url.lower()

    def extra_params(
        self,
        base_params: dict[str, Any],
        reasoning: bool,
    ) -> dict[str, Any]:
        return {}


class OpenRouterAdapter(LLMProviderAdapter):
    """OpenRouter — adds the required identification headers.

    OpenRouter requires ``HTTP-Referer`` and ``X-Title`` to route
    requests correctly and attribute usage to the project.
    """

    _REFERER = "https://github.com/rizkiwijanarko/KickUp"
    _TITLE = "VentureForge"

    @property
    def name(self) -> str:
        return "openrouter"

    def matches(self, base_url: str) -> bool:
        return "openrouter.ai" in base_url.lower()

    def extra_params(
        self,
        base_params: dict[str, Any],
        reasoning: bool,
    ) -> dict[str, Any]:
        return {
            "default_headers": {
                "HTTP-Referer": self._REFERER,
                "X-Title": self._TITLE,
            }
        }


# ---------------------------------------------------------------------------
# Registry — first match wins, so more specific adapters go first
# ---------------------------------------------------------------------------

PROVIDER_ADAPTERS: list[LLMProviderAdapter] = [
    OpenRouterAdapter(),
    OpenAIAdapter(),
]


def get_adapter(base_url: str) -> LLMProviderAdapter | None:
    """Return the first registered adapter that matches ``base_url``.

    Returns ``None`` if no adapter matches; the factory will use bare
    base params without provider-specific extras.
    """
    for adapter in PROVIDER_ADAPTERS:
        if adapter.matches(base_url):
            return adapter
    return None
