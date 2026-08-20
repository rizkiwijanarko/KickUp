"""
Provider protocol and evidence models for data mining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.models.common import DataSource


@dataclass(frozen=True)
class RawEvidence:
    """A single piece of raw text evidence extracted by a mining provider."""

    text: str
    url: str
    source: DataSource
    title: str = ""
    author: str = ""
    score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SourceProvider(Protocol):
    """Protocol for data source scrapers and search providers."""

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    def source_type(self) -> DataSource:
        """Domain DataSource enum mapping."""
        ...

    def is_available(self) -> bool:
        """Check if provider credentials and network prerequisites are met."""
        ...

    def fetch(self, domain: str, limit: int = 50) -> list[RawEvidence]:
        """Fetch raw complaint/pain point evidence for a domain."""
        ...
