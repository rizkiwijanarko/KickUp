"""
Domain models for extracted market pain points and evidence.
"""

from __future__ import annotations

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, computed_field

from src.models.common import DataSource


class PainPointRubric(BaseModel):
    """Binary rubric applied by the Pain Point Miner to self-filter output."""

    is_genuine_current_frustration: bool
    has_verbatim_quote: bool
    user_segment_specific: bool

    @computed_field
    @property
    def all_pass(self) -> bool:
        return all(self.model_dump(exclude={"all_pass"}).values())


class PainPointEvidence(BaseModel):
    """A single piece of grounded evidence supporting a pain point."""

    source_url: str = Field(..., min_length=5)
    raw_quote: str = Field(..., min_length=5)
    source: DataSource
    # Composite engagement score of the source comment/post (0 if unknown).
    score: int = 0


class PainPoint(BaseModel):
    """A structured user pain point with grounded evidence sources."""

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=500)
    rubric: PainPointRubric
    passes_rubric: bool

    # Multiple evidence sources (1-10 per pain point)
    evidence: list[PainPointEvidence] = Field(min_length=1, max_length=10)

    # Computed properties for backward compatibility
    @property
    def source_url(self) -> str:
        """Primary source URL (first evidence item)."""
        return self.evidence[0].source_url if self.evidence else ""

    @property
    def raw_quote(self) -> str:
        """Primary quote (first evidence item)."""
        return self.evidence[0].raw_quote if self.evidence else ""

    @property
    def source(self) -> DataSource:
        """Primary source (first evidence item)."""
        return self.evidence[0].source if self.evidence else DataSource.WEB

    @property
    def evidence_count(self) -> int:
        """Number of evidence sources."""
        return len(self.evidence)

    @computed_field
    @property
    def strength(self) -> int:
        """Aggregate engagement strength across all evidence (sum of scores)."""
        return sum(ev.score for ev in self.evidence)
