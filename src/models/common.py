"""
Common domain enumerations and execution event models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Origin of a pain point or research evidence."""

    REDDIT = "reddit"
    HACKERNEWS = "hackernews"
    PRODUCTHUNT = "producthunt"
    WEB = "web"
    YOUTUBE = "youtube"


class Verdict(str, Enum):
    """Scorer recommendation for an idea."""

    PURSUE = "pursue"
    EXPLORE = "explore"
    PARK = "park"


class TargetAgent(str, Enum):
    """Target agent for a reflection revision cycle."""

    PAIN_POINT_MINER = "pain_point_miner"
    IDEA_GENERATOR = "idea_generator"
    PITCH_WRITER = "pitch_writer"


class PipelineStage(str, Enum):
    """Execution lifecycle stage of the discovery pipeline."""

    IDLE = "idle"
    MINING = "mining"
    GENERATING = "generating"
    SCORING = "scoring"
    WRITING = "writing"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunEvent(BaseModel):
    """High-level event emitted during a pipeline run for UI/logging."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    stage: PipelineStage
    kind: Literal["info", "warning", "error"] = "info"
    message: str
    idea_id: UUID | None = None


class ErrorEntry(BaseModel):
    """Structured error log entry."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    error: str
