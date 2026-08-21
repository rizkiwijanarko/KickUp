"""
LangGraph State Schema for VentureForge Pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from src.models import (
    CompetitiveLandscape,
    Critique,
    CritiqueRubric,
    DataSource,
    DemandRubric,
    FatalFlaw,
    FeasibilityRubric,
    Idea,
    NoveltyRubric,
    PainPoint,
    PainPointEvidence,
    PainPointRubric,
    PipelineStage,
    PitchBrief,
    RunEvent,
    ScoredIdea,
    TargetAgent,
    ValidationPlan,
    Verdict,
)


class VentureForgeState(BaseModel):
    """
    Shared state container passed across LangGraph pipeline nodes.
    Pure immutable state updates are produced by returning patch dictionaries.
    """

    # -----------------------------------------------------------------
    # Input / Run configuration
    # -----------------------------------------------------------------
    domain: str = Field(..., min_length=2, max_length=100)
    max_pain_points: int = Field(default=30, ge=5, le=100)
    ideas_per_run: int = Field(default=5, ge=1, le=20)
    top_n_pitches: int = Field(default=3, ge=1, le=10)
    max_revisions: int = Field(default=3, ge=0, le=5)

    # -----------------------------------------------------------------
    # Pipeline data (populated by worker agents)
    # -----------------------------------------------------------------
    pain_points: list[PainPoint] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    scored_ideas: list[ScoredIdea] = Field(default_factory=list)
    pitch_briefs: list[PitchBrief] = Field(default_factory=list)

    # -----------------------------------------------------------------
    # Reflection loop state (per-pitch revision tracking)
    # -----------------------------------------------------------------
    critique: Critique | None = None
    critiques: list[Critique] = Field(default_factory=list)
    revision_counts: dict[str, int] = Field(default_factory=dict)
    revision_feedback: str | None = None
    current_revision_idea_id: UUID | None = None
    current_critique_index: int = Field(default=0, ge=0)
    pain_point_miner_revision_count: int = Field(default=0, ge=0)

    # -----------------------------------------------------------------
    # Retry tracking (prevent infinite loops when validation fails)
    # -----------------------------------------------------------------
    idea_generation_attempts: int = Field(default=0, ge=0)
    pitch_writer_attempts: int = Field(default=0, ge=0)
    scorer_attempts: int = Field(default=0, ge=0)
    max_idea_generation_attempts: int = Field(default=10, ge=1)
    max_total_llm_calls_per_agent: int = Field(default=15, ge=1)

    # -----------------------------------------------------------------
    # Orchestration control
    # -----------------------------------------------------------------
    next_node: str = Field(default="orchestrator")
    current_stage: PipelineStage = Field(default=PipelineStage.IDLE)
    previous_stage: PipelineStage | None = None

    # -----------------------------------------------------------------
    # Metadata & diagnostics
    # -----------------------------------------------------------------
    run_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    error_log: list[str] = Field(default_factory=list)
    agent_timings: dict[str, float] = Field(default_factory=dict)
    events: list[RunEvent] = Field(default_factory=list)

    # -----------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------
    @computed_field
    @property
    def filtered_pain_points(self) -> list[PainPoint]:
        """Pain points that passed rubric, sorted by evidence count descending."""
        passing = [pp for pp in self.pain_points if pp.passes_rubric]
        return sorted(passing, key=lambda pp: len(pp.evidence), reverse=True)

    @computed_field
    @property
    def top_scored_ideas(self) -> list[ScoredIdea]:
        """Ideas with verdict 'pursue' sorted by yes_count desc, limited to top_n_pitches.

        If no ideas achieve 'pursue', returns empty list to trigger idea generator revision.
        """
        pursue_ideas = [s for s in self.scored_ideas if s.verdict == "pursue"]
        ranked = sorted(
            pursue_ideas,
            key=lambda s: (s.yes_count, s.rank or 0),
            reverse=True,
        )
        return ranked[: self.top_n_pitches]

    @computed_field
    @property
    def can_revise(self) -> bool:
        """True if the current critique or reflection can still trigger revision."""
        if self.critique is None:
            if self.revision_counts:
                return max(self.revision_counts.values()) < self.max_revisions
            return True

        if self.critique.target_agent == "pain_point_miner":
            return self.pain_point_miner_revision_count < self.max_revisions
        else:
            return self.get_revision_count(self.critique.idea_id) < self.max_revisions

    @computed_field
    @property
    def is_complete(self) -> bool:
        """All expected pipeline outputs are present."""
        return all(
            [
                self.pain_points,
                self.ideas,
                self.scored_ideas,
                self.pitch_briefs,
                self.critique is not None or self.current_stage == PipelineStage.COMPLETED,
            ]
        )

    @computed_field
    @property
    def approved_pitches(self) -> list[PitchBrief]:
        """Pitches that passed 100% of the Critic rubric checks."""
        # Check critiques history and current critique
        all_critiques = list(self.critiques)
        if self.critique is not None:
            all_critiques.append(self.critique)

        # Map idea_id to latest critique
        latest_by_idea: dict[UUID, Critique] = {c.idea_id: c for c in all_critiques}

        approved: list[PitchBrief] = []
        for brief in self.pitch_briefs:
            crit = latest_by_idea.get(brief.idea_id)
            if crit is not None and crit.all_pass:
                approved.append(brief)
        return approved

    @computed_field
    @property
    def quarantined_pitches(self) -> list[PitchBrief]:
        """Pitches that failed one or more rubric checks after maximum revisions."""
        all_critiques = list(self.critiques)
        if self.critique is not None:
            all_critiques.append(self.critique)

        latest_by_idea: dict[UUID, Critique] = {c.idea_id: c for c in all_critiques}

        quarantined: list[PitchBrief] = []
        for brief in self.pitch_briefs:
            crit = latest_by_idea.get(brief.idea_id)
            if (
                crit is not None
                and not crit.all_pass
                and (
                    crit.approval_status == "max_revisions_reached"
                    or self.get_revision_count(brief.idea_id) >= self.max_revisions
                )
            ):
                quarantined.append(brief)
        return quarantined

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def get_revision_count(self, idea_id: UUID) -> int:
        """Return the number of revisions already done for a specific idea."""
        return self.revision_counts.get(str(idea_id), 0)

    def increment_revision_count(self, idea_id: UUID) -> "VentureForgeState":
        """Return a new state with incremented revision count."""
        updated = dict(self.revision_counts)
        updated[str(idea_id)] = updated.get(str(idea_id), 0) + 1
        return self.model_copy(update={"revision_counts": updated})

    def log_error(self, agent_id: str, message: str) -> dict[str, Any]:
        """Return patch logging an error."""
        entry = f"[{agent_id}] {message}"
        events = self.events + [
            RunEvent(
                agent=agent_id,
                stage=self.current_stage,
                kind="error",
                message=message,
            )
        ]
        return {"error_log": self.error_log + [entry], "events": events}

    def add_event(
        self,
        *,
        agent: str,
        stage: PipelineStage,
        kind: Literal["info", "warning", "error"] = "info",
        message: str,
        idea_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Return patch appending a RunEvent."""
        ev = RunEvent(agent=agent, stage=stage, kind=kind, message=message, idea_id=idea_id)
        return {"events": self.events + [ev]}

    @staticmethod
    def merge_patches(*patches: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple state patches, concatenating list fields and merging dict fields."""
        result: dict[str, Any] = {}
        list_fields = {"events", "error_log", "critiques", "pain_points", "ideas", "scored_ideas", "pitch_briefs"}
        dict_fields = {"revision_counts", "agent_timings"}

        for patch in patches:
            for key, value in patch.items():
                if key in list_fields and isinstance(value, list):
                    existing = result.get(key, [])
                    if isinstance(existing, list):
                        result[key] = existing + value
                    else:
                        result[key] = value
                elif key in dict_fields and isinstance(value, dict):
                    existing = result.get(key, {})
                    if isinstance(existing, dict):
                        result[key] = {**existing, **value}
                    else:
                        result[key] = value
                else:
                    result[key] = value

        return result

    def record_timing(self, agent_id: str, elapsed_s: float) -> dict[str, Any]:
        """Return a state patch recording agent timing."""
        timing = {**self.agent_timings, agent_id: elapsed_s}
        return {"agent_timings": timing}

    def bump_revision(self, critique: Critique) -> dict[str, Any]:
        """Return a state patch recording a revision."""
        patch: dict[str, Any] = {
            "critiques": self.critiques + [critique],
            "revision_feedback": critique.revision_feedback,
            "previous_stage": self.current_stage,
            "current_stage": PipelineStage.REVISING,
            "next_node": critique.target_agent,
        }

        if critique.target_agent == "pain_point_miner":
            patch["pain_point_miner_revision_count"] = self.pain_point_miner_revision_count + 1
        else:
            idea_id = str(critique.idea_id)
            updated_counts = {**self.revision_counts, idea_id: self.revision_counts.get(idea_id, 0) + 1}
            patch["revision_counts"] = updated_counts

        return patch

    def mark_completed(self) -> dict[str, Any]:
        """Return patch marking pipeline complete."""
        return {
            "current_stage": PipelineStage.COMPLETED,
            "next_node": "__end__",
            "completed_at": datetime.now(timezone.utc),
        }

    def mark_failed(self, reason: str) -> dict[str, Any]:
        """Return patch marking pipeline failed."""
        patch: dict[str, Any] = {
            "current_stage": PipelineStage.FAILED,
            "next_node": "__end__",
            "completed_at": datetime.now(timezone.utc),
        }
        patch.update(self.log_error("orchestrator", reason))
        return patch

    def mark_cancelled(self, reason: str = "Cancelled by user") -> dict[str, Any]:
        """Return patch marking pipeline cancelled."""
        patch: dict[str, Any] = {
            "current_stage": PipelineStage.CANCELLED,
            "next_node": "__end__",
            "completed_at": datetime.now(timezone.utc),
        }
        patch.update(
            self.add_event(
                agent="orchestrator",
                stage=self.current_stage,
                kind="warning",
                message=reason,
            )
        )
        return patch

    def reset_for_revision(self, target_agent: TargetAgent | str, idea_id: UUID) -> dict[str, Any]:
        """Clear only the data for the specific idea being revised."""
        target = target_agent if isinstance(target_agent, str) else target_agent.value
        updates: dict[str, Any] = {
            "current_stage": PipelineStage.REVISING,
            "current_revision_idea_id": idea_id,
        }

        if target == "pain_point_miner":
            # Strict downstream invalidation cascade: re-mining pain points invalidates all downstream ideas
            updates.update({
                "pain_points": [],
                "ideas": [],
                "scored_ideas": [],
                "pitch_briefs": [],
                "critique": None,
                "current_critique_index": 0,
            })
        elif target == "idea_generator":
            filtered_ideas = [i for i in self.ideas if i.id != idea_id]
            filtered_scored = [s for s in self.scored_ideas if s.idea_id != idea_id]
            filtered_briefs = [b for b in self.pitch_briefs if b.idea_id != idea_id]
            updates.update({
                "ideas": filtered_ideas,
                "scored_ideas": filtered_scored,
                "pitch_briefs": filtered_briefs,
                "critique": None,
            })
        elif target == "pitch_writer":
            filtered_briefs = [b for b in self.pitch_briefs if b.idea_id != idea_id]
            updates.update({
                "pitch_briefs": filtered_briefs,
                "critique": None,
                "current_critique_index": 0,
            })
        return updates
