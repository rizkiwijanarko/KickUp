"""
LangGraph State Schema for VentureForge Pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from src.models import (
    Critique,
    Idea,
    PainPoint,
    PipelineStage,
    PitchBrief,
    RunEvent,
    ScoredIdea,
    TargetAgent,
)


class RevisionLedger:
    """Read-only query interface over the revision-tracking fields of
    VentureForgeState.  Created on demand via ``state.revisions``; never
    stored, never serialised.

    Interface (the seam external code crosses):
        can_revise(idea_id?)  -> bool
        count(idea_id)        -> int
        feedback              -> str | None
        current_critique      -> Critique | None
        all_critiques         -> list[Critique]
        approved_idea_ids     -> frozenset[UUID]
        quarantined_idea_ids  -> frozenset[UUID]

    Implementation: six scattered raw fields snapshotted at construction.
    Callers never touch ``critique``, ``critiques``, ``revision_counts``,
    ``revision_feedback``, or ``pain_point_miner_revision_count`` directly.
    """

    def __init__(self, state: VentureForgeState) -> None:
        self._critique = state.critique
        self._critiques = list(state.critiques)
        self._revision_counts = dict(state.revision_counts)
        self._revision_feedback = state.revision_feedback
        self._pp_miner_count = state.pain_point_miner_revision_count
        self._max_revisions = state.max_revisions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def can_revise(self, idea_id: UUID | None = None) -> bool:
        """Return True if another revision cycle is allowed.

        Without idea_id: global check (mirrors ``state.can_revise``).
        With idea_id:    per-idea check used by critic and orchestrator.
        """
        if idea_id is None:
            return self._global_can_revise()
        return self._revision_counts.get(str(idea_id), 0) < self._max_revisions

    def count(self, idea_id: UUID) -> int:
        """Number of revisions already applied to this idea."""
        return self._revision_counts.get(str(idea_id), 0)

    @property
    def feedback(self) -> str | None:
        """Most recent revision feedback string."""
        return self._revision_feedback

    @property
    def current_critique(self) -> Critique | None:
        """The active Critique object, if any."""
        return self._critique

    @property
    def all_critiques(self) -> list[Critique]:
        """Historical critiques plus the current one."""
        combined = list(self._critiques)
        if self._critique is not None:
            combined.append(self._critique)
        return combined

    @property
    def approved_idea_ids(self) -> frozenset[UUID]:
        """IDs of ideas whose latest Critique has all_pass=True."""
        latest = self._latest_by_idea()
        return frozenset(id_ for id_, c in latest.items() if c.all_pass)

    @property
    def quarantined_idea_ids(self) -> frozenset[UUID]:
        """IDs of ideas that failed critique at max revisions."""
        latest = self._latest_by_idea()
        return frozenset(
            id_
            for id_, c in latest.items()
            if not c.all_pass
            and (
                c.approval_status == "max_revisions_reached"
                or self.count(id_) >= self._max_revisions
            )
        )

    def pending_briefs(self, pitch_briefs: list[PitchBrief]) -> list[PitchBrief]:
        """Return briefs that have not been approved and have not reached max revisions."""
        approved = self.approved_idea_ids
        quarantined = self.quarantined_idea_ids
        return [
            b for b in pitch_briefs if b.idea_id not in approved and b.idea_id not in quarantined
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _global_can_revise(self) -> bool:
        if self._critique is None:
            if self._revision_counts:
                return max(self._revision_counts.values()) < self._max_revisions
            return True
        if self._critique.target_agent == "pain_point_miner":
            return self._pp_miner_count < self._max_revisions
        return self.can_revise(self._critique.idea_id)

    def _latest_by_idea(self) -> dict[UUID, Critique]:
        return {c.idea_id: c for c in self.all_critiques}


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
        return self.revisions.can_revise()

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
        approved_ids = self.revisions.approved_idea_ids
        return [b for b in self.pitch_briefs if b.idea_id in approved_ids]

    @computed_field
    @property
    def quarantined_pitches(self) -> list[PitchBrief]:
        """Pitches that failed one or more rubric checks after maximum revisions."""
        quarantined_ids = self.revisions.quarantined_idea_ids
        return [b for b in self.pitch_briefs if b.idea_id in quarantined_ids]

    # -----------------------------------------------------------------
    # Revision query accessor (plain property — not serialised)
    # -----------------------------------------------------------------
    @property
    def revisions(self) -> RevisionLedger:
        """Read-only query interface over all revision-tracking state.

        Use this instead of accessing critique / critiques / revision_counts
        / revision_feedback / pain_point_miner_revision_count directly.
        Not a computed_field — RevisionLedger is not serialised by LangGraph.
        """
        return RevisionLedger(self)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def get_revision_count(self, idea_id: UUID) -> int:
        """Return the number of revisions already done for a specific idea.

        Prefer ``state.revisions.count(idea_id)`` for new code.
        Kept for backward compatibility with existing tests.
        """
        return self.revisions.count(idea_id)

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
        list_fields = {
            "events",
            "error_log",
            "critiques",
            "pain_points",
            "ideas",
            "scored_ideas",
            "pitch_briefs",
        }
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
            updated_counts = {
                **self.revision_counts,
                idea_id: self.revision_counts.get(idea_id, 0) + 1,
            }
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
            updates.update(
                {
                    "pain_points": [],
                    "ideas": [],
                    "scored_ideas": [],
                    "pitch_briefs": [],
                    "critique": None,
                }
            )
        elif target == "idea_generator":
            filtered_ideas = [i for i in self.ideas if i.id != idea_id]
            filtered_scored = [s for s in self.scored_ideas if s.idea_id != idea_id]
            filtered_briefs = [b for b in self.pitch_briefs if b.idea_id != idea_id]
            updates.update(
                {
                    "ideas": filtered_ideas,
                    "scored_ideas": filtered_scored,
                    "pitch_briefs": filtered_briefs,
                    "critique": None,
                }
            )
        elif target == "pitch_writer":
            filtered_briefs = [b for b in self.pitch_briefs if b.idea_id != idea_id]
            updates.update(
                {
                    "pitch_briefs": filtered_briefs,
                    "critique": None,
                }
            )
        return updates
