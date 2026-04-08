"""
deso/plan_state.py — Pure dataclasses for DESO Dynamic Planning and Adaptation.

No logic, no dependencies outside stdlib. All state containers used by
adaptive_planner.py and deso_runner.py (adaptive mode).

Public types
------------
AdaptationType      Enum: RETRY, SKIP, SUBSTITUTE, ESCALATE_HITL, DEGRADE
StepStatus          Enum: PENDING, READY, BLOCKED, RUNNING, COMPLETED, FAILED, SKIPPED, RETRYING
QualityExpectation  What we expect from an agent step
QualityScore        Scored output quality after execution
RetryContext        Tracks retry history for a step
StepState           Per-step execution state
AdaptationEvent     Logged adaptation action
GoalProgress        Progress tracker against IS goals
ReadinessResult     Input availability check for one agent
AdaptivePlanConfig  Tunable knobs (thresholds, max retries)
AdaptivePlanState   Top-level container for all adaptive state
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AdaptationType(Enum):
    """Types of mid-flight adaptation the planner can take."""
    RETRY = "RETRY"
    SKIP = "SKIP"
    SUBSTITUTE = "SUBSTITUTE"
    ESCALATE_HITL = "ESCALATE_HITL"
    DEGRADE = "DEGRADE"


class StepStatus(Enum):
    """Lifecycle states for a single execution step."""
    PENDING = "PENDING"
    READY = "READY"
    BLOCKED = "BLOCKED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    RETRYING = "RETRYING"


@dataclass
class QualityExpectation:
    """What we expect from an agent's output, derived from its agent card."""
    agent_id: str = ""
    min_output_keys: list[str] = field(default_factory=list)
    min_row_count: int | None = None
    max_null_pct: float | None = None
    goal_contribution: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScore:
    """Scored output quality for a single step."""
    agent_id: str = ""
    step_order: int = 0
    completeness: float = 0.0
    validity: float = 0.0
    volume: float = 0.0
    overall: float = 0.0
    passed: bool = False
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class RetryContext:
    """Tracks retry history for a step."""
    attempt: int = 0
    original_params: dict[str, Any] = field(default_factory=dict)
    adjusted_params: dict[str, Any] = field(default_factory=dict)
    previous_quality: QualityScore | None = None


@dataclass
class StepState:
    """Per-step execution state."""
    step_order: int = 0
    agent_id: str = ""
    status: StepStatus = StepStatus.PENDING
    quality: QualityScore | None = None
    retries: list[RetryContext] = field(default_factory=list)
    output: dict[str, Any] | None = None
    blocked_reason: str = ""
    is_optional: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class AdaptationEvent:
    """Logged adaptation action taken during execution."""
    event_type: AdaptationType = AdaptationType.DEGRADE
    agent_id: str = ""
    step_order: int = 0
    reason: str = ""
    resolution: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["event_type"] = self.event_type.value
        return d


@dataclass
class GoalProgress:
    """Progress tracker against IS explicit goal."""
    explicit_goal: str = ""
    steps_total: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    goal_coverage_pct: float = 0.0
    on_track: bool = True
    drift_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ReadinessResult:
    """Input availability check for one agent step."""
    agent_id: str = ""
    ready: bool = True
    missing_inputs: list[str] = field(default_factory=list)
    available_inputs: list[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class AdaptivePlanConfig:
    """Tunable knobs for adaptive execution."""
    enabled: bool = True
    quality_threshold: float = 0.6
    max_retries_per_step: int = 2
    goal_drift_threshold: float = 0.5

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class AdaptivePlanState:
    """Top-level container for all adaptive planning state."""
    config: AdaptivePlanConfig = field(default_factory=AdaptivePlanConfig)
    step_states: dict[int, StepState] = field(default_factory=dict)
    adaptation_log: list[AdaptationEvent] = field(default_factory=list)
    goal_progress: GoalProgress | None = None
    original_plan_hash: str = ""
    plan_version: int = 1
    is_adapted: bool = False

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "step_states": {
                k: v.to_dict() for k, v in self.step_states.items()
            },
            "adaptation_log": [e.to_dict() for e in self.adaptation_log],
            "goal_progress": self.goal_progress.to_dict() if self.goal_progress else None,
            "original_plan_hash": self.original_plan_hash,
            "plan_version": self.plan_version,
            "is_adapted": self.is_adapted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdaptivePlanState:
        """Hydrate from serialized dict."""
        config = AdaptivePlanConfig(**{
            k: v for k, v in data.get("config", {}).items()
            if k in AdaptivePlanConfig.__dataclass_fields__
        })
        state = cls(
            config=config,
            original_plan_hash=data.get("original_plan_hash", ""),
            plan_version=data.get("plan_version", 1),
            is_adapted=data.get("is_adapted", False),
        )
        # Hydrate step_states (keys are int step_order)
        for k, v in data.get("step_states", {}).items():
            ss = StepState(
                step_order=v.get("step_order", int(k)),
                agent_id=v.get("agent_id", ""),
                status=StepStatus(v.get("status", "PENDING")),
                is_optional=v.get("is_optional", False),
                blocked_reason=v.get("blocked_reason", ""),
            )
            state.step_states[int(k)] = ss
        # Hydrate adaptation_log
        for evt in data.get("adaptation_log", []):
            state.adaptation_log.append(AdaptationEvent(
                event_type=AdaptationType(evt.get("event_type", "DEGRADE")),
                agent_id=evt.get("agent_id", ""),
                step_order=evt.get("step_order", 0),
                reason=evt.get("reason", ""),
                resolution=evt.get("resolution", ""),
                details=evt.get("details", {}),
                timestamp=evt.get("timestamp", ""),
            ))
        # Hydrate goal_progress
        gp = data.get("goal_progress")
        if gp:
            state.goal_progress = GoalProgress(**{
                k: v for k, v in gp.items()
                if k in GoalProgress.__dataclass_fields__
            })
        return state
