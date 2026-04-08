"""Pure dataclasses for the DESO RL feedback loop. No external dependencies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enums ────────────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """Discrete actions the RL policy can recommend."""

    # Agent Selection
    SELECT_PROFILER = "select_profiler"
    SELECT_HARMONIZER = "select_harmonizer"
    SELECT_DQ = "select_dq"
    SELECT_ENRICHER = "select_enricher"
    SELECT_PUBLISHER = "select_publisher"

    # LLM Routing
    USE_CLAUDE = "use_claude"
    USE_GROQ = "use_groq"
    USE_SMALL_MODEL = "use_small_model"

    # Retrieval Strategy
    USE_DOMAIN_GRAPH = "use_domain_graph"
    USE_EXECUTION_GRAPH = "use_execution_graph"
    SKIP_RETRIEVAL = "skip_retrieval"

    # Execution Control
    CONTINUE = "continue"
    FINALIZE = "finalize"
    ESCALATE_HITL = "escalate_hitl"


class ActionCategory(str, Enum):
    """Categories grouping related actions."""

    AGENT_SELECTION = "agent_selection"
    LLM_ROUTING = "llm_routing"
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    EXECUTION_CONTROL = "execution_control"


# Map each action to its category
ACTION_CATEGORY_MAP: dict[ActionType, ActionCategory] = {
    ActionType.SELECT_PROFILER: ActionCategory.AGENT_SELECTION,
    ActionType.SELECT_HARMONIZER: ActionCategory.AGENT_SELECTION,
    ActionType.SELECT_DQ: ActionCategory.AGENT_SELECTION,
    ActionType.SELECT_ENRICHER: ActionCategory.AGENT_SELECTION,
    ActionType.SELECT_PUBLISHER: ActionCategory.AGENT_SELECTION,
    ActionType.USE_CLAUDE: ActionCategory.LLM_ROUTING,
    ActionType.USE_GROQ: ActionCategory.LLM_ROUTING,
    ActionType.USE_SMALL_MODEL: ActionCategory.LLM_ROUTING,
    ActionType.USE_DOMAIN_GRAPH: ActionCategory.RETRIEVAL_STRATEGY,
    ActionType.USE_EXECUTION_GRAPH: ActionCategory.RETRIEVAL_STRATEGY,
    ActionType.SKIP_RETRIEVAL: ActionCategory.RETRIEVAL_STRATEGY,
    ActionType.CONTINUE: ActionCategory.EXECUTION_CONTROL,
    ActionType.FINALIZE: ActionCategory.EXECUTION_CONTROL,
    ActionType.ESCALATE_HITL: ActionCategory.EXECUTION_CONTROL,
}

# Actions available per category
ACTIONS_BY_CATEGORY: dict[ActionCategory, list[ActionType]] = {}
for _act, _cat in ACTION_CATEGORY_MAP.items():
    ACTIONS_BY_CATEGORY.setdefault(_cat, []).append(_act)


# ── Dataclasses ──────────────────────────────────────────────────────────────


def _default_reward_weights() -> dict[str, float]:
    return {
        "quality_outcome": 0.4,
        "goal_coverage": 0.3,
        "efficiency": 0.2,
        "user_acceptance": 0.1,
    }


@dataclass
class RLConfig:
    """Tunable knobs for the RL subsystem."""

    enabled: bool = True
    exploration_rate: float = 0.3
    reward_weights: dict[str, float] = field(default_factory=_default_reward_weights)
    min_experiences: int = 20
    safe_actions_only: bool = True
    quality_threshold: float = 0.6
    max_latency_ms: int = 120_000
    max_cost_usd: float = 0.50

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "exploration_rate": self.exploration_rate,
            "reward_weights": dict(self.reward_weights),
            "min_experiences": self.min_experiences,
            "safe_actions_only": self.safe_actions_only,
            "quality_threshold": self.quality_threshold,
            "max_latency_ms": self.max_latency_ms,
            "max_cost_usd": self.max_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RLConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class StateSnapshot:
    """Frozen state at a decision point."""

    run_id: str = ""
    step_label: str = ""
    domain: str = ""
    lifecycle_stage: str = ""
    dataset_rows: int = 0
    dataset_cols: int = 0
    avg_null_pct: float = 0.0
    agents_called: list[str] = field(default_factory=list)
    llm_used: str = ""
    steps_completed: int = 0
    prior_quality_scores: list[float] = field(default_factory=list)
    latency_ms: int = 0
    cost_usd: float = 0.0
    domain_embedding: list[float] = field(default_factory=list)
    query_embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "step_label": self.step_label,
            "domain": self.domain,
            "lifecycle_stage": self.lifecycle_stage,
            "dataset_rows": self.dataset_rows,
            "dataset_cols": self.dataset_cols,
            "avg_null_pct": self.avg_null_pct,
            "agents_called": list(self.agents_called),
            "llm_used": self.llm_used,
            "steps_completed": self.steps_completed,
            "prior_quality_scores": list(self.prior_quality_scores),
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "domain_embedding": list(self.domain_embedding),
            "query_embedding": list(self.query_embedding),
        }

    @classmethod
    def from_dict(cls, d: dict) -> StateSnapshot:
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_feature_vector(self) -> list[float]:
        """Fixed-length numeric vector for future neural policy input."""
        features = [
            float(self.dataset_rows),
            float(self.dataset_cols),
            self.avg_null_pct,
            float(self.steps_completed),
            float(self.latency_ms),
            self.cost_usd,
            float(len(self.agents_called)),
        ]
        # Append mean of prior quality scores
        if self.prior_quality_scores:
            features.append(sum(self.prior_quality_scores) / len(self.prior_quality_scores))
        else:
            features.append(0.0)
        return features


@dataclass
class Experience:
    """Single SARS tuple: (state, action, reward, next_state)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = ""
    step_label: str = ""
    state: StateSnapshot | None = None
    action_category: str = ""
    action_taken: str = ""
    alternatives: list[str] = field(default_factory=list)
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    next_state: StateSnapshot | None = None
    done: bool = False
    outcome_metadata: dict[str, Any] = field(default_factory=dict)
    policy_version: int = 0
    app_id: str = "base"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_label": self.step_label,
            "state": self.state.to_dict() if self.state else None,
            "action_category": self.action_category,
            "action_taken": self.action_taken,
            "alternatives": list(self.alternatives),
            "reward": self.reward,
            "reward_components": dict(self.reward_components),
            "next_state": self.next_state.to_dict() if self.next_state else None,
            "done": self.done,
            "outcome_metadata": dict(self.outcome_metadata),
            "policy_version": self.policy_version,
            "app_id": self.app_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Experience:
        exp = cls(
            id=d.get("id", str(uuid.uuid4())),
            run_id=d.get("run_id", ""),
            step_label=d.get("step_label", ""),
            action_category=d.get("action_category", ""),
            action_taken=d.get("action_taken", ""),
            alternatives=d.get("alternatives", []),
            reward=d.get("reward"),
            reward_components=d.get("reward_components", {}),
            done=d.get("done", False),
            outcome_metadata=d.get("outcome_metadata", {}),
            policy_version=d.get("policy_version", 0),
            app_id=d.get("app_id", "base"),
        )
        if d.get("state"):
            exp.state = StateSnapshot.from_dict(d["state"])
        if d.get("next_state"):
            exp.next_state = StateSnapshot.from_dict(d["next_state"])
        return exp


@dataclass
class EpisodeSummary:
    """Per-run aggregate of RL decisions."""

    run_id: str = ""
    total_reward: float = 0.0
    num_decisions: int = 0
    actions_taken: list[str] = field(default_factory=list)
    avg_quality: float = 0.0
    health_score: int = 0
    policy_version: int = 0
    app_id: str = "base"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_reward": self.total_reward,
            "num_decisions": self.num_decisions,
            "actions_taken": list(self.actions_taken),
            "avg_quality": self.avg_quality,
            "health_score": self.health_score,
            "policy_version": self.policy_version,
            "app_id": self.app_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EpisodeSummary:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PolicyVersion:
    """Checkpoint record for a trained policy."""

    version: int = 0
    trained_at: str = ""
    experience_count: int = 0
    avg_reward: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    method: str = "thompson_sampling"
    is_active: bool = False
    app_id: str = "base"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "experience_count": self.experience_count,
            "avg_reward": self.avg_reward,
            "parameters": dict(self.parameters),
            "method": self.method,
            "is_active": self.is_active,
            "app_id": self.app_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PolicyVersion:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
