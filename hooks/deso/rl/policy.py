"""Thompson Sampling bandit policy for DESO RL (Phase 1).

Maintains Beta(alpha, beta) per (category, action) pair.
No PyTorch — pure numpy Beta sampling.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np

from hooks.deso.rl.rl_state import (
    ACTIONS_BY_CATEGORY,
    ActionCategory,
    ActionType,
    PolicyVersion,
    RLConfig,
    StateSnapshot,
)

log = logging.getLogger(__name__)

# Agents that require HITL approval before execution
_HITL_AGENTS = {"AG03", "AG10", "AG11", "AG16", "AG22", "AG28", "AG33", "AG35"}

# Map action types to the agent IDs they imply
_ACTION_AGENT_MAP: dict[ActionType, str] = {
    ActionType.SELECT_PROFILER: "AG06",
    ActionType.SELECT_HARMONIZER: "AG09",
    ActionType.SELECT_DQ: "AG11",
    ActionType.SELECT_ENRICHER: "AG25",
    ActionType.SELECT_PUBLISHER: "AG34",
}


class ThompsonSamplingPolicy:
    """Contextual bandit using Thompson Sampling with Beta priors."""

    def __init__(self, config: RLConfig | None = None) -> None:
        self.config = config or RLConfig()
        # Beta parameters: {(category_value, action_value): (alpha, beta)}
        self._params: dict[tuple[str, str], tuple[float, float]] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._version: int = 0
        self._init_priors()

    def _init_priors(self) -> None:
        """Initialize uniform Beta(1,1) priors for all (category, action) pairs."""
        for cat, actions in ACTIONS_BY_CATEGORY.items():
            for act in actions:
                key = (cat.value, act.value)
                if key not in self._params:
                    self._params[key] = (1.0, 1.0)
                    self._counts[key] = 0

    def select_action(
        self,
        category: ActionCategory,
        state: StateSnapshot | None = None,
        safe_actions: list[ActionType] | None = None,
    ) -> ActionType:
        """Sample from Beta distributions and pick the highest-scoring action.

        Falls back to epsilon-greedy when experience count is below min_experiences.

        Args:
            category: Which action category to select from.
            state: Current state (reserved for future contextual features).
            safe_actions: Pre-filtered list of allowed actions. If None, all in category.

        Returns:
            Selected ActionType.
        """
        candidates = safe_actions or ACTIONS_BY_CATEGORY.get(category, [])
        if not candidates:
            return ActionType.CONTINUE  # absolute fallback

        # Check if we have enough experience for Thompson Sampling
        total_count = sum(
            self._counts.get((category.value, a.value), 0) for a in candidates
        )

        if total_count < self.config.min_experiences:
            # Epsilon-greedy fallback: explore with exploration_rate
            if random.random() < self.config.exploration_rate:
                return random.choice(candidates)
            # Exploit: pick action with best empirical mean
            return self._best_empirical(category, candidates)

        # Thompson Sampling: sample from Beta for each candidate
        scores: list[tuple[float, ActionType]] = []
        for act in candidates:
            key = (category.value, act.value)
            alpha, beta = self._params.get(key, (1.0, 1.0))
            sample = np.random.beta(alpha, beta)
            scores.append((sample, act))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def _best_empirical(
        self, category: ActionCategory, candidates: list[ActionType]
    ) -> ActionType:
        """Pick action with highest empirical mean (alpha / (alpha + beta))."""
        best_action = candidates[0]
        best_mean = -1.0
        for act in candidates:
            key = (category.value, act.value)
            alpha, beta = self._params.get(key, (1.0, 1.0))
            mean = alpha / (alpha + beta)
            if mean > best_mean:
                best_mean = mean
                best_action = act
        return best_action

    def update(self, category: ActionCategory, action: ActionType, reward: float) -> None:
        """Update Beta parameters based on observed reward.

        reward > 0.5 → alpha += 1 (success)
        reward <= 0.5 → beta += 1 (failure)
        """
        key = (category.value, action.value)
        alpha, beta = self._params.get(key, (1.0, 1.0))
        if reward > 0.5:
            alpha += 1.0
        else:
            beta += 1.0
        self._params[key] = (alpha, beta)
        self._counts[key] = self._counts.get(key, 0) + 1

    def get_params(self) -> dict[str, Any]:
        """Serialize Beta parameters for persistence."""
        return {
            "beta_params": {
                f"{cat}|{act}": {"alpha": a, "beta": b}
                for (cat, act), (a, b) in self._params.items()
            },
            "counts": {
                f"{cat}|{act}": c
                for (cat, act), c in self._counts.items()
            },
            "version": self._version,
        }

    def load_params(self, params: dict[str, Any]) -> None:
        """Restore Beta parameters from persisted dict."""
        beta_params = params.get("beta_params", {})
        for key_str, vals in beta_params.items():
            parts = key_str.split("|", 1)
            if len(parts) == 2:
                self._params[(parts[0], parts[1])] = (
                    vals.get("alpha", 1.0),
                    vals.get("beta", 1.0),
                )
        counts = params.get("counts", {})
        for key_str, c in counts.items():
            parts = key_str.split("|", 1)
            if len(parts) == 2:
                self._counts[(parts[0], parts[1])] = c
        self._version = params.get("version", 0)

    def save(self, conn: Any) -> PolicyVersion:
        """Save current parameters as a new policy version to DB."""
        from hooks.deso.rl.rl_store import save_policy_version, load_active_policy

        # Deactivate current active policy
        try:
            current = load_active_policy(conn)
            if current:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE rl_policy_versions SET is_active = FALSE WHERE version = %s",
                    (current.version,),
                )
                conn.commit()
                cur.close()
        except Exception:
            pass

        self._version += 1
        total_count = sum(self._counts.values())
        total_reward = sum(
            a / (a + b) * self._counts.get(k, 0)
            for k, (a, b) in self._params.items()
        )
        avg_reward = total_reward / total_count if total_count > 0 else 0.0

        pv = PolicyVersion(
            version=self._version,
            trained_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            experience_count=total_count,
            avg_reward=avg_reward,
            parameters=self.get_params(),
            method="thompson_sampling",
            is_active=True,
        )
        save_policy_version(conn, pv)
        return pv

    @classmethod
    def load(cls, conn: Any, config: RLConfig | None = None) -> ThompsonSamplingPolicy:
        """Load active policy from DB, or return fresh policy if none exists."""
        policy = cls(config=config)
        try:
            from hooks.deso.rl.rl_store import load_active_policy

            pv = load_active_policy(conn)
            if pv and pv.parameters:
                policy.load_params(pv.parameters)
                policy._version = pv.version
        except Exception as exc:
            log.debug("Could not load policy from DB (using fresh): %s", exc)
        return policy


def filter_safe_actions(
    category: ActionCategory,
    dcp: dict | None = None,
    permission_result: dict | None = None,
    config: RLConfig | None = None,
) -> list[ActionType]:
    """Filter actions to only those satisfying safety constraints.

    Guarantees:
    - ESCALATE_HITL is always in the safe set (safety valve).
    - At least one action is always returned.
    - HITL-requiring agents are filtered if not pre-approved.
    - FINALIZE blocked if quality below threshold.
    - USE_CLAUDE blocked if ANTHROPIC_API_KEY absent.

    Args:
        category: Action category to filter.
        dcp: Domain Context Packet dict (optional).
        permission_result: HITL permission result (optional).
        config: RL config for thresholds.

    Returns:
        List of safe ActionType values (never empty).
    """
    cfg = config or RLConfig()
    all_actions = ACTIONS_BY_CATEGORY.get(category, [])
    if not all_actions:
        return [ActionType.CONTINUE]

    safe: list[ActionType] = []
    permissions = permission_result or {}
    approved_agents = set(permissions.get("approved_agents", []))

    for action in all_actions:
        # Check agent HITL constraints
        if action in _ACTION_AGENT_MAP:
            agent_id = _ACTION_AGENT_MAP[action]
            if agent_id in _HITL_AGENTS and agent_id not in approved_agents:
                continue

        # FINALIZE blocked if quality below threshold
        if action == ActionType.FINALIZE:
            avg_quality = permissions.get("avg_quality", 1.0)
            if avg_quality < cfg.quality_threshold:
                continue

        # USE_CLAUDE blocked if no API key
        if action == ActionType.USE_CLAUDE:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                continue

        safe.append(action)

    # ESCALATE_HITL is always available as safety valve
    if category == ActionCategory.EXECUTION_CONTROL:
        if ActionType.ESCALATE_HITL not in safe:
            safe.append(ActionType.ESCALATE_HITL)

    # Guarantee at least one action
    if not safe:
        # Deterministic fallback per category
        fallbacks = {
            ActionCategory.AGENT_SELECTION: ActionType.SELECT_PROFILER,
            ActionCategory.LLM_ROUTING: ActionType.USE_SMALL_MODEL,
            ActionCategory.RETRIEVAL_STRATEGY: ActionType.SKIP_RETRIEVAL,
            ActionCategory.EXECUTION_CONTROL: ActionType.ESCALATE_HITL,
        }
        safe.append(fallbacks.get(category, ActionType.CONTINUE))

    return safe
