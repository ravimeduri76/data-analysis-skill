"""RLLogger: accumulate decisions within an episode, attach rewards, finalize."""

from __future__ import annotations

import logging
import threading
from typing import Any

from hooks.deso.rl.rl_state import (
    ActionCategory,
    ActionType,
    EpisodeSummary,
    Experience,
    StateSnapshot,
)

log = logging.getLogger(__name__)


class RLLogger:
    """Accumulates RL decisions for a single DESO run (episode).

    Usage:
        logger = RLLogger(run_id="abc", policy_version=1)
        logger.log_decision("SA3", ActionCategory.AGENT_SELECTION,
                            ActionType.SELECT_PROFILER, [...], state)
        logger.attach_reward("SA3", 0.75, {"quality": 0.8, ...})
        ...
        episode = logger.finalize_episode(health_score=85)
        logger.flush(conn)
    """

    def __init__(self, run_id: str, policy_version: int = 0, app_id: str = "base") -> None:
        self.run_id = run_id
        self.policy_version = policy_version
        self.app_id = app_id
        self._experiences: dict[str, Experience] = {}  # step_label -> Experience
        self._order: list[str] = []  # insertion order of step_labels

    def log_decision(
        self,
        step_label: str,
        category: ActionCategory,
        action: ActionType,
        alternatives: list[ActionType],
        state: StateSnapshot,
    ) -> None:
        """Stage a new decision (Experience with no reward yet)."""
        exp = Experience(
            run_id=self.run_id,
            step_label=step_label,
            state=state,
            action_category=category.value,
            action_taken=action.value,
            alternatives=[a.value for a in alternatives],
            policy_version=self.policy_version,
            app_id=self.app_id,
        )
        self._experiences[step_label] = exp
        if step_label not in self._order:
            self._order.append(step_label)

    def attach_reward(
        self,
        step_label: str,
        reward: float,
        components: dict[str, float],
    ) -> None:
        """Link a computed reward to a previously staged decision."""
        exp = self._experiences.get(step_label)
        if exp is None:
            log.debug("No staged decision for step_label=%s; reward dropped", step_label)
            return
        exp.reward = reward
        exp.reward_components = components

    def finalize_episode(self, health_score: int = 0) -> EpisodeSummary:
        """Chain next_state pointers and mark last decision as done.

        Returns an EpisodeSummary for the episode.
        """
        # Chain next_state: each experience's next_state = next experience's state
        for i in range(len(self._order) - 1):
            current_label = self._order[i]
            next_label = self._order[i + 1]
            current_exp = self._experiences.get(current_label)
            next_exp = self._experiences.get(next_label)
            if current_exp and next_exp and next_exp.state:
                current_exp.next_state = next_exp.state

        # Mark last as done
        if self._order:
            last_exp = self._experiences.get(self._order[-1])
            if last_exp:
                last_exp.done = True

        # Build summary
        total_reward = 0.0
        quality_scores: list[float] = []
        actions_taken: list[str] = []

        for label in self._order:
            exp = self._experiences.get(label)
            if exp:
                if exp.reward is not None:
                    total_reward += exp.reward
                actions_taken.append(exp.action_taken)
                q = exp.reward_components.get("quality_outcome")
                if q is not None:
                    quality_scores.append(q)

        avg_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        return EpisodeSummary(
            run_id=self.run_id,
            total_reward=total_reward,
            num_decisions=len(self._order),
            actions_taken=actions_taken,
            avg_quality=avg_quality,
            health_score=health_score,
            policy_version=self.policy_version,
            app_id=self.app_id,
        )

    def flush(self, conn: Any, async_write: bool = True) -> int:
        """Batch-write all staged experiences to the DB.

        Args:
            conn: PostgreSQL connection (or a callable returning one for async).
            async_write: If True (default), writes in a background thread so
                         the DESO hot path is never blocked by DB I/O.

        Returns count of experiences queued/written. Best-effort — never raises.
        """
        if not self._experiences:
            return 0
        try:
            exps = [self._experiences[label] for label in self._order
                    if label in self._experiences]
            count = len(exps)

            if async_write:
                # Snapshot data for the background thread; share nothing mutable
                exp_dicts = [e.to_dict() for e in exps]
                thread = threading.Thread(
                    target=self._flush_background,
                    args=(conn, exp_dicts),
                    daemon=True,
                    name="rl-flush",
                )
                thread.start()
                return count

            # Synchronous fallback (used by tests)
            from hooks.deso.rl.rl_store import save_experience_batch
            save_experience_batch(conn, exps)
            return count
        except Exception as exc:
            log.debug("RLLogger.flush failed (non-fatal): %s", exc)
            return 0

    @staticmethod
    def _flush_background(conn: Any, exp_dicts: list[dict]) -> None:
        """Background thread target: deserialize and persist experiences."""
        try:
            from hooks.deso.rl.rl_state import Experience
            from hooks.deso.rl.rl_store import save_experience_batch

            exps = [Experience.from_dict(d) for d in exp_dicts]
            save_experience_batch(conn, exps)
        except Exception as exc:
            log.debug("Background RL flush failed (non-fatal): %s", exc)

    @property
    def decision_count(self) -> int:
        return len(self._order)
