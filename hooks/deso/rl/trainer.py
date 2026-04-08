"""Offline batch training for DESO RL Thompson Sampling policy.

Loads experiences from the DB, updates Beta parameters, and saves a new
policy version. No neural network — pure Beta distribution updates.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from hooks.deso.rl.policy import ThompsonSamplingPolicy
from hooks.deso.rl.rl_state import (
    ActionCategory,
    ActionType,
    PolicyVersion,
    RLConfig,
)

log = logging.getLogger(__name__)


def train_from_buffer(
    conn: Any,
    config: RLConfig | None = None,
) -> PolicyVersion:
    """Train a new policy version from the experience buffer.

    Steps:
    1. Load active policy (or init fresh).
    2. Load experiences since last training.
    3. Update Beta params per (category, action) from rewards.
    4. Deactivate old policy, save new version as active.
    5. Return new PolicyVersion.

    Args:
        conn: PostgreSQL connection.
        config: RL configuration overrides.

    Returns:
        The newly created PolicyVersion.
    """
    from hooks.deso.rl.rl_store import load_active_policy, load_experiences

    cfg = config or RLConfig()

    # 1. Load or create policy
    policy = ThompsonSamplingPolicy.load(conn, config=cfg)
    old_version = policy._version

    # 2. Load experiences since last training version
    exps = load_experiences(conn, limit=2000, since_version=old_version)
    if not exps:
        log.info("No new experiences to train on (version=%d)", old_version)
        # Return current policy version info
        current = load_active_policy(conn)
        if current:
            return current
        return PolicyVersion(version=old_version, method="thompson_sampling")

    # 3. Update Beta parameters from experience rewards
    updated_count = 0
    for exp in exps:
        if exp.reward is None:
            continue
        try:
            category = ActionCategory(exp.action_category)
            action = ActionType(exp.action_taken)
            policy.update(category, action, exp.reward)
            updated_count += 1
        except (ValueError, KeyError) as exc:
            log.debug("Skipping invalid experience %s: %s", exp.id, exc)

    if updated_count == 0:
        log.info("No experiences with rewards to learn from")
        current = load_active_policy(conn)
        if current:
            return current
        return PolicyVersion(version=old_version, method="thompson_sampling")

    log.info(
        "Updated policy from %d experiences (version %d -> %d)",
        updated_count, old_version, old_version + 1,
    )

    # 4. Save new policy version
    new_pv = policy.save(conn)
    return new_pv


def evaluate_policy(conn: Any, policy_version: int | None = None) -> dict[str, Any]:
    """Evaluate a policy version's performance.

    Args:
        conn: PostgreSQL connection.
        policy_version: Version to evaluate, or None for active.

    Returns:
        Dict with avg_reward, action_distribution, per-category stats.
    """
    from hooks.deso.rl.rl_store import load_experiences

    if policy_version is None:
        from hooks.deso.rl.rl_store import load_active_policy

        active = load_active_policy(conn)
        policy_version = active.version if active else 0

    exps = load_experiences(conn, limit=5000, since_version=policy_version)
    # Filter to only experiences made under this version
    version_exps = [e for e in exps if e.policy_version == policy_version]

    if not version_exps:
        return {
            "policy_version": policy_version,
            "total_experiences": 0,
            "avg_reward": 0.0,
            "action_distribution": {},
            "per_category": {},
        }

    # Aggregate
    total_reward = 0.0
    rewarded_count = 0
    action_counts: dict[str, int] = {}
    action_rewards: dict[str, list[float]] = {}
    category_stats: dict[str, dict[str, Any]] = {}

    for exp in version_exps:
        action_counts[exp.action_taken] = action_counts.get(exp.action_taken, 0) + 1

        if exp.reward is not None:
            total_reward += exp.reward
            rewarded_count += 1
            action_rewards.setdefault(exp.action_taken, []).append(exp.reward)

            cat = exp.action_category
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "total_reward": 0.0, "actions": {}}
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_reward"] += exp.reward
            cat_actions = category_stats[cat]["actions"]
            cat_actions[exp.action_taken] = cat_actions.get(exp.action_taken, 0) + 1

    avg_reward = total_reward / rewarded_count if rewarded_count > 0 else 0.0

    return {
        "policy_version": policy_version,
        "total_experiences": len(version_exps),
        "avg_reward": avg_reward,
        "action_distribution": action_counts,
        "action_avg_rewards": {
            act: sum(rews) / len(rews) for act, rews in action_rewards.items()
        },
        "per_category": {
            cat: {
                "count": s["count"],
                "avg_reward": s["total_reward"] / s["count"] if s["count"] > 0 else 0.0,
                "actions": s["actions"],
            }
            for cat, s in category_stats.items()
        },
    }


def rollback_policy(conn: Any, to_version: int) -> None:
    """Reactivate a previous policy version.

    Sets is_active=True on the target version and False on all others.
    """
    cur = conn.cursor()
    cur.execute("UPDATE rl_policy_versions SET is_active = FALSE WHERE is_active = TRUE")
    cur.execute(
        "UPDATE rl_policy_versions SET is_active = TRUE WHERE version = %s",
        (to_version,),
    )
    conn.commit()
    cur.close()
    log.info("Rolled back to policy version %d", to_version)
