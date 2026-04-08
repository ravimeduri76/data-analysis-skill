"""Pure reward computation for DESO RL feedback loop.

R = w1*quality + w2*coverage + w3*efficiency + w4*acceptance

All functions are pure — no DB, no side effects.
"""

from __future__ import annotations


def default_weights() -> dict[str, float]:
    """Default reward weight distribution."""
    return {
        "quality_outcome": 0.4,
        "goal_coverage": 0.3,
        "efficiency": 0.2,
        "user_acceptance": 0.1,
    }


def compute_reward(
    quality_score: float,
    goal_coverage: float,
    latency_ms: int,
    cost_usd: float,
    user_accepted: bool | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute composite reward from outcome signals.

    Args:
        quality_score: QualityScore.overall (0-1).
        goal_coverage: GoalProgress.coverage_pct / 100 (0-1).
        latency_ms: Step or total latency in milliseconds.
        cost_usd: API cost in USD.
        user_accepted: HITL feedback (True/False/None).
        weights: Custom weight dict, or None for defaults.

    Returns:
        (total_reward, component_dict) where total_reward is in [-1, 1].
    """
    w = weights or default_weights()

    # Quality outcome: direct pass-through, clamped
    quality_component = max(0.0, min(1.0, quality_score))

    # Goal coverage: normalized, clamped
    coverage_component = max(0.0, min(1.0, goal_coverage))

    # Efficiency: penalize high latency and cost
    max_latency = 120_000  # 120s baseline
    max_cost = 0.50  # $0.50 baseline
    latency_penalty = min(1.0, latency_ms / max_latency) if max_latency > 0 else 0.0
    cost_penalty = min(1.0, cost_usd / max_cost) if max_cost > 0 else 0.0
    efficiency_component = max(0.0, 1.0 - 0.5 * latency_penalty - 0.5 * cost_penalty)

    # User acceptance: +1 accept, -1 reject, 0 no feedback
    if user_accepted is True:
        acceptance_component = 1.0
    elif user_accepted is False:
        acceptance_component = -1.0
    else:
        acceptance_component = 0.0

    components = {
        "quality_outcome": quality_component,
        "goal_coverage": coverage_component,
        "efficiency": efficiency_component,
        "user_acceptance": acceptance_component,
    }

    total = (
        w.get("quality_outcome", 0.4) * quality_component
        + w.get("goal_coverage", 0.3) * coverage_component
        + w.get("efficiency", 0.2) * efficiency_component
        + w.get("user_acceptance", 0.1) * acceptance_component
    )

    # Clamp to [-1, 1]
    total = max(-1.0, min(1.0, total))

    return total, components


def compute_step_reward(
    quality_score: float,
    latency_ms: int,
    cost_usd: float,
    user_accepted: bool | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Simplified per-step reward (no goal coverage, redistributed weights).

    Uses quality=0.5, efficiency=0.35, acceptance=0.15 by default.
    """
    step_weights = weights or {
        "quality_outcome": 0.5,
        "goal_coverage": 0.0,
        "efficiency": 0.35,
        "user_acceptance": 0.15,
    }
    return compute_reward(
        quality_score=quality_score,
        goal_coverage=0.0,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        user_accepted=user_accepted,
        weights=step_weights,
    )
