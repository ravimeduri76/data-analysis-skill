"""
deso/adaptive_planner.py — Pure functions for DESO Dynamic Planning and Adaptation.

Takes state in, returns decisions out. No LLM calls, no DB access.
All functions are deterministic given the same inputs.

Public API
----------
build_quality_expectations(ap, agent_cards)            -> dict[str, QualityExpectation]
validate_agent_readiness(step_order, agent_id, ...)    -> ReadinessResult
validate_plan_readiness(ap, initial_data, agent_cards) -> list[ReadinessResult]
evaluate_step_output(agent_id, step_order, output, ..) -> QualityScore
decide_adaptation(step_state, config, agent_card)      -> AdaptationEvent | None
update_goal_progress(is_spec, step_states, ap)         -> GoalProgress
check_goal_drift(progress, config)                     -> list[str]
init_adaptive_state(ap, config)                        -> AdaptivePlanState
render_adaptation_log(events)                          -> str
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from hooks.deso.plan_state import (
    AdaptationEvent,
    AdaptationType,
    AdaptivePlanConfig,
    AdaptivePlanState,
    GoalProgress,
    QualityExpectation,
    QualityScore,
    ReadinessResult,
    StepState,
    StepStatus,
)


# ── Quality Expectations ────────────────────────────────────────────────────


def build_quality_expectations(
    ap: dict,
    agent_cards: dict[str, dict],
) -> dict[str, QualityExpectation]:
    """Derive quality expectations from agent card egress_schema for each agent in AP.

    Returns a dict keyed by agent_id.
    """
    expectations: dict[str, QualityExpectation] = {}
    for agent in ap.get("agents", []):
        agent_id = agent.get("agent_id", "")
        if not agent_id:
            continue
        card = agent_cards.get(agent_id, {})
        egress = card.get("egress_schema", {})
        properties = egress.get("properties", {})
        required = egress.get("required", list(properties.keys()))

        expectations[agent_id] = QualityExpectation(
            agent_id=agent_id,
            min_output_keys=list(required) if required else list(properties.keys()),
            min_row_count=card.get("min_row_count"),
            max_null_pct=card.get("max_null_pct"),
            goal_contribution=agent.get("purpose", ""),
        )
    return expectations


# ── Readiness Checks ────────────────────────────────────────────────────────


def validate_agent_readiness(
    step_order: int,
    agent_id: str,
    ingress_schema: dict | None,
    available_data: set[str],
) -> ReadinessResult:
    """Check whether one agent's required inputs are available.

    Args:
        step_order: Position in the execution plan.
        agent_id: Agent identifier (e.g. "AG07").
        ingress_schema: Agent card's ingress_schema (may be None).
        available_data: Set of data keys available at this point in execution.

    Returns:
        ReadinessResult with ready=True if all required inputs are available.
    """
    if not ingress_schema:
        # No schema defined — assume ready (graceful default)
        return ReadinessResult(
            agent_id=agent_id,
            ready=True,
            available_inputs=sorted(available_data),
        )

    required = set(ingress_schema.get("required", []))
    if not required:
        # Schema exists but no required fields
        return ReadinessResult(
            agent_id=agent_id,
            ready=True,
            available_inputs=sorted(available_data),
        )

    present = required & available_data
    missing = required - available_data

    suggestion = ""
    if missing:
        suggestion = (
            f"Step {step_order} ({agent_id}): missing {sorted(missing)}. "
            f"Ensure upstream agents produce these outputs before execution."
        )

    return ReadinessResult(
        agent_id=agent_id,
        ready=len(missing) == 0,
        missing_inputs=sorted(missing),
        available_inputs=sorted(present),
        suggestion=suggestion,
    )


def validate_plan_readiness(
    ap: dict,
    initial_data: set[str],
    agent_cards: dict[str, dict],
) -> list[ReadinessResult]:
    """Walk the AP agent DAG, accumulate egress as available data, check all agents.

    Args:
        ap: Action Plan dict with "agents" list (each has agent_id, step_order).
        initial_data: Set of data keys available before any agent runs
                      (e.g. dataset columns, DCP fields).
        agent_cards: Dict of agent_id -> agent card dicts (with ingress_schema/egress_schema).

    Returns:
        List of ReadinessResult, one per agent in plan order.
    """
    results: list[ReadinessResult] = []
    available = set(initial_data)

    # Sort agents by step_order
    agents = sorted(ap.get("agents", []), key=lambda a: a.get("step_order", 0))

    for agent in agents:
        agent_id = agent.get("agent_id", "")
        step_order = agent.get("step_order", 0)
        card = agent_cards.get(agent_id, {})

        ingress = card.get("ingress_schema")
        result = validate_agent_readiness(step_order, agent_id, ingress, available)
        results.append(result)

        # Accumulate egress outputs as available data for downstream agents
        egress = card.get("egress_schema", {})
        egress_keys = set(egress.get("properties", {}).keys())
        available |= egress_keys

    return results


# ── Quality Evaluation ──────────────────────────────────────────────────────


def evaluate_step_output(
    agent_id: str,
    step_order: int,
    output: dict | None,
    expectation: QualityExpectation,
    egress_schema: dict | None = None,
    threshold: float = 0.6,
) -> QualityScore:
    """Score output quality: completeness (50%) + validity (30%) + volume (20%).

    Graceful defaults:
    - No schema -> completeness=1.0 if output non-empty
    - No row expectation -> volume=1.0
    - None output -> all zeros
    """
    issues: list[str] = []

    if output is None or not output:
        return QualityScore(
            agent_id=agent_id,
            step_order=step_order,
            completeness=0.0,
            validity=0.0,
            volume=0.0,
            overall=0.0,
            passed=False,
            issues=["Output is empty or None"],
        )

    # ── Completeness (weight 0.5) ───────────────────────────────────────
    required_keys = expectation.min_output_keys
    if required_keys:
        present_keys = [k for k in required_keys if k in output]
        completeness = len(present_keys) / len(required_keys)
        missing = [k for k in required_keys if k not in output]
        if missing:
            issues.append(f"Missing output keys: {missing}")
    else:
        # No schema — non-empty output counts as complete
        completeness = 1.0

    # ── Validity (weight 0.3) ───────────────────────────────────────────
    if egress_schema and egress_schema.get("properties"):
        type_map = _build_type_map(egress_schema["properties"])
        valid_count = 0
        check_count = 0
        for key, expected_type in type_map.items():
            if key in output:
                check_count += 1
                if _type_matches(output[key], expected_type):
                    valid_count += 1
                else:
                    issues.append(
                        f"Type mismatch: {key} expected {expected_type}, "
                        f"got {type(output[key]).__name__}"
                    )
        validity = valid_count / check_count if check_count > 0 else 1.0
    else:
        validity = 1.0  # No schema to validate against

    # ── Volume (weight 0.2) ─────────────────────────────────────────────
    if expectation.min_row_count is not None and expectation.min_row_count > 0:
        # Look for row count indicators in output
        actual_rows = _extract_row_count(output)
        if actual_rows is not None:
            volume = min(1.0, actual_rows / expectation.min_row_count)
            if actual_rows < expectation.min_row_count:
                issues.append(
                    f"Row count {actual_rows} below minimum {expectation.min_row_count}"
                )
        else:
            volume = 1.0  # Can't determine row count — assume OK
    else:
        volume = 1.0  # No row expectation

    # ── Overall ─────────────────────────────────────────────────────────
    overall = (completeness * 0.5) + (validity * 0.3) + (volume * 0.2)
    passed = overall >= threshold

    return QualityScore(
        agent_id=agent_id,
        step_order=step_order,
        completeness=round(completeness, 4),
        validity=round(validity, 4),
        volume=round(volume, 4),
        overall=round(overall, 4),
        passed=passed,
        issues=issues,
    )


def _build_type_map(properties: dict) -> dict[str, str]:
    """Extract expected types from JSON-schema-style properties."""
    type_map: dict[str, str] = {}
    for key, spec in properties.items():
        if isinstance(spec, dict) and "type" in spec:
            type_map[key] = spec["type"]
        elif isinstance(spec, str):
            type_map[key] = spec
    return type_map


def _type_matches(value: Any, expected_type: str) -> bool:
    """Check if a value matches a JSON-schema type string."""
    mapping = {
        "string": str,
        "str": str,
        "integer": int,
        "int": int,
        "number": (int, float),
        "float": float,
        "boolean": bool,
        "bool": bool,
        "array": list,
        "list": list,
        "object": dict,
        "dict": dict,
    }
    expected = mapping.get(expected_type)
    if expected is None:
        return True  # Unknown type — pass
    return isinstance(value, expected)


def _extract_row_count(output: dict) -> int | None:
    """Try to find a row count in the output dict."""
    for key in ("row_count", "rows", "count", "num_rows", "total_rows"):
        val = output.get(key)
        if isinstance(val, (int, float)):
            return int(val)
    # Check for a DataFrame-like structure
    for key in ("data", "records", "results"):
        val = output.get(key)
        if isinstance(val, list):
            return len(val)
    return None


# ── Adaptation Decision ─────────────────────────────────────────────────────


def decide_adaptation(
    step_state: StepState,
    config: AdaptivePlanConfig,
    agent_card: dict | None = None,
) -> AdaptationEvent | None:
    """Decide next action after a step fails quality evaluation.

    Decision tree:
        quality.passed? -> None (no action needed)
        retries < max?  -> RETRY
        is_optional?    -> SKIP
        hitl_required?  -> ESCALATE_HITL
        else            -> DEGRADE (accept, flag)
    """
    quality = step_state.quality
    if quality is None or quality.passed:
        return None  # No adaptation needed

    agent_id = step_state.agent_id
    step_order = step_state.step_order
    retry_count = len(step_state.retries)

    # Can we retry?
    if retry_count < config.max_retries_per_step:
        return AdaptationEvent(
            event_type=AdaptationType.RETRY,
            agent_id=agent_id,
            step_order=step_order,
            reason=f"Quality score {quality.overall:.2f} below threshold; "
                   f"attempt {retry_count + 1}/{config.max_retries_per_step}",
            resolution=f"Re-execute {agent_id} (attempt {retry_count + 2})",
            details={"quality_score": quality.overall, "issues": quality.issues},
        )

    # Is this step optional? (can skip)
    if step_state.is_optional:
        return AdaptationEvent(
            event_type=AdaptationType.SKIP,
            agent_id=agent_id,
            step_order=step_order,
            reason=f"Quality score {quality.overall:.2f} after {retry_count} retries; "
                   f"step is optional",
            resolution=f"Skipping {agent_id} — optional step",
            details={"quality_score": quality.overall, "issues": quality.issues},
        )

    # Does agent card require HITL?
    card = agent_card or {}
    hitl_required = card.get("hitl_required", False)
    if hitl_required:
        return AdaptationEvent(
            event_type=AdaptationType.ESCALATE_HITL,
            agent_id=agent_id,
            step_order=step_order,
            reason=f"Quality score {quality.overall:.2f} after {retry_count} retries; "
                   f"agent requires HITL escalation",
            resolution=f"Escalating {agent_id} to SA5 for human review",
            details={"quality_score": quality.overall, "issues": quality.issues},
        )

    # Last resort: degrade
    return AdaptationEvent(
        event_type=AdaptationType.DEGRADE,
        agent_id=agent_id,
        step_order=step_order,
        reason=f"Quality score {quality.overall:.2f} after {retry_count} retries; "
               f"not optional, not HITL-eligible",
        resolution=f"Accepting degraded output from {agent_id} — flagged for SA6",
        details={"quality_score": quality.overall, "issues": quality.issues},
    )


# ── Goal Progress Tracking ──────────────────────────────────────────────────


def update_goal_progress(
    is_spec: dict,
    step_states: dict[int, StepState],
    ap: dict,
) -> GoalProgress:
    """Compute goal coverage and drift signals from current step states.

    Args:
        is_spec: Intent Specification dict (needs explicit_goal at minimum).
        step_states: Dict of step_order -> StepState.
        ap: Action Plan dict.
    """
    total = len(step_states)
    completed = sum(
        1 for s in step_states.values() if s.status == StepStatus.COMPLETED
    )
    failed = sum(
        1 for s in step_states.values() if s.status == StepStatus.FAILED
    )
    skipped = sum(
        1 for s in step_states.values() if s.status == StepStatus.SKIPPED
    )

    # Coverage = (completed + skipped) / total — skipped optional steps count as handled
    coverage = (completed + skipped) / total if total > 0 else 0.0

    drift_signals: list[str] = []

    # Drift signal: high failure rate
    if total > 0 and failed / total > 0.3:
        drift_signals.append(
            f"High failure rate: {failed}/{total} steps failed "
            f"({failed/total:.0%})"
        )

    # Drift signal: too many skips on non-optional steps
    non_optional_skipped = sum(
        1 for s in step_states.values()
        if s.status == StepStatus.SKIPPED and not s.is_optional
    )
    if non_optional_skipped > 0:
        drift_signals.append(
            f"{non_optional_skipped} non-optional step(s) were skipped"
        )

    return GoalProgress(
        explicit_goal=is_spec.get("explicit_goal", ""),
        steps_total=total,
        steps_completed=completed,
        steps_failed=failed,
        steps_skipped=skipped,
        goal_coverage_pct=round(coverage * 100, 1),
        on_track=len(drift_signals) == 0,
        drift_signals=drift_signals,
    )


def check_goal_drift(
    progress: GoalProgress,
    config: AdaptivePlanConfig,
) -> list[str]:
    """Check for goal drift at midpoint or based on failure patterns.

    Returns list of drift warning strings (empty = no drift).
    """
    warnings: list[str] = []

    # Midpoint check: if we're past halfway and coverage is below threshold
    if progress.steps_total > 0:
        halfway = progress.steps_total / 2
        executed = progress.steps_completed + progress.steps_failed + progress.steps_skipped
        if executed >= halfway:
            effective_pct = progress.goal_coverage_pct / 100.0
            if effective_pct < config.goal_drift_threshold:
                warnings.append(
                    f"Midpoint drift: {progress.goal_coverage_pct:.1f}% coverage "
                    f"at {executed}/{progress.steps_total} steps executed "
                    f"(threshold: {config.goal_drift_threshold:.0%})"
                )

    # Failure rate check
    if progress.steps_total > 0:
        failure_rate = progress.steps_failed / progress.steps_total
        if failure_rate > 0.5:
            warnings.append(
                f"Critical failure rate: {progress.steps_failed}/{progress.steps_total} "
                f"steps failed ({failure_rate:.0%})"
            )

    # Append any existing drift signals
    warnings.extend(progress.drift_signals)

    return warnings


# ── State Initialization ────────────────────────────────────────────────────


def init_adaptive_state(
    ap: dict,
    config: AdaptivePlanConfig | None = None,
) -> AdaptivePlanState:
    """Bootstrap AdaptivePlanState from an Action Plan.

    Creates a StepState for each agent in the AP.
    """
    if config is None:
        config = AdaptivePlanConfig()

    step_states: dict[int, StepState] = {}
    agents = sorted(ap.get("agents", []), key=lambda a: a.get("step_order", 0))
    for agent in agents:
        step_order = agent.get("step_order", 0)
        agent_id = agent.get("agent_id", "")
        is_optional = agent.get("optional", False)
        step_states[step_order] = StepState(
            step_order=step_order,
            agent_id=agent_id,
            status=StepStatus.PENDING,
            is_optional=is_optional,
        )

    # Compute plan hash for change detection
    plan_hash = hashlib.sha256(
        json.dumps(ap, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    return AdaptivePlanState(
        config=config,
        step_states=step_states,
        original_plan_hash=plan_hash,
        plan_version=1,
        is_adapted=False,
    )


# ── Rendering ───────────────────────────────────────────────────────────────


def render_adaptation_log(events: list[AdaptationEvent]) -> str:
    """Render adaptation events as ASCII-safe text for SA6.

    Returns a human-readable summary of all adaptations taken.
    """
    if not events:
        return "No adaptations were needed during execution."

    lines: list[str] = [
        "--- Adaptation Log ---",
        f"Total adaptations: {len(events)}",
        "",
    ]

    # Group by type
    by_type: dict[str, list[AdaptationEvent]] = {}
    for evt in events:
        key = evt.event_type.value
        by_type.setdefault(key, []).append(evt)

    for etype, evts in sorted(by_type.items()):
        lines.append(f"[{etype}] x{len(evts)}")
        for evt in evts:
            lines.append(
                f"  Step {evt.step_order} ({evt.agent_id}): {evt.reason}"
            )
            lines.append(f"    -> {evt.resolution}")
        lines.append("")

    # ASCII-safe encode
    text = "\n".join(lines)
    return text.encode("ascii", errors="replace").decode("ascii")
