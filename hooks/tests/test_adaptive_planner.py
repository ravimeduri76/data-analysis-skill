"""
test_adaptive_planner.py — 32 tests for DESO Dynamic Planning and Adaptation.

All pure-Python tests (no DB dependency).

Test Classes:
    TestReadiness           5 tests — validate_agent_readiness, validate_plan_readiness
    TestQualityEvaluation   6 tests — evaluate_step_output scoring
    TestAdaptationDecision  6 tests — decide_adaptation decision tree
    TestGoalTracking        5 tests — update_goal_progress, check_goal_drift
    TestIntegration         5 tests — happy path, retry, skip, serialization, render
    TestBackwardCompat      5 tests — existing API unchanged
"""

from __future__ import annotations

import json
import sys
import os

import pytest

# Ensure hooks package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hooks.deso.plan_state import (
    AdaptationEvent,
    AdaptationType,
    AdaptivePlanConfig,
    AdaptivePlanState,
    GoalProgress,
    QualityExpectation,
    QualityScore,
    ReadinessResult,
    RetryContext,
    StepState,
    StepStatus,
)
from hooks.deso.adaptive_planner import (
    build_quality_expectations,
    check_goal_drift,
    decide_adaptation,
    evaluate_step_output,
    init_adaptive_state,
    render_adaptation_log,
    update_goal_progress,
    validate_agent_readiness,
    validate_plan_readiness,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_ap(*agents):
    """Helper to build an AP dict from (agent_id, step_order, optional) tuples."""
    return {
        "agents": [
            {"agent_id": a[0], "step_order": a[1], "optional": a[2] if len(a) > 2 else False,
             "purpose": f"Purpose of {a[0]}"}
            for a in agents
        ]
    }


def _make_agent_cards(cards: dict[str, dict]) -> dict[str, dict]:
    """Helper to build agent cards dict."""
    return cards


def _make_config(**overrides) -> AdaptivePlanConfig:
    return AdaptivePlanConfig(**overrides)


# ═══════════════════════════════════════════════════════════════════════════
# TestReadiness — 5 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReadiness:
    """Tests for validate_agent_readiness and validate_plan_readiness."""

    def test_agent_ready_when_all_inputs_available(self):
        """Agent is ready when all required inputs are in available_data."""
        result = validate_agent_readiness(
            step_order=1,
            agent_id="AG07",
            ingress_schema={"required": ["dataset", "dcp"]},
            available_data={"dataset", "dcp", "extra"},
        )
        assert result.ready is True
        assert result.missing_inputs == []
        assert "dataset" in result.available_inputs

    def test_agent_not_ready_when_inputs_missing(self):
        """Agent is not ready when required inputs are missing."""
        result = validate_agent_readiness(
            step_order=1,
            agent_id="AG07",
            ingress_schema={"required": ["dataset", "dcp", "schema_map"]},
            available_data={"dataset"},
        )
        assert result.ready is False
        assert "dcp" in result.missing_inputs
        assert "schema_map" in result.missing_inputs
        assert result.suggestion != ""

    def test_agent_ready_when_no_schema(self):
        """Agent with no ingress_schema is assumed ready (graceful default)."""
        result = validate_agent_readiness(
            step_order=1,
            agent_id="AG99",
            ingress_schema=None,
            available_data={"dataset"},
        )
        assert result.ready is True

    def test_plan_readiness_accumulates_egress(self):
        """validate_plan_readiness accumulates egress outputs as available data."""
        ap = _make_ap(("AG07", 1), ("AG08", 2))
        agent_cards = {
            "AG07": {
                "ingress_schema": {"required": ["dataset"]},
                "egress_schema": {"properties": {"profile_report": {"type": "object"}}},
            },
            "AG08": {
                "ingress_schema": {"required": ["profile_report"]},
                "egress_schema": {"properties": {"schema_map": {"type": "object"}}},
            },
        }
        results = validate_plan_readiness(ap, {"dataset"}, agent_cards)
        assert len(results) == 2
        assert results[0].ready is True   # AG07: dataset available
        assert results[1].ready is True   # AG08: profile_report from AG07

    def test_plan_readiness_detects_broken_chain(self):
        """validate_plan_readiness detects when an agent's inputs are not produced upstream."""
        ap = _make_ap(("AG07", 1), ("AG08", 2))
        agent_cards = {
            "AG07": {
                "ingress_schema": {"required": ["dataset"]},
                "egress_schema": {"properties": {"profile_report": {"type": "object"}}},
            },
            "AG08": {
                "ingress_schema": {"required": ["golden_record"]},  # Not produced by AG07
                "egress_schema": {},
            },
        }
        results = validate_plan_readiness(ap, {"dataset"}, agent_cards)
        assert results[0].ready is True
        assert results[1].ready is False
        assert "golden_record" in results[1].missing_inputs


# ═══════════════════════════════════════════════════════════════════════════
# TestQualityEvaluation — 6 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQualityEvaluation:
    """Tests for evaluate_step_output."""

    def test_full_score_all_keys_present(self):
        """Full score when all expected keys are present with correct types."""
        expectation = QualityExpectation(
            agent_id="AG07",
            min_output_keys=["report", "summary"],
        )
        output = {"report": {"data": [1, 2]}, "summary": "All good"}
        score = evaluate_step_output("AG07", 1, output, expectation)
        assert score.completeness == 1.0
        assert score.passed is True

    def test_partial_keys_reduces_completeness(self):
        """Missing keys reduce completeness score."""
        expectation = QualityExpectation(
            agent_id="AG07",
            min_output_keys=["report", "summary", "metadata", "stats"],
        )
        output = {"report": {"data": []}, "summary": "partial"}
        score = evaluate_step_output("AG07", 1, output, expectation)
        assert score.completeness == 0.5  # 2/4 keys
        assert len(score.issues) > 0

    def test_type_mismatch_reduces_validity(self):
        """Type mismatches in egress_schema reduce validity score."""
        expectation = QualityExpectation(
            agent_id="AG07",
            min_output_keys=["count", "name"],
        )
        egress_schema = {
            "properties": {
                "count": {"type": "integer"},
                "name": {"type": "string"},
            }
        }
        output = {"count": "not_a_number", "name": "valid"}
        score = evaluate_step_output(
            "AG07", 1, output, expectation, egress_schema=egress_schema
        )
        assert score.validity == 0.5  # 1/2 valid
        assert any("Type mismatch" in i for i in score.issues)

    def test_threshold_determines_pass(self):
        """Score below threshold fails, at-or-above passes."""
        expectation = QualityExpectation(
            agent_id="AG07",
            min_output_keys=["a", "b"],
        )
        output = {"a": 1}  # 50% completeness → overall = 0.5*0.5 + 0.3*1.0 + 0.2*1.0 = 0.75
        score_low = evaluate_step_output("AG07", 1, output, expectation, threshold=0.8)
        score_high = evaluate_step_output("AG07", 1, output, expectation, threshold=0.7)
        assert score_low.passed is False
        assert score_high.passed is True

    def test_empty_output_scores_zero(self):
        """Empty or None output scores all zeros."""
        expectation = QualityExpectation(agent_id="AG07")
        score_none = evaluate_step_output("AG07", 1, None, expectation)
        score_empty = evaluate_step_output("AG07", 1, {}, expectation)
        assert score_none.overall == 0.0
        assert score_none.passed is False
        assert score_empty.overall == 0.0

    def test_row_count_affects_volume(self):
        """Row count below min_row_count reduces volume score."""
        expectation = QualityExpectation(
            agent_id="AG07",
            min_output_keys=["data"],
            min_row_count=100,
        )
        output = {"data": list(range(50)), "row_count": 50}
        score = evaluate_step_output("AG07", 1, output, expectation)
        assert score.volume == 0.5  # 50/100
        assert score.overall < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TestAdaptationDecision — 6 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAdaptationDecision:
    """Tests for decide_adaptation decision tree."""

    def test_no_adapt_when_passed(self):
        """No adaptation when quality passed."""
        step = StepState(
            step_order=1, agent_id="AG07",
            quality=QualityScore(passed=True, overall=0.8),
        )
        event = decide_adaptation(step, _make_config())
        assert event is None

    def test_no_adapt_when_no_quality(self):
        """No adaptation when quality is None (not yet evaluated)."""
        step = StepState(step_order=1, agent_id="AG07")
        event = decide_adaptation(step, _make_config())
        assert event is None

    def test_retry_when_under_max(self):
        """RETRY when retries < max and quality failed."""
        step = StepState(
            step_order=1, agent_id="AG07",
            quality=QualityScore(passed=False, overall=0.4),
            retries=[],  # 0 retries so far
        )
        event = decide_adaptation(step, _make_config(max_retries_per_step=2))
        assert event is not None
        assert event.event_type == AdaptationType.RETRY

    def test_skip_optional_after_max_retries(self):
        """SKIP when retries exhausted and step is optional."""
        step = StepState(
            step_order=1, agent_id="AG07",
            quality=QualityScore(passed=False, overall=0.3),
            retries=[RetryContext(attempt=1), RetryContext(attempt=2)],
            is_optional=True,
        )
        event = decide_adaptation(step, _make_config(max_retries_per_step=2))
        assert event is not None
        assert event.event_type == AdaptationType.SKIP

    def test_escalate_hitl_after_retries(self):
        """ESCALATE_HITL when retries exhausted and agent requires HITL."""
        step = StepState(
            step_order=1, agent_id="AG10",
            quality=QualityScore(passed=False, overall=0.3),
            retries=[RetryContext(attempt=1), RetryContext(attempt=2)],
            is_optional=False,
        )
        agent_card = {"hitl_required": True}
        event = decide_adaptation(step, _make_config(max_retries_per_step=2), agent_card)
        assert event is not None
        assert event.event_type == AdaptationType.ESCALATE_HITL

    def test_degrade_as_last_resort(self):
        """DEGRADE when retries exhausted, not optional, not HITL."""
        step = StepState(
            step_order=1, agent_id="AG07",
            quality=QualityScore(passed=False, overall=0.3),
            retries=[RetryContext(attempt=1), RetryContext(attempt=2)],
            is_optional=False,
        )
        event = decide_adaptation(step, _make_config(max_retries_per_step=2))
        assert event is not None
        assert event.event_type == AdaptationType.DEGRADE


# ═══════════════════════════════════════════════════════════════════════════
# TestGoalTracking — 5 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGoalTracking:
    """Tests for update_goal_progress and check_goal_drift."""

    def test_all_complete(self):
        """All steps completed → 100% coverage, on track."""
        step_states = {
            1: StepState(step_order=1, status=StepStatus.COMPLETED),
            2: StepState(step_order=2, status=StepStatus.COMPLETED),
            3: StepState(step_order=3, status=StepStatus.COMPLETED),
        }
        progress = update_goal_progress(
            {"explicit_goal": "Profile data"}, step_states, {}
        )
        assert progress.goal_coverage_pct == 100.0
        assert progress.on_track is True
        assert progress.steps_failed == 0

    def test_with_failures(self):
        """Failed steps reduce coverage and may trigger drift signals."""
        step_states = {
            1: StepState(step_order=1, status=StepStatus.COMPLETED),
            2: StepState(step_order=2, status=StepStatus.FAILED),
            3: StepState(step_order=3, status=StepStatus.COMPLETED),
        }
        progress = update_goal_progress(
            {"explicit_goal": "Profile data"}, step_states, {}
        )
        assert progress.steps_failed == 1
        # 2/3 completed + 0 skipped = 66.7%
        assert progress.goal_coverage_pct == pytest.approx(66.7, abs=0.1)

    def test_drift_at_midpoint(self):
        """Drift signal when coverage is low at midpoint."""
        step_states = {
            1: StepState(step_order=1, status=StepStatus.FAILED),
            2: StepState(step_order=2, status=StepStatus.FAILED),
            3: StepState(step_order=3, status=StepStatus.PENDING),
            4: StepState(step_order=4, status=StepStatus.PENDING),
        }
        progress = update_goal_progress(
            {"explicit_goal": "Profile data"}, step_states, {}
        )
        config = _make_config(goal_drift_threshold=0.5)
        warnings = check_goal_drift(progress, config)
        assert len(warnings) > 0
        assert any("drift" in w.lower() or "failure" in w.lower() for w in warnings)

    def test_no_drift_when_on_track(self):
        """No drift warnings when everything is going well."""
        progress = GoalProgress(
            steps_total=4, steps_completed=3, steps_failed=0,
            steps_skipped=0, goal_coverage_pct=75.0, on_track=True,
        )
        config = _make_config(goal_drift_threshold=0.5)
        warnings = check_goal_drift(progress, config)
        assert len(warnings) == 0

    def test_high_failure_rate_triggers_drift(self):
        """More than 50% failure rate triggers critical drift warning."""
        step_states = {
            1: StepState(step_order=1, status=StepStatus.FAILED),
            2: StepState(step_order=2, status=StepStatus.FAILED),
            3: StepState(step_order=3, status=StepStatus.FAILED),
            4: StepState(step_order=4, status=StepStatus.COMPLETED),
        }
        progress = update_goal_progress(
            {"explicit_goal": "Profile data"}, step_states, {}
        )
        config = _make_config()
        warnings = check_goal_drift(progress, config)
        assert any("failure rate" in w.lower() for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════
# TestIntegration — 5 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests combining multiple adaptive planner functions."""

    def test_happy_path_no_adaptation(self):
        """Full flow: init → evaluate (pass) → no adaptation → progress."""
        ap = _make_ap(("AG07", 1), ("AG08", 2))
        state = init_adaptive_state(ap)
        assert len(state.step_states) == 2
        assert state.step_states[1].status == StepStatus.PENDING

        # Simulate successful execution
        exp = QualityExpectation(agent_id="AG07", min_output_keys=["report"])
        score = evaluate_step_output(
            "AG07", 1, {"report": "ok"}, exp, threshold=0.6
        )
        assert score.passed is True

        state.step_states[1].quality = score
        state.step_states[1].status = StepStatus.COMPLETED

        event = decide_adaptation(state.step_states[1], state.config)
        assert event is None  # No adaptation needed

    def test_retry_flow(self):
        """Flow: evaluate (fail) → retry decision → re-evaluate (pass)."""
        config = _make_config(max_retries_per_step=2)
        step = StepState(
            step_order=1, agent_id="AG07",
            quality=QualityScore(passed=False, overall=0.3),
        )
        event = decide_adaptation(step, config)
        assert event.event_type == AdaptationType.RETRY

        # Simulate retry
        step.retries.append(RetryContext(attempt=1, previous_quality=step.quality))
        step.quality = QualityScore(passed=True, overall=0.8)
        event2 = decide_adaptation(step, config)
        assert event2 is None  # Now passing

    def test_skip_flow(self):
        """Flow: 2 retries exhausted → skip optional step."""
        config = _make_config(max_retries_per_step=2)
        step = StepState(
            step_order=1, agent_id="AG07",
            quality=QualityScore(passed=False, overall=0.2),
            retries=[RetryContext(attempt=1), RetryContext(attempt=2)],
            is_optional=True,
        )
        event = decide_adaptation(step, config)
        assert event.event_type == AdaptationType.SKIP

    def test_serialization_roundtrip(self):
        """AdaptivePlanState serializes and deserializes correctly."""
        ap = _make_ap(("AG07", 1), ("AG08", 2))
        state = init_adaptive_state(ap)
        state.adaptation_log.append(AdaptationEvent(
            event_type=AdaptationType.RETRY,
            agent_id="AG07", step_order=1,
            reason="test", resolution="re-run",
        ))
        state.goal_progress = GoalProgress(
            explicit_goal="test", steps_total=2,
        )

        data = state.to_dict()
        json_str = json.dumps(data, default=str)
        restored = AdaptivePlanState.from_dict(json.loads(json_str))

        assert len(restored.step_states) == 2
        assert len(restored.adaptation_log) == 1
        assert restored.adaptation_log[0].event_type == AdaptationType.RETRY
        assert restored.goal_progress.explicit_goal == "test"

    def test_render_adaptation_log(self):
        """render_adaptation_log produces ASCII-safe output."""
        events = [
            AdaptationEvent(
                event_type=AdaptationType.RETRY,
                agent_id="AG07", step_order=1,
                reason="Low quality", resolution="Re-execute",
            ),
            AdaptationEvent(
                event_type=AdaptationType.SKIP,
                agent_id="AG08", step_order=2,
                reason="Optional failed", resolution="Skipped",
            ),
        ]
        text = render_adaptation_log(events)
        assert "Adaptation Log" in text
        assert "RETRY" in text
        assert "SKIP" in text
        # Verify ASCII-safe
        assert text == text.encode("ascii", errors="replace").decode("ascii")


# ═══════════════════════════════════════════════════════════════════════════
# TestBackwardCompat — 5 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardCompat:
    """Ensure existing DESORunner API is unchanged when adaptive mode is NOT enabled."""

    def test_existing_register_step_works(self):
        """register_step with existing Step values still works."""
        from hooks.deso_runner import DESORunner, Step
        runner = DESORunner("test query")
        runner.run_preflight()
        runner.register_step(Step.SA1, "DCP", {
            "domain": "Healthcare",
            "domain_confidence": "high",
            "lifecycle_stage": "assessment",
            "graph_available": False,
        })
        assert Step.SA1 in runner._completed_steps

    def test_adaptive_methods_noop_without_enable(self):
        """Adaptive methods return safe defaults when adaptive mode not enabled."""
        from hooks.deso_runner import DESORunner
        runner = DESORunner("test query")
        runner.run_preflight()

        ready, results = runner.check_readiness({}, {})
        assert ready is True
        assert results == []

        score = runner.evaluate_step("AG07", 1, {"report": "ok"})
        assert score is None

        event = runner.decide_next(1)
        assert event is None

        progress = runner.update_progress({"explicit_goal": "test"})
        assert progress is None

        log = runner.get_adaptation_log()
        assert log == []

        state = runner.get_adaptive_state()
        assert state is None

    def test_audit_unchanged_without_adaptive(self):
        """Audit returns 6 checks (not 8) when adaptive mode is off."""
        from hooks.deso_runner import DESORunner
        runner = DESORunner("test query")
        runner.run_preflight()
        result = runner.audit()
        assert len(result["checks"]) == 6  # Original 6 checks only

    def test_audit_adds_checks_with_adaptive(self):
        """Audit returns 8 checks when adaptive mode is enabled."""
        from hooks.deso_runner import DESORunner
        runner = DESORunner("test query")
        runner.run_preflight()
        runner.enable_adaptive()
        result = runner.audit()
        assert len(result["checks"]) == 8  # 6 original + 2 adaptive

    def test_get_state_still_works(self):
        """get_state returns expected structure."""
        from hooks.deso_runner import DESORunner
        runner = DESORunner("test query")
        runner.run_preflight()
        state = runner.get_state()
        assert "completed_steps" in state
        assert "PF" in state["completed_steps"]
        assert "packets" in state
        assert "health" in state
