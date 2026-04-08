"""
deso_runner.py -- Thin Python state machine for DESO orchestration validation.

The LLM calls this at each step to verify compliance. Does NOT call LLM --
only validates and tracks.

Public API
----------
DESORunner(user_query, dataset_columns=None)
    .run_preflight()          -> dict  (app_id, solution_id, method)
    .validate_packet(name, data) -> (bool, list[str])
    .register_step(step, packet_name=None, packet_data=None)
    .validate_plan(ap, dcp, is_spec) -> (bool, list[str])
    .audit()                  -> dict  (checks, score)
    .get_state()              -> dict

Adaptive mode (opt-in via .enable_adaptive()):
    .enable_adaptive(config=None) -> None
    .check_readiness(ap, agent_cards, initial_data=None) -> (bool, list)
    .evaluate_step(agent_id, step_order, output) -> QualityScore
    .decide_next(step_order, agent_card=None) -> AdaptationEvent | None
    .update_progress(is_spec) -> GoalProgress
    .get_adaptation_log()     -> list[dict]
    .get_adaptive_state()     -> dict
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class Step(Enum):
    PREFLIGHT = "PF"
    SA1 = "SA1"
    SA2 = "SA2"
    SA3 = "SA3"
    SA4 = "SA4"
    SA5_PLAN = "SA5_PLAN"
    SA5_ASSESS = "SA5_ASSESS"
    EXECUTE = "EXECUTE"
    SA6 = "SA6"
    # Adaptive mode steps
    READINESS_CHECK = "RC"
    EXECUTE_ADAPTIVE = "EXEC_A"
    ADAPT = "ADAPT"


PREREQUISITES: dict[Step, set[Step]] = {
    Step.SA1: {Step.PREFLIGHT},
    Step.SA2: {Step.SA1},
    Step.SA3: {Step.SA2},
    Step.SA4: {Step.SA3},
    Step.EXECUTE: {Step.SA4},
    Step.SA6: {Step.EXECUTE},
    # Adaptive mode prerequisites
    Step.READINESS_CHECK: {Step.SA3},
    Step.EXECUTE_ADAPTIVE: {Step.SA4},
}

PACKET_SCHEMAS: dict[str, dict[str, Any]] = {
    "DCP": {
        "required": ["domain", "domain_confidence", "lifecycle_stage", "graph_available"],
        "types": {
            "domain": str,
            "domain_confidence": str,
            "lifecycle_stage": str,
            "graph_available": bool,
        },
    },
    "IS": {
        "required": ["explicit_goal", "requires_hitl"],
        "types": {
            "explicit_goal": str,
            "requires_hitl": bool,
        },
    },
    "AP": {
        "required": ["agents"],
        "types": {
            "agents": list,
        },
    },
    "ES": {
        "required": ["interaction_mode"],
        "types": {
            "interaction_mode": str,
        },
    },
}

# Domain signal keywords — if these appear in query, base app is suspicious
_DOMAIN_KEYWORDS = {
    "product", "catalog", "syndication", "healthcare", "patient",
    "medication", "customer", "mdm", "master data", "harmonize",
    "schema", "enrichment", "golden record", "amazon", "gpo",
}

# Agent IDs that handle domain knowledge during transformation
_DOMAIN_AGENTS = {"AG08", "AG09", "AG24", "AG25", "AG26"}

# Agent IDs that handle invariant checking
_INVARIANT_AGENTS = {"AG17"}


class DESORunner:
    """Lightweight state machine for DESO orchestration compliance."""

    def __init__(self, user_query: str, dataset_columns: list[str] | None = None) -> None:
        self.user_query = user_query
        self.dataset_columns = dataset_columns or []
        self._completed_steps: set[Step] = set()
        self._packets: dict[str, dict] = {}
        self._preflight_result: dict | None = None
        self._step_log: list[dict] = []
        self._created_at = datetime.now(timezone.utc).isoformat()

        # L1 session context — lazy-initialized on first use
        self._session: Any = None  # SessionContext from hooks.deso.memory
        self._conn = None  # Lazy DB connection for L2/L3

        # Adaptive mode state — None until enable_adaptive() is called
        self._adaptive_state: Any = None  # AdaptivePlanState
        self._quality_expectations: dict[str, Any] = {}  # agent_id -> QualityExpectation

        # RL mode state — None until enable_rl() is called
        self._rl_policy: Any = None  # ThompsonSamplingPolicy
        self._rl_logger: Any = None  # RLLogger
        self._rl_config: Any = None  # RLConfig

    @property
    def session(self):
        """L1 SessionContext, created on first access."""
        if self._session is None:
            from hooks.deso.memory import SessionContext
            import uuid
            self._session = SessionContext(
                run_id=f"deso_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                session_id=f"s_{uuid.uuid4().hex[:12]}",
                user_query=self.user_query,
                dataset_columns=self.dataset_columns,
                created_at=self._created_at,
            )
        return self._session

    def _get_conn(self):
        """Lazy, best-effort DB connection. Returns None if unavailable."""
        if self._conn is not None:
            return self._conn
        try:
            from hooks.deso.memory import get_session_conn, init_session_db
            self._conn = get_session_conn()
            init_session_db(self._conn)
        except Exception as exc:
            log.debug("Session DB unavailable (L2/L3 disabled): %s", exc)
            self._conn = None
        return self._conn

    # ── Pre-flight ─────────────────────────────────────────────────────────

    def run_preflight(self) -> dict:
        """Execute PF-1 through PF-4. Returns {app_id, solution_id, method}."""
        app_id = "base"
        solution_id = None
        method = "fallback"

        try:
            from hooks.app_registry import resolve_app_from_query
            app_id, solution_id = resolve_app_from_query(
                self.user_query, dataset_columns=self.dataset_columns
            )
            method = "resolved"
        except ImportError:
            pass
        except Exception:
            pass

        self._preflight_result = {
            "app_id": app_id,
            "solution_id": solution_id,
            "method": method,
        }
        self._completed_steps.add(Step.PREFLIGHT)
        self._log_step(Step.PREFLIGHT, "Pre-flight completed")

        # Sync to L1 session
        self.session.app_id = app_id
        self.session.solution_id = solution_id

        return self._preflight_result

    # ── Packet validation ──────────────────────────────────────────────────

    def validate_packet(self, name: str, data: dict) -> tuple[bool, list[str]]:
        """Schema check: required keys present, correct types. Returns (valid, errors)."""
        schema = PACKET_SCHEMAS.get(name)
        if schema is None:
            return True, []  # Unknown packet — no schema to validate

        errors: list[str] = []
        for key in schema["required"]:
            if key not in data:
                errors.append(f"{name} missing required key: {key}")
            elif key in schema.get("types", {}) and not isinstance(data[key], schema["types"][key]):
                expected = schema["types"][key].__name__
                actual = type(data[key]).__name__
                errors.append(f"{name}.{key}: expected {expected}, got {actual}")

        return len(errors) == 0, errors

    # ── Step registration ──────────────────────────────────────────────────

    def register_step(
        self,
        step: Step,
        packet_name: str | None = None,
        packet_data: dict | None = None,
    ) -> None:
        """Record step completion. Enforces prerequisites (raises ValueError)."""
        required = PREREQUISITES.get(step, set())
        missing = required - self._completed_steps
        if missing:
            missing_names = ", ".join(s.value for s in sorted(missing, key=lambda s: s.value))
            raise ValueError(
                f"Cannot register {step.value}: prerequisites not met — missing {missing_names}"
            )

        if packet_name and packet_data is not None:
            valid, errors = self.validate_packet(packet_name, packet_data)
            if not valid:
                raise ValueError(f"Packet validation failed for {packet_name}: {'; '.join(errors)}")
            self._packets[packet_name] = copy.deepcopy(packet_data)

        self._completed_steps.add(step)
        self._log_step(step, f"Registered {step.value}" + (f" with {packet_name}" if packet_name else ""))

        # Sync packet data into L1 session context
        self._sync_session(step, packet_name, packet_data)

    # ── Plan Validation Gate ───────────────────────────────────────────────

    def validate_plan(self, ap: dict, dcp: dict, is_spec: dict) -> tuple[bool, list[str]]:
        """4-rule plan validation gate. Returns (passed, gaps)."""
        gaps: list[str] = []
        ap_agent_ids = {
            a.get("agent_id", "") for a in ap.get("agents", [])
        }

        # Rule 1: DCP has graph_entities + transformation lifecycle → need domain agents
        has_graph_entities = bool(dcp.get("graph_entities"))
        lifecycle = dcp.get("lifecycle_stage", "")
        is_transformation = lifecycle in ("preparation", "enrichment", "publishing")
        if has_graph_entities and is_transformation:
            needed = _DOMAIN_AGENTS
            present = ap_agent_ids & needed
            if not present:
                gaps.append(
                    f"DCP has graph_entities + {lifecycle} lifecycle but AP has none of "
                    f"{sorted(needed)}. Add AG08/AG09/AG24/AG25/AG26."
                )

        # Rule 2: DCP has invariants → AP must include AG17 or invariant check
        if dcp.get("invariants"):
            invariant_present = ap_agent_ids & _INVARIANT_AGENTS
            # Also check if any agent has "invariant" in its role/purpose
            has_invariant_check = any(
                "invariant" in str(a.get("purpose", "")).lower() or
                "invariant" in str(a.get("role", "")).lower()
                for a in ap.get("agents", [])
            )
            if not invariant_present and not has_invariant_check:
                gaps.append(
                    "DCP has invariants but AP missing AG17 or explicit invariant check."
                )

        # Rule 3: Each IS risk_signal → must map to AP blocking_risks
        risk_signals = is_spec.get("risk_signals", [])
        blocking_risks = ap.get("blocking_risks", [])
        blocking_text = " ".join(str(r) for r in blocking_risks).lower()
        for signal in risk_signals:
            # Check if the signal is mentioned in blocking risks
            signal_words = set(signal.lower().split())
            if not any(w in blocking_text for w in signal_words if len(w) > 3):
                gaps.append(
                    f"IS risk_signal '{signal}' has no matching mitigation in AP blocking_risks."
                )

        # Rule 4: Each required IS missing_input → must have HITL gate
        missing_inputs = is_spec.get("missing_inputs", [])
        hitl_gates = ap.get("hitl_gates", [])
        hitl_text = " ".join(str(g) for g in hitl_gates).lower()
        for mi in missing_inputs:
            mi_words = set(mi.lower().split())
            if not any(w in hitl_text for w in mi_words if len(w) > 3):
                gaps.append(
                    f"IS missing_input '{mi}' has no HITL gate in AP."
                )

        return len(gaps) == 0, gaps

    # ── Session Memory ────────────────────────────────────────────────────

    def _sync_session(
        self, step: Step, packet_name: str | None, packet_data: dict | None,
    ) -> None:
        """Populate L1 SessionContext fields from registered packet data."""
        if packet_data is None:
            return
        try:
            from hooks.deso.memory import (
                DomainContextPacket, IntentSummary, ActionPlan,
                ExecSchedule, HITLPacket,
            )
        except ImportError:
            return

        ctx = self.session
        ctx.turn_count += 1

        if packet_name == "DCP":
            ctx.dcp = DomainContextPacket(
                **{k: v for k, v in packet_data.items()
                   if k in DomainContextPacket.__dataclass_fields__}
            )
        elif packet_name == "IS":
            ctx.intent = IntentSummary(
                **{k: v for k, v in packet_data.items()
                   if k in IntentSummary.__dataclass_fields__}
            )
        elif packet_name == "AP":
            ctx.plan = ActionPlan(
                **{k: v for k, v in packet_data.items()
                   if k in ActionPlan.__dataclass_fields__}
            )
        elif packet_name == "ES":
            ctx.schedule = ExecSchedule(
                **{k: v for k, v in packet_data.items()
                   if k in ExecSchedule.__dataclass_fields__}
            )
        elif packet_name in ("HP", "UD"):
            ctx.hitl_packets.append(HITLPacket(
                **{k: v for k, v in packet_data.items()
                   if k in HITLPacket.__dataclass_fields__}
            ))

    def checkpoint(self) -> None:
        """Persist L1 → L2 (best-effort, never raises)."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            from hooks.deso.memory import checkpoint as _checkpoint
            _checkpoint(conn, self.session)
        except Exception as exc:
            log.debug("Checkpoint failed (non-fatal): %s", exc)

    def finalize(self, health_score: int) -> None:
        """Promote L2 → L3 and clean up (best-effort, never raises)."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            from hooks.deso.memory import promote
            promote(conn, self.session, health_score)
        except Exception as exc:
            log.debug("Finalize failed (non-fatal): %s", exc)

        # RL finalization (best-effort)
        self.rl_finalize(health_score)

    # ── Adaptive Mode ────────────────────────────────────────────────────

    def enable_adaptive(self, config: Any = None) -> None:
        """Opt-in to adaptive execution mode. Creates AdaptivePlanState.

        All adaptive methods are no-ops if this is never called.

        Args:
            config: AdaptivePlanConfig instance, or None for defaults.
        """
        from hooks.deso.plan_state import AdaptivePlanConfig, AdaptivePlanState
        if config is None:
            config = AdaptivePlanConfig()
        self._adaptive_state = AdaptivePlanState(config=config)

    def check_readiness(
        self,
        ap: dict,
        agent_cards: dict[str, dict],
        initial_data: set[str] | None = None,
    ) -> tuple[bool, list[dict]]:
        """Step 4c: Validate all agent inputs flow correctly using ingress/egress schemas.

        Always validates readiness (even without adaptive mode). Adaptive-only
        logic (step state init, quality expectations) runs only when adaptive
        mode is enabled.

        Args:
            ap: Action Plan dict.
            agent_cards: Dict of agent_id -> agent card dicts.
            initial_data: Set of data keys available before agents run.

        Returns:
            (all_ready, list of ReadinessResult dicts)
        """
        try:
            from hooks.deso.adaptive_planner import (
                validate_plan_readiness,
                build_quality_expectations,
                init_adaptive_state,
            )
        except ImportError:
            log.debug("adaptive_planner not available — skipping readiness check")
            return True, []

        # Adaptive-only: initialize step states and quality expectations
        if self._adaptive_state is not None:
            self._adaptive_state = init_adaptive_state(ap, self._adaptive_state.config)
            self._quality_expectations = build_quality_expectations(ap, agent_cards)

        # Always validate readiness
        results = validate_plan_readiness(
            ap, initial_data or set(), agent_cards
        )

        all_ready = all(r.ready for r in results)

        # Register readiness check step
        self._completed_steps.add(Step.READINESS_CHECK)
        self._log_step(
            Step.READINESS_CHECK,
            f"Readiness check: {'PASSED' if all_ready else 'FAILED'} "
            f"({sum(1 for r in results if not r.ready)} agents not ready)"
        )

        return all_ready, [r.to_dict() for r in results]

    def evaluate_step(
        self,
        agent_id: str,
        step_order: int,
        output: dict | None,
    ) -> Any:
        """Post-step quality scoring. Updates StepState in adaptive state.

        No-op returning None if adaptive mode not enabled.

        Returns:
            QualityScore or None.
        """
        if self._adaptive_state is None:
            return None

        from hooks.deso.adaptive_planner import evaluate_step_output
        from hooks.deso.plan_state import StepStatus, RetryContext

        expectation = self._quality_expectations.get(agent_id)
        if expectation is None:
            from hooks.deso.plan_state import QualityExpectation
            expectation = QualityExpectation(agent_id=agent_id)

        # Look up egress_schema if available in AP agents
        egress_schema = None
        # We don't have agent_cards stored, but expectation was built from them
        # Use threshold from config
        threshold = self._adaptive_state.config.quality_threshold

        score = evaluate_step_output(
            agent_id=agent_id,
            step_order=step_order,
            output=output,
            expectation=expectation,
            egress_schema=egress_schema,
            threshold=threshold,
        )

        # Update step state
        step_state = self._adaptive_state.step_states.get(step_order)
        if step_state is not None:
            step_state.quality = score
            step_state.output = output
            step_state.status = (
                StepStatus.COMPLETED if score.passed else StepStatus.FAILED
            )

        return score

    def decide_next(
        self, step_order: int, agent_card: dict | None = None,
    ) -> Any:
        """Decide retry/skip/escalate/degrade after quality eval.

        No-op returning None if adaptive mode not enabled.

        Returns:
            AdaptationEvent or None (None = no adaptation needed).
        """
        if self._adaptive_state is None:
            return None

        from hooks.deso.adaptive_planner import decide_adaptation
        from hooks.deso.plan_state import (
            StepStatus,
            RetryContext,
        )

        step_state = self._adaptive_state.step_states.get(step_order)
        if step_state is None:
            return None

        event = decide_adaptation(
            step_state=step_state,
            config=self._adaptive_state.config,
            agent_card=agent_card,
        )

        if event is not None:
            self._adaptive_state.adaptation_log.append(event)
            self._adaptive_state.is_adapted = True

            # Update step status based on decision
            if event.event_type.value == "RETRY":
                step_state.retries.append(RetryContext(
                    attempt=len(step_state.retries) + 1,
                    previous_quality=step_state.quality,
                ))
                step_state.status = StepStatus.RETRYING
            elif event.event_type.value == "SKIP":
                step_state.status = StepStatus.SKIPPED
            elif event.event_type.value == "ESCALATE_HITL":
                step_state.status = StepStatus.BLOCKED
                step_state.blocked_reason = "Escalated to SA5 for human review"
            elif event.event_type.value == "DEGRADE":
                step_state.status = StepStatus.COMPLETED  # Accept degraded

            self._log_step(
                Step.ADAPT,
                f"Adaptation: {event.event_type.value} for {event.agent_id} "
                f"at step {step_order}"
            )

            return event

        return None

    def update_progress(self, is_spec: dict) -> Any:
        """Update GoalProgress and check drift. Returns GoalProgress or None.

        No-op returning None if adaptive mode not enabled.
        """
        if self._adaptive_state is None:
            return None

        from hooks.deso.adaptive_planner import update_goal_progress, check_goal_drift

        progress = update_goal_progress(
            is_spec=is_spec,
            step_states=self._adaptive_state.step_states,
            ap=self._packets.get("AP", {}),
        )

        # Check for drift
        drift_warnings = check_goal_drift(progress, self._adaptive_state.config)
        if drift_warnings:
            progress.drift_signals = drift_warnings
            progress.on_track = False

        self._adaptive_state.goal_progress = progress
        return progress

    def get_adaptation_log(self) -> list[dict]:
        """Return adaptation events as list of dicts. Empty if adaptive mode not enabled."""
        if self._adaptive_state is None:
            return []
        return [e.to_dict() for e in self._adaptive_state.adaptation_log]

    def get_adaptive_state(self) -> dict | None:
        """Return full adaptive state as dict for L2 checkpoint. None if not enabled."""
        if self._adaptive_state is None:
            return None
        return self._adaptive_state.to_dict()

    # ── Self-Audit ─────────────────────────────────────────────────────────

    def audit(self) -> dict:
        """6-check self-audit. Returns {checks: [...], orchestration_health_score: 0-100}."""
        checks: list[dict] = []
        score = 100

        # Check 1: Pre-flight completed (-20 if skipped)
        pf_ok = Step.PREFLIGHT in self._completed_steps
        checks.append({
            "check": "Pre-flight completed",
            "passed": pf_ok,
            "weight": -20,
        })
        if not pf_ok:
            score -= 20

        # Check 2: App resolution not base with domain signals (-15)
        app_id = (self._preflight_result or {}).get("app_id", "base")
        has_domain_signals = any(
            kw in self.user_query.lower() for kw in _DOMAIN_KEYWORDS
        )
        app_ok = not (app_id == "base" and has_domain_signals)
        checks.append({
            "check": "App resolution not base with domain signals",
            "passed": app_ok,
            "weight": -15,
        })
        if not app_ok:
            score -= 15

        # Check 3: Domain agents in plan when graph available (-20)
        dcp = self._packets.get("DCP", {})
        ap = self._packets.get("AP", {})
        graph_available = dcp.get("graph_available", False)
        ap_agent_ids = {a.get("agent_id", "") for a in ap.get("agents", [])}
        domain_agents_present = bool(ap_agent_ids & _DOMAIN_AGENTS)
        graph_ok = not graph_available or domain_agents_present
        checks.append({
            "check": "Domain agents in plan when graph available",
            "passed": graph_ok,
            "weight": -20,
        })
        if not graph_ok:
            score -= 20

        # Check 4: All HITL gates presented (-20)
        es = self._packets.get("ES", {})
        hitl_checkpoints = es.get("hitl_checkpoints", [])
        sa5_ran = Step.SA5_PLAN in self._completed_steps
        hitl_ok = not hitl_checkpoints or sa5_ran
        checks.append({
            "check": "All HITL gates presented",
            "passed": hitl_ok,
            "weight": -20,
        })
        if not hitl_ok:
            score -= 20

        # Check 5: DCP invariants checked (-15)
        has_invariants = bool(dcp.get("invariants"))
        invariant_agents = ap_agent_ids & _INVARIANT_AGENTS
        inv_ok = not has_invariants or bool(invariant_agents)
        checks.append({
            "check": "DCP invariants checked",
            "passed": inv_ok,
            "weight": -15,
        })
        if not inv_ok:
            score -= 15

        # Check 6: Context packets saved to trace (-10)
        packets_ok = len(self._packets) >= 3  # At minimum DCP, IS, AP
        checks.append({
            "check": "Context packets saved to trace",
            "passed": packets_ok,
            "weight": -10,
        })
        if not packets_ok:
            score -= 10

        # Adaptive-mode checks (only score when adaptive mode is enabled)
        if self._adaptive_state is not None:
            # Check 7: Readiness check ran (-10)
            rc_ok = Step.READINESS_CHECK in self._completed_steps
            checks.append({
                "check": "Readiness check ran",
                "passed": rc_ok,
                "weight": -10,
            })
            if not rc_ok:
                score -= 10

            # Check 8: No unresolved degradations (-10)
            from hooks.deso.plan_state import AdaptationType
            degradations = [
                e for e in self._adaptive_state.adaptation_log
                if e.event_type == AdaptationType.DEGRADE
            ]
            degrade_ok = len(degradations) == 0
            checks.append({
                "check": "No unresolved degradations",
                "passed": degrade_ok,
                "weight": -10,
            })
            if not degrade_ok:
                score -= 10

        return {
            "checks": checks,
            "orchestration_health_score": max(0, score),
        }

    # ── RL Mode ───────────────────────────────────────────────────────────

    def enable_rl(self, config: Any = None) -> None:
        """Opt-in to RL feedback loop. Loads active policy from DB.

        All RL methods are no-ops if this is never called.

        Args:
            config: RLConfig instance, or None for defaults.
        """
        from hooks.deso.rl.rl_state import RLConfig
        from hooks.deso.rl.policy import ThompsonSamplingPolicy
        from hooks.deso.rl.logger import RLLogger

        if config is None:
            config = RLConfig()
        self._rl_config = config

        # Resolve app_id from session
        app_id = "base"
        if self._session:
            app_id = getattr(self._session, "app_id", "base") or "base"

        # Load policy from DB (best-effort), scoped by app_id
        conn = self._get_conn()
        if conn:
            try:
                from hooks.deso.rl.rl_store import init_rl_db, load_active_policy
                init_rl_db(conn)
                self._rl_policy = ThompsonSamplingPolicy.load(conn, config=config)
            except Exception as exc:
                log.debug("RL policy load failed, using fresh: %s", exc)
                self._rl_policy = ThompsonSamplingPolicy(config=config)
        else:
            self._rl_policy = ThompsonSamplingPolicy(config=config)

        run_id = ""
        if self._session:
            run_id = getattr(self._session, "run_id", "") or getattr(self._session, "session_id", "")
        if not run_id:
            run_id = self.user_query[:50]
        self._rl_logger = RLLogger(run_id=run_id, policy_version=self._rl_policy._version, app_id=app_id)

    def rl_select_action(
        self,
        step_label: str,
        category: Any,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> Any:
        """Ask RL policy for a recommendation at a decision point.

        Returns ActionType or None (if RL not enabled or insufficient experience).
        """
        if self._rl_policy is None or self._rl_logger is None:
            return None

        from hooks.deso.rl.state_builder import build_state_snapshot
        from hooks.deso.rl.policy import filter_safe_actions

        state = build_state_snapshot(self, step_label, latency_ms, cost_usd)

        # Filter to safe actions
        dcp = self._packets.get("DCP")
        safe = filter_safe_actions(category, dcp=dcp, config=self._rl_config)

        # Select action
        action = self._rl_policy.select_action(category, state, safe)

        # Log decision
        self._rl_logger.log_decision(step_label, category, action, safe, state)

        return action

    def rl_log_reward(
        self,
        step_label: str,
        quality_score: float = 0.0,
        goal_coverage: float = 0.0,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        user_accepted: bool | None = None,
    ) -> float | None:
        """Compute and attach reward to a previously logged decision.

        Returns the computed reward, or None if RL not enabled.
        """
        if self._rl_logger is None or self._rl_config is None:
            return None

        from hooks.deso.rl.reward import compute_reward

        reward, components = compute_reward(
            quality_score=quality_score,
            goal_coverage=goal_coverage,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            user_accepted=user_accepted,
            weights=self._rl_config.reward_weights,
        )
        self._rl_logger.attach_reward(step_label, reward, components)
        return reward

    def rl_finalize(self, health_score: int = 0) -> None:
        """Finalize RL episode: chain states, flush to DB (best-effort).

        DB writes happen in a background thread so they never block the
        DESO critical path.
        """
        if self._rl_logger is None:
            return
        try:
            import threading
            episode = self._rl_logger.finalize_episode(health_score)
            conn = self._get_conn()
            if conn:
                # Flush experiences asynchronously (default)
                self._rl_logger.flush(conn, async_write=True)
                # Save episode summary in a separate background thread
                ep_dict = episode.to_dict()

                def _save_episode_bg() -> None:
                    try:
                        from hooks.deso.rl.rl_store import save_episode
                        from hooks.deso.rl.rl_state import EpisodeSummary
                        save_episode(conn, EpisodeSummary.from_dict(ep_dict))
                    except Exception as exc:
                        log.debug("Background episode save failed (non-fatal): %s", exc)

                threading.Thread(
                    target=_save_episode_bg, daemon=True, name="rl-episode"
                ).start()
        except Exception as exc:
            log.debug("rl_finalize failed (non-fatal): %s", exc)

    # ── State accessor ─────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Current completed steps, packets, health."""
        return {
            "completed_steps": sorted(s.value for s in self._completed_steps),
            "packets": list(self._packets.keys()),
            "preflight": self._preflight_result,
            "step_log": self._step_log,
            "health": self.audit()["orchestration_health_score"],
        }

    # ── Internal ───────────────────────────────────────────────────────────

    def _log_step(self, step: Step, message: str) -> None:
        self._step_log.append({
            "step": step.value,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
