"""Tests for DESO RL feedback loop (Phase 1: Thompson Sampling).

25+ tests covering reward, policy, safety, serialization, logger, integration.
All pure Python — no DB dependency (mock conn where needed).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hooks.deso.rl.rl_state import (
    ACTIONS_BY_CATEGORY,
    ActionCategory,
    ActionType,
    EpisodeSummary,
    Experience,
    PolicyVersion,
    RLConfig,
    StateSnapshot,
)
from hooks.deso.rl.reward import compute_reward, compute_step_reward, default_weights
from hooks.deso.rl.policy import ThompsonSamplingPolicy, filter_safe_actions
from hooks.deso.rl.logger import RLLogger


# ══════════════════════════════════════════════════════════════════════════════
# TestReward — 5 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestReward:
    def test_basic_computation(self):
        """Perfect inputs should yield a high reward."""
        reward, components = compute_reward(
            quality_score=1.0,
            goal_coverage=1.0,
            latency_ms=0,
            cost_usd=0.0,
            user_accepted=True,
        )
        assert reward > 0.9
        assert components["quality_outcome"] == 1.0
        assert components["goal_coverage"] == 1.0
        assert components["efficiency"] == 1.0
        assert components["user_acceptance"] == 1.0

    def test_custom_weights(self):
        """Custom weights should shift the reward."""
        weights = {"quality_outcome": 1.0, "goal_coverage": 0.0,
                   "efficiency": 0.0, "user_acceptance": 0.0}
        reward, _ = compute_reward(
            quality_score=0.5, goal_coverage=1.0,
            latency_ms=120_000, cost_usd=1.0,
            weights=weights,
        )
        assert abs(reward - 0.5) < 0.01

    def test_none_feedback(self):
        """None user feedback should contribute 0."""
        reward, components = compute_reward(
            quality_score=0.5, goal_coverage=0.5,
            latency_ms=60_000, cost_usd=0.25,
            user_accepted=None,
        )
        assert components["user_acceptance"] == 0.0
        assert reward > 0

    def test_zero_latency_and_cost(self):
        """Zero latency/cost should give max efficiency."""
        _, components = compute_reward(
            quality_score=0.5, goal_coverage=0.5,
            latency_ms=0, cost_usd=0.0,
        )
        assert components["efficiency"] == 1.0

    def test_rejection_penalty(self):
        """User rejection should reduce reward."""
        reward_accept, _ = compute_reward(
            quality_score=0.5, goal_coverage=0.5,
            latency_ms=0, cost_usd=0.0, user_accepted=True,
        )
        reward_reject, _ = compute_reward(
            quality_score=0.5, goal_coverage=0.5,
            latency_ms=0, cost_usd=0.0, user_accepted=False,
        )
        assert reward_accept > reward_reject

    def test_step_reward(self):
        """Step reward should work without goal coverage."""
        reward, components = compute_step_reward(
            quality_score=0.8, latency_ms=5000, cost_usd=0.01,
        )
        assert reward > 0
        assert components["goal_coverage"] == 0.0

    def test_reward_clamped(self):
        """Reward should be clamped to [-1, 1]."""
        reward, _ = compute_reward(
            quality_score=1.0, goal_coverage=1.0,
            latency_ms=0, cost_usd=0.0, user_accepted=True,
        )
        assert -1.0 <= reward <= 1.0

        reward_bad, _ = compute_reward(
            quality_score=0.0, goal_coverage=0.0,
            latency_ms=240_000, cost_usd=2.0, user_accepted=False,
        )
        assert -1.0 <= reward_bad <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TestPolicy — 6 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPolicy:
    def test_thompson_selection_returns_valid_action(self):
        """Selected action should be in the category's action set."""
        policy = ThompsonSamplingPolicy()
        cat = ActionCategory.AGENT_SELECTION
        action = policy.select_action(cat)
        assert action in ACTIONS_BY_CATEGORY[cat]

    def test_update_shifts_beta(self):
        """High reward should increase alpha; low reward should increase beta."""
        policy = ThompsonSamplingPolicy()
        cat = ActionCategory.AGENT_SELECTION
        act = ActionType.SELECT_PROFILER
        key = (cat.value, act.value)

        alpha_before, beta_before = policy._params[key]
        policy.update(cat, act, 0.8)  # success
        alpha_after, _ = policy._params[key]
        assert alpha_after == alpha_before + 1

        _, beta_before2 = policy._params[key]
        policy.update(cat, act, 0.3)  # failure
        _, beta_after2 = policy._params[key]
        assert beta_after2 == beta_before2 + 1

    def test_epsilon_fallback_with_low_experience(self):
        """With no experience, policy should still return a valid action."""
        config = RLConfig(min_experiences=100)
        policy = ThompsonSamplingPolicy(config=config)
        action = policy.select_action(ActionCategory.LLM_ROUTING)
        assert action in ACTIONS_BY_CATEGORY[ActionCategory.LLM_ROUTING]

    def test_deterministic_with_strong_signal(self):
        """After many updates, should converge toward best action."""
        config = RLConfig(min_experiences=5)
        policy = ThompsonSamplingPolicy(config=config)
        cat = ActionCategory.AGENT_SELECTION
        best = ActionType.SELECT_PROFILER

        # Overwhelm one action with high rewards
        for _ in range(50):
            policy.update(cat, best, 0.9)
        for act in ACTIONS_BY_CATEGORY[cat]:
            if act != best:
                for _ in range(50):
                    policy.update(cat, act, 0.1)

        # Sample many times — best should dominate
        counts = {a: 0 for a in ACTIONS_BY_CATEGORY[cat]}
        for _ in range(200):
            a = policy.select_action(cat)
            counts[a] += 1
        assert counts[best] > 100  # should be heavily favored

    def test_save_load_roundtrip(self):
        """Policy params should survive serialization."""
        policy = ThompsonSamplingPolicy()
        policy.update(ActionCategory.AGENT_SELECTION, ActionType.SELECT_PROFILER, 0.9)
        policy.update(ActionCategory.LLM_ROUTING, ActionType.USE_GROQ, 0.2)

        params = policy.get_params()
        new_policy = ThompsonSamplingPolicy()
        new_policy.load_params(params)

        assert new_policy._params == policy._params
        assert new_policy._counts == policy._counts

    def test_safe_actions_filter(self):
        """select_action should respect safe_actions filter."""
        policy = ThompsonSamplingPolicy()
        safe = [ActionType.SELECT_DQ, ActionType.SELECT_ENRICHER]
        action = policy.select_action(ActionCategory.AGENT_SELECTION, safe_actions=safe)
        assert action in safe


# ══════════════════════════════════════════════════════════════════════════════
# TestSafetyFiltering — 4 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSafetyFiltering:
    def test_hitl_always_allowed(self):
        """ESCALATE_HITL should always be in safe set for execution control."""
        safe = filter_safe_actions(ActionCategory.EXECUTION_CONTROL)
        assert ActionType.ESCALATE_HITL in safe

    def test_hitl_agent_blocked_without_approval(self):
        """Agent requiring HITL should be blocked if not pre-approved."""
        # AG11 (SELECT_DQ) is in _HITL_AGENTS
        safe = filter_safe_actions(
            ActionCategory.AGENT_SELECTION,
            permission_result={"approved_agents": []},
        )
        assert ActionType.SELECT_DQ not in safe

    def test_hitl_agent_allowed_with_approval(self):
        """Agent requiring HITL should be allowed if pre-approved."""
        safe = filter_safe_actions(
            ActionCategory.AGENT_SELECTION,
            permission_result={"approved_agents": ["AG11"]},
        )
        assert ActionType.SELECT_DQ in safe

    def test_finalize_blocked_below_quality(self):
        """FINALIZE should be blocked when quality is below threshold."""
        safe = filter_safe_actions(
            ActionCategory.EXECUTION_CONTROL,
            permission_result={"avg_quality": 0.3},
            config=RLConfig(quality_threshold=0.6),
        )
        assert ActionType.FINALIZE not in safe
        # But ESCALATE_HITL must still be there
        assert ActionType.ESCALATE_HITL in safe

    def test_empty_safe_set_fallback(self):
        """Should always return at least one action even with max constraints."""
        safe = filter_safe_actions(ActionCategory.RETRIEVAL_STRATEGY)
        assert len(safe) >= 1

    @patch.dict("os.environ", {}, clear=True)
    def test_claude_blocked_without_key(self):
        """USE_CLAUDE should be blocked if ANTHROPIC_API_KEY not set."""
        safe = filter_safe_actions(ActionCategory.LLM_ROUTING)
        assert ActionType.USE_CLAUDE not in safe


# ══════════════════════════════════════════════════════════════════════════════
# TestStateSnapshot — 4 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestStateSnapshot:
    def test_to_dict_from_dict_roundtrip(self):
        state = StateSnapshot(
            run_id="test-1", step_label="SA3", domain="Healthcare",
            lifecycle_stage="preparation", dataset_rows=1000, dataset_cols=20,
            avg_null_pct=0.05, agents_called=["AG06", "AG09"],
            llm_used="claude", steps_completed=3,
            prior_quality_scores=[0.8, 0.7],
            latency_ms=5000, cost_usd=0.10,
        )
        d = state.to_dict()
        restored = StateSnapshot.from_dict(d)
        assert restored.run_id == "test-1"
        assert restored.domain == "Healthcare"
        assert restored.agents_called == ["AG06", "AG09"]
        assert restored.prior_quality_scores == [0.8, 0.7]

    def test_feature_vector_dimensions(self):
        state = StateSnapshot(dataset_rows=100, dataset_cols=10,
                              prior_quality_scores=[0.5, 0.7])
        vec = state.to_feature_vector()
        assert len(vec) == 8  # 7 numeric + 1 mean quality
        assert vec[7] == pytest.approx(0.6)  # mean of [0.5, 0.7]

    def test_feature_vector_no_quality(self):
        state = StateSnapshot()
        vec = state.to_feature_vector()
        assert len(vec) == 8
        assert vec[7] == 0.0  # no prior scores

    def test_from_dict_none(self):
        state = StateSnapshot.from_dict(None)
        assert state.run_id == ""


# ══════════════════════════════════════════════════════════════════════════════
# TestExperience — 2 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExperience:
    def test_serialization_roundtrip(self):
        state = StateSnapshot(run_id="r1", domain="MDM")
        exp = Experience(
            run_id="r1", step_label="SA3",
            state=state, action_category="agent_selection",
            action_taken="select_profiler", alternatives=["select_dq"],
            reward=0.75, reward_components={"quality_outcome": 0.8},
            done=True,
        )
        d = exp.to_dict()
        restored = Experience.from_dict(d)
        assert restored.run_id == "r1"
        assert restored.reward == 0.75
        assert restored.state.domain == "MDM"
        assert restored.done is True

    def test_policy_version_roundtrip(self):
        pv = PolicyVersion(
            version=3, experience_count=100, avg_reward=0.65,
            parameters={"beta_params": {"a|b": {"alpha": 5, "beta": 2}}},
            method="thompson_sampling", is_active=True,
        )
        d = pv.to_dict()
        restored = PolicyVersion.from_dict(d)
        assert restored.version == 3
        assert restored.is_active is True
        assert "beta_params" in restored.parameters


# ══════════════════════════════════════════════════════════════════════════════
# TestLogger — 3 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLogger:
    def test_log_attach_finalize_chains_states(self):
        """Finalize should chain next_state pointers and mark last done."""
        logger = RLLogger(run_id="run-1", policy_version=1)
        s1 = StateSnapshot(step_label="SA3", domain="Healthcare")
        s2 = StateSnapshot(step_label="SA4", domain="Healthcare")

        logger.log_decision("SA3", ActionCategory.AGENT_SELECTION,
                            ActionType.SELECT_PROFILER, [ActionType.SELECT_DQ], s1)
        logger.attach_reward("SA3", 0.8, {"quality_outcome": 0.8})

        logger.log_decision("SA4", ActionCategory.EXECUTION_CONTROL,
                            ActionType.CONTINUE, [ActionType.FINALIZE], s2)
        logger.attach_reward("SA4", 0.6, {"quality_outcome": 0.6})

        episode = logger.finalize_episode(health_score=85)

        # Check chaining
        exp_sa3 = logger._experiences["SA3"]
        assert exp_sa3.next_state is not None
        assert exp_sa3.next_state.step_label == "SA4"
        assert exp_sa3.done is False

        exp_sa4 = logger._experiences["SA4"]
        assert exp_sa4.done is True

        # Check episode summary
        assert episode.run_id == "run-1"
        assert episode.num_decisions == 2
        assert episode.total_reward == pytest.approx(1.4)
        assert episode.health_score == 85

    def test_flush_with_mock_conn(self):
        """Flush (sync mode) should call save_experience_batch."""
        logger = RLLogger(run_id="run-2", policy_version=0)
        s = StateSnapshot(step_label="SA3")
        logger.log_decision("SA3", ActionCategory.AGENT_SELECTION,
                            ActionType.SELECT_PROFILER, [], s)
        logger.attach_reward("SA3", 0.5, {})
        logger.finalize_episode()

        with patch("hooks.deso.rl.rl_store.save_experience_batch") as mock_save:
            mock_conn = MagicMock()
            count = logger.flush(mock_conn, async_write=False)
            assert count == 1
            mock_save.assert_called_once()

    def test_empty_logger_finalize(self):
        """Empty logger should finalize gracefully."""
        logger = RLLogger(run_id="empty")
        episode = logger.finalize_episode(health_score=50)
        assert episode.num_decisions == 0
        assert episode.total_reward == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TestRLConfig — 2 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRLConfig:
    def test_defaults(self):
        config = RLConfig()
        assert config.enabled is True
        assert config.exploration_rate == 0.3
        assert config.min_experiences == 20

    def test_roundtrip(self):
        config = RLConfig(exploration_rate=0.5, min_experiences=50)
        d = config.to_dict()
        restored = RLConfig.from_dict(d)
        assert restored.exploration_rate == 0.5
        assert restored.min_experiences == 50


# ══════════════════════════════════════════════════════════════════════════════
# TestIntegration — 3 tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_full_flow_state_action_reward_finalize(self):
        """End-to-end: build state, select action, compute reward, finalize."""
        config = RLConfig(min_experiences=0)
        policy = ThompsonSamplingPolicy(config=config)
        logger = RLLogger(run_id="integration-1", policy_version=0)

        state = StateSnapshot(
            run_id="integration-1", step_label="SA3",
            domain="Healthcare", steps_completed=2,
        )

        # Select action
        safe = filter_safe_actions(ActionCategory.AGENT_SELECTION)
        action = policy.select_action(ActionCategory.AGENT_SELECTION, state, safe)
        assert action in ACTIONS_BY_CATEGORY[ActionCategory.AGENT_SELECTION]

        # Log decision
        logger.log_decision("SA3", ActionCategory.AGENT_SELECTION, action, safe, state)

        # Compute and attach reward
        reward, components = compute_reward(
            quality_score=0.8, goal_coverage=0.7,
            latency_ms=3000, cost_usd=0.05,
            user_accepted=True,
        )
        logger.attach_reward("SA3", reward, components)

        # Update policy
        policy.update(ActionCategory.AGENT_SELECTION, action, reward)

        # Finalize
        episode = logger.finalize_episode(health_score=80)
        assert episode.num_decisions == 1
        assert episode.total_reward > 0

    def test_serialization_full_roundtrip(self):
        """All dataclasses should survive JSON serialization."""
        state = StateSnapshot(run_id="ser-1", domain="MDM", steps_completed=5)
        exp = Experience(
            run_id="ser-1", step_label="SA3", state=state,
            action_category="agent_selection", action_taken="select_profiler",
            reward=0.7, reward_components={"quality_outcome": 0.8},
        )

        # Serialize to JSON and back
        d = exp.to_dict()
        json_str = json.dumps(d, default=str)
        restored_dict = json.loads(json_str)
        restored = Experience.from_dict(restored_dict)

        assert restored.run_id == "ser-1"
        assert restored.state.domain == "MDM"
        assert restored.reward == 0.7

    def test_backward_compat_rl_disabled(self):
        """DESORunner should work normally when RL is not enabled."""
        from hooks.deso_runner import DESORunner

        runner = DESORunner("test query")
        # RL methods should be no-ops
        assert runner.rl_select_action("SA3", ActionCategory.AGENT_SELECTION) is None
        assert runner.rl_log_reward("SA3", quality_score=0.8) is None
        runner.rl_finalize(80)  # should not raise
