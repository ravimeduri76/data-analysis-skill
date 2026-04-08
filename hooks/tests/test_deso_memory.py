"""
test_deso_memory.py — Tests for the DESO three-tier memory system.

TC15: SessionContext dataclass (pure Python, always runs)
TC16: L2 Session Store (requires PostgreSQL + DOMAIN_DB_URL)
TC17: L3 Promotion & Patterns (requires PostgreSQL + DOMAIN_DB_URL)
"""

from __future__ import annotations

import json
import os
import sys
import uuid

import pytest

# Ensure hooks package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hooks.deso.memory import (
    ActionPlan,
    DomainContextPacket,
    ExecSchedule,
    HITLPacket,
    IntentSummary,
    SessionContext,
)


# ── TC15: SessionContext dataclass (pure Python) ────────────────────────────


class TestSessionContextDataclass:
    """TC15 — dataclass serialization round-trips."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        ctx = SessionContext(
            run_id="run_001",
            session_id="s_001",
            app_id="dataaxl",
            solution_id="sol_42",
            user_query="deduplicate customers",
            dataset_path="/data/customers.csv",
            dataset_columns=["name", "email", "phone"],
            created_at="2026-04-06T10:00:00+00:00",
            turn_count=3,
            dcp=DomainContextPacket(
                domain="MDM",
                domain_confidence="high",
                lifecycle_stage="enrichment",
                graph_available=True,
                graph_entities=[{"name": "Customer", "type": "entity"}],
                invariants=[{"rule": "email must be unique"}],
            ),
            intent=IntentSummary(
                explicit_goal="deduplicate customer records",
                latent_goal="improve data quality for CRM",
                requires_hitl=True,
                ambiguities=["merge strategy unclear"],
            ),
            plan=ActionPlan(
                agents=[
                    {"agent_id": "AG27", "role": "matcher"},
                    {"agent_id": "AG28", "role": "golden_record"},
                ],
                hitl_gates=[{"gate": "approve_golden_record"}],
            ),
            schedule=ExecSchedule(
                interaction_mode="hybrid",
                sync_steps=[{"step": "match"}],
                async_jobs=[{"job": "batch_dedup"}],
            ),
            hitl_packets=[
                HITLPacket(
                    question="Which merge strategy?",
                    options=[{"label": "latest wins"}, {"label": "manual"}],
                    response="latest wins",
                    timestamp="2026-04-06T10:05:00+00:00",
                ),
            ],
            worker_outputs={"AG27": {"matched": 150}},
        )

        data = ctx.to_dict()
        restored = SessionContext.from_dict(data)

        assert restored.run_id == "run_001"
        assert restored.app_id == "dataaxl"
        assert restored.solution_id == "sol_42"
        assert restored.turn_count == 3
        assert restored.dcp is not None
        assert restored.dcp.domain == "MDM"
        assert restored.dcp.graph_available is True
        assert len(restored.dcp.graph_entities) == 1
        assert restored.intent is not None
        assert restored.intent.requires_hitl is True
        assert restored.intent.ambiguities == ["merge strategy unclear"]
        assert restored.plan is not None
        assert len(restored.plan.agents) == 2
        assert restored.schedule is not None
        assert restored.schedule.interaction_mode == "hybrid"
        assert len(restored.hitl_packets) == 1
        assert restored.hitl_packets[0].response == "latest wins"
        assert restored.worker_outputs == {"AG27": {"matched": 150}}

    def test_from_dict_handles_partial_data(self) -> None:
        data = {"run_id": "run_002", "session_id": "s_002"}
        ctx = SessionContext.from_dict(data)

        assert ctx.run_id == "run_002"
        assert ctx.app_id == "base"  # default
        assert ctx.solution_id is None
        assert ctx.dcp is None
        assert ctx.intent is None
        assert ctx.plan is None
        assert ctx.schedule is None
        assert ctx.hitl_packets == []
        assert ctx.worker_outputs == {}
        assert ctx.turn_count == 0

    def test_nested_dataclass_serialization(self) -> None:
        """DCP, IS, AP, ES all survive to_dict → from_dict."""
        ctx = SessionContext(
            run_id="run_003",
            session_id="s_003",
            dcp=DomainContextPacket(domain="Healthcare", lifecycle_stage="analytics"),
            intent=IntentSummary(explicit_goal="profile patients"),
            plan=ActionPlan(agents=[{"agent_id": "AG06"}]),
            schedule=ExecSchedule(interaction_mode="sync"),
        )

        data = ctx.to_dict()
        # Verify JSON-safe
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        restored = SessionContext.from_dict(json.loads(json_str))
        assert restored.dcp.domain == "Healthcare"
        assert restored.intent.explicit_goal == "profile patients"
        assert restored.plan.agents[0]["agent_id"] == "AG06"
        assert restored.schedule.interaction_mode == "sync"

    def test_empty_session_context(self) -> None:
        ctx = SessionContext()
        data = ctx.to_dict()
        restored = SessionContext.from_dict(data)
        assert restored.run_id == ""
        assert restored.dcp is None

    def test_from_dict_ignores_extra_keys(self) -> None:
        """Extra keys in nested dicts should not cause errors."""
        data = {
            "run_id": "run_004",
            "session_id": "s_004",
            "dcp": {
                "domain": "Retail",
                "domain_confidence": "medium",
                "lifecycle_stage": "analytics",
                "graph_available": False,
                "unknown_future_field": "should be ignored",
            },
        }
        ctx = SessionContext.from_dict(data)
        assert ctx.dcp.domain == "Retail"


# ── TC16: L2 Session Store (DB-dependent) ──────────────────────────────────


def _get_test_conn():
    """Get a DB connection or skip the test."""
    psycopg2 = pytest.importorskip("psycopg2")
    url = os.environ.get(
        "DOMAIN_DB_URL",
        "postgresql://postgres:memorystorepg@localhost:5432/domain",
    )
    try:
        conn = psycopg2.connect(url)
        return conn
    except psycopg2.OperationalError:
        pytest.skip("Domain DB not available")


@pytest.fixture
def db_conn():
    """Provide a DB connection with session tables initialized."""
    from hooks.deso.memory import init_session_db
    conn = _get_test_conn()
    init_session_db(conn)
    yield conn
    conn.close()


class TestL2SessionStore:
    """TC16 — checkpoint/hydrate against PostgreSQL."""

    def _make_ctx(self, suffix: str = "") -> SessionContext:
        uid = uuid.uuid4().hex[:8]
        return SessionContext(
            run_id=f"test_run_{uid}{suffix}",
            session_id=f"test_sess_{uid}{suffix}",
            app_id="dataaxl",
            user_query="test query",
            dcp=DomainContextPacket(domain="TestDomain", lifecycle_stage="analytics"),
            intent=IntentSummary(explicit_goal="test goal"),
        )

    def test_checkpoint_and_hydrate(self, db_conn) -> None:
        from hooks.deso.memory import checkpoint, hydrate
        ctx = self._make_ctx()

        checkpoint(db_conn, ctx)
        loaded = hydrate(db_conn, ctx.session_id)

        assert loaded is not None
        assert loaded.run_id == ctx.run_id
        assert loaded.app_id == "dataaxl"
        assert loaded.dcp is not None
        assert loaded.dcp.domain == "TestDomain"

        # Cleanup
        cur = db_conn.cursor()
        cur.execute("DELETE FROM session_context WHERE session_id = %s", (ctx.session_id,))
        db_conn.commit()
        cur.close()

    def test_checkpoint_upsert(self, db_conn) -> None:
        from hooks.deso.memory import checkpoint, hydrate
        ctx = self._make_ctx()

        checkpoint(db_conn, ctx)
        ctx.turn_count = 5
        ctx.dcp.domain = "UpdatedDomain"
        checkpoint(db_conn, ctx)

        loaded = hydrate(db_conn, ctx.session_id)
        assert loaded.turn_count == 5
        assert loaded.dcp.domain == "UpdatedDomain"

        # Verify only one row
        cur = db_conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM session_context WHERE session_id = %s",
            (ctx.session_id,),
        )
        assert cur.fetchone()[0] == 1
        cur.execute("DELETE FROM session_context WHERE session_id = %s", (ctx.session_id,))
        db_conn.commit()
        cur.close()

    def test_hydrate_nonexistent_returns_none(self, db_conn) -> None:
        from hooks.deso.memory import hydrate
        result = hydrate(db_conn, "nonexistent_session_id_xyz")
        assert result is None

    def test_cleanup_expired(self, db_conn) -> None:
        from hooks.deso.memory import cleanup_expired
        ctx = self._make_ctx("_expired")

        # Insert with already-expired timestamp
        cur = db_conn.cursor()
        cur.execute(
            """
            INSERT INTO session_context
                (session_id, run_id, app_id, context_data, expires_at)
            VALUES (%s, %s, %s, '{}'::jsonb, NOW() - INTERVAL '1 hour')
            """,
            (ctx.session_id, ctx.run_id, ctx.app_id),
        )
        db_conn.commit()
        cur.close()

        count = cleanup_expired(db_conn)
        assert count >= 1


# ── TC17: L3 Promotion & Patterns (DB-dependent) ──────────────────────────


class TestL3Promotion:
    """TC17 — promote and recall_patterns against PostgreSQL."""

    def _make_full_ctx(self) -> SessionContext:
        uid = uuid.uuid4().hex[:8]
        return SessionContext(
            run_id=f"test_run_{uid}",
            session_id=f"test_sess_{uid}",
            app_id="dataaxl",
            user_query="deduplicate customer records",
            dcp=DomainContextPacket(
                domain="MDM",
                domain_confidence="high",
                lifecycle_stage="enrichment",
                graph_available=True,
            ),
            intent=IntentSummary(
                explicit_goal="deduplicate customers",
                requires_hitl=True,
            ),
            plan=ActionPlan(
                agents=[
                    {"agent_id": "AG27", "role": "matcher"},
                    {"agent_id": "AG28", "role": "golden_record"},
                ],
            ),
            hitl_packets=[
                HITLPacket(
                    question="Which merge strategy?",
                    response="latest wins",
                    resolved_as={"strategy": "latest_wins"},
                ),
            ],
        )

    def test_promote_creates_history(self, db_conn) -> None:
        from hooks.deso.memory import checkpoint, promote
        ctx = self._make_full_ctx()

        # Must checkpoint first (promote deletes L2)
        checkpoint(db_conn, ctx)
        promote(db_conn, ctx, health_score=85)

        cur = db_conn.cursor()
        cur.execute(
            "SELECT domain, health_score, agents_used FROM session_history WHERE run_id = %s",
            (ctx.run_id,),
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "MDM"
        assert row[1] == 85
        assert "AG27" in row[2]
        assert "AG28" in row[2]

        # Cleanup
        cur.execute("DELETE FROM session_history WHERE run_id = %s", (ctx.run_id,))
        cur.execute(
            "DELETE FROM learned_patterns WHERE domain = 'MDM' AND app_id = 'dataaxl'"
        )
        db_conn.commit()
        cur.close()

    def test_promote_extracts_agent_pattern(self, db_conn) -> None:
        from hooks.deso.memory import checkpoint, promote, recall_patterns
        ctx = self._make_full_ctx()

        checkpoint(db_conn, ctx)
        promote(db_conn, ctx, health_score=90)

        patterns = recall_patterns(db_conn, "MDM", "dataaxl", "agent_selection")
        agent_pat = [p for p in patterns if p["pattern_key"] == "enrichment"]
        assert len(agent_pat) >= 1
        assert "AG27" in agent_pat[0]["pattern_value"]["agents"]

        # Cleanup
        cur = db_conn.cursor()
        cur.execute("DELETE FROM session_history WHERE run_id = %s", (ctx.run_id,))
        cur.execute(
            "DELETE FROM learned_patterns WHERE domain = 'MDM' AND app_id = 'dataaxl'"
        )
        db_conn.commit()
        cur.close()

    def test_recall_patterns(self, db_conn) -> None:
        from hooks.deso.memory import recall_patterns

        # Insert a test pattern directly
        cur = db_conn.cursor()
        test_key = f"test_recall_{uuid.uuid4().hex[:8]}"
        cur.execute(
            """
            INSERT INTO learned_patterns
                (pattern_type, domain, app_id, pattern_key, pattern_value)
            VALUES ('agent_selection', 'TestRecall', 'base', %s, '{"agents": ["AG06"]}'::jsonb)
            ON CONFLICT DO NOTHING
            """,
            (test_key,),
        )
        db_conn.commit()
        cur.close()

        patterns = recall_patterns(db_conn, "TestRecall", "base", "agent_selection")
        matched = [p for p in patterns if p["pattern_key"] == test_key]
        assert len(matched) == 1
        assert matched[0]["pattern_value"]["agents"] == ["AG06"]

        # Cleanup
        cur = db_conn.cursor()
        cur.execute(
            "DELETE FROM learned_patterns WHERE domain = 'TestRecall' AND pattern_key = %s",
            (test_key,),
        )
        db_conn.commit()
        cur.close()
