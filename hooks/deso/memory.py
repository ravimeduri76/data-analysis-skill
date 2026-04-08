"""
deso/memory.py — Three-tier memory management for DESO sub-agents.

L1: SessionContext (dataclass)  — in-process, passed by reference to all SAs
L2: session_context (UNLOGGED)  — PostgreSQL, ephemeral, inspectable via SQL
L3: session_history + learned_patterns (permanent tables) — durable run memory

Database: domain (same as intent_graph, execution_graph, domain_graph)
via DOMAIN_DB_URL env var.

Public API
----------
get_session_conn()                                   -> connection
init_session_db(conn)                                -> None
checkpoint(conn, ctx)                                -> None
hydrate(conn, session_id)                            -> SessionContext | None
promote(conn, ctx, health_score)                     -> None
recall_patterns(conn, domain, app_id, pattern_type)  -> list[dict]
cleanup_expired(conn)                                -> int
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)


# ── L1: Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class DomainContextPacket:
    domain: str = ""
    domain_confidence: str = ""
    lifecycle_stage: str = ""
    graph_available: bool = False
    graph_entities: list[dict] = field(default_factory=list)
    terminology: list[dict] = field(default_factory=list)
    invariants: list[dict] = field(default_factory=list)
    ontology_triplets: list[dict] = field(default_factory=list)
    property_model: list[dict] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentSummary:
    explicit_goal: str = ""
    latent_goal: str = ""
    requires_hitl: bool = False
    ambiguities: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    risk_signals: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPlan:
    agents: list[dict] = field(default_factory=list)
    hitl_gates: list[dict] = field(default_factory=list)
    blocking_risks: list[dict] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecSchedule:
    interaction_mode: str = "sync"
    sync_steps: list[dict] = field(default_factory=list)
    async_jobs: list[dict] = field(default_factory=list)
    hitl_checkpoints: list[dict] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class HITLPacket:
    question: str = ""
    options: list[dict] = field(default_factory=list)
    response: str = ""
    resolved_as: dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class SessionContext:
    """L1: Shared mutable state for all sub-agents during a single DESO execution."""

    run_id: str = ""
    session_id: str = ""
    app_id: str = "base"
    solution_id: str | None = None
    user_query: str = ""
    dataset_path: str | None = None
    dataset_columns: list[str] = field(default_factory=list)
    created_at: str = ""
    turn_count: int = 0

    dcp: DomainContextPacket | None = None
    intent: IntentSummary | None = None
    plan: ActionPlan | None = None
    schedule: ExecSchedule | None = None
    hitl_packets: list[HITLPacket] = field(default_factory=list)
    worker_outputs: dict[str, Any] = field(default_factory=dict)

    # Adaptive planning state (populated when adaptive mode is enabled)
    adaptive_state: dict[str, Any] | None = None
    adaptation_events: list[dict] = field(default_factory=list)
    goal_progress: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (handles dataclass nesting)."""
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    @classmethod
    def from_dict(cls, data: dict) -> SessionContext:
        """Hydrate from dict (e.g. from L2 JSONB)."""
        ctx = cls(
            run_id=data.get("run_id", ""),
            session_id=data.get("session_id", ""),
            app_id=data.get("app_id", "base"),
            solution_id=data.get("solution_id"),
            user_query=data.get("user_query", ""),
            dataset_path=data.get("dataset_path"),
            dataset_columns=data.get("dataset_columns", []),
            created_at=data.get("created_at", ""),
            turn_count=data.get("turn_count", 0),
            worker_outputs=data.get("worker_outputs", {}),
            adaptive_state=data.get("adaptive_state"),
            adaptation_events=data.get("adaptation_events", []),
            goal_progress=data.get("goal_progress"),
        )
        if data.get("dcp"):
            ctx.dcp = DomainContextPacket(
                **{k: v for k, v in data["dcp"].items()
                   if k in DomainContextPacket.__dataclass_fields__}
            )
        if data.get("intent"):
            ctx.intent = IntentSummary(
                **{k: v for k, v in data["intent"].items()
                   if k in IntentSummary.__dataclass_fields__}
            )
        if data.get("plan"):
            ctx.plan = ActionPlan(
                **{k: v for k, v in data["plan"].items()
                   if k in ActionPlan.__dataclass_fields__}
            )
        if data.get("schedule"):
            ctx.schedule = ExecSchedule(
                **{k: v for k, v in data["schedule"].items()
                   if k in ExecSchedule.__dataclass_fields__}
            )
        if data.get("hitl_packets"):
            ctx.hitl_packets = [
                HITLPacket(**{k: v for k, v in h.items()
                             if k in HITLPacket.__dataclass_fields__})
                for h in data["hitl_packets"]
            ]
        return ctx


# ── Connection ───────────────────────────────────────────────────────────────


def _load_env() -> None:
    """Load DOMAIN_DB_URL from Data_Analysis/.env or env_vault.py."""
    try:
        import importlib.util
        _vault_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "shared", "env_vault.py",
        )
        _spec = importlib.util.spec_from_file_location("env_vault", _vault_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        if _mod.load_vault():
            return
    except Exception:
        pass
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Data_Analysis", ".env",
    )
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_DEFAULT_URL = "postgresql://postgres:memorystorepg@localhost:5432/domain"


def get_session_conn():
    """Return a psycopg2 connection to the domain DB for session management.

    Raises psycopg2.OperationalError on connection failure.
    """
    _load_env()
    url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
    import psycopg2
    return psycopg2.connect(url)


# ── L2: Session Store (UNLOGGED) ────────────────────────────────────────────


_SESSION_DDL = """
CREATE UNLOGGED TABLE IF NOT EXISTS session_context (
    session_id      TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    app_id          TEXT DEFAULT 'base',
    solution_id     TEXT,
    user_query      TEXT,
    resolved_domain TEXT,
    resolved_entities JSONB DEFAULT '[]',
    context_data    JSONB NOT NULL DEFAULT '{}',
    hitl_history    JSONB DEFAULT '[]',
    turn_count      INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    expires_at      TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '24 hours')
);
CREATE INDEX IF NOT EXISTS idx_session_expiry ON session_context (expires_at);
CREATE INDEX IF NOT EXISTS idx_session_run ON session_context (run_id);
"""


# ── L3: Permanent Tables ────────────────────────────────────────────────────


_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS session_history (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT UNIQUE NOT NULL,
    session_id      TEXT NOT NULL,
    app_id          TEXT,
    solution_id     TEXT,
    user_query      TEXT,
    domain          TEXT,
    lifecycle_stage TEXT,
    agents_used     TEXT[],
    health_score    INT,
    hitl_count      INT DEFAULT 0,
    context_data    JSONB,
    completed_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_history_domain ON session_history (domain);
CREATE INDEX IF NOT EXISTS idx_history_app ON session_history (app_id);

CREATE TABLE IF NOT EXISTS learned_patterns (
    id              SERIAL PRIMARY KEY,
    pattern_type    TEXT NOT NULL,
    domain          TEXT,
    app_id          TEXT,
    pattern_key     TEXT NOT NULL,
    pattern_value   JSONB NOT NULL,
    frequency       INT DEFAULT 1,
    last_seen       TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (pattern_type, domain, app_id, pattern_key)
);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON learned_patterns (pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_domain ON learned_patterns (domain);
"""


def init_session_db(conn) -> None:
    """Create L2 + L3 tables idempotently."""
    cur = conn.cursor()
    cur.execute(_SESSION_DDL)
    cur.execute(_HISTORY_DDL)
    conn.commit()
    cur.close()


# ── L2 Lifecycle Functions ───────────────────────────────────────────────────


def checkpoint(conn, ctx: SessionContext) -> None:
    """Upsert L1 SessionContext → L2 session_context table."""
    data = ctx.to_dict()
    domain = (ctx.dcp.domain if ctx.dcp else "") or ""
    entities = json.dumps(
        ctx.dcp.graph_entities if ctx.dcp else [], default=str
    )
    hitl_hist = json.dumps(
        [dataclasses.asdict(h) for h in ctx.hitl_packets], default=str
    )
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO session_context
            (session_id, run_id, app_id, solution_id, user_query,
             resolved_domain, resolved_entities, context_data,
             hitl_history, turn_count, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, NOW())
        ON CONFLICT (session_id) DO UPDATE SET
            context_data    = EXCLUDED.context_data,
            resolved_domain = EXCLUDED.resolved_domain,
            resolved_entities = EXCLUDED.resolved_entities,
            hitl_history    = EXCLUDED.hitl_history,
            turn_count      = EXCLUDED.turn_count,
            updated_at      = NOW()
        """,
        (
            ctx.session_id, ctx.run_id, ctx.app_id, ctx.solution_id,
            ctx.user_query, domain, entities,
            json.dumps(data, default=str), hitl_hist, ctx.turn_count,
        ),
    )
    conn.commit()
    cur.close()


def hydrate(conn, session_id: str) -> SessionContext | None:
    """Load L2 session_context → L1 SessionContext. Returns None if not found."""
    cur = conn.cursor()
    cur.execute(
        "SELECT context_data FROM session_context WHERE session_id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    cur.close()
    if row is None:
        return None
    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    return SessionContext.from_dict(data)


# ── L3 Lifecycle Functions ───────────────────────────────────────────────────


def promote(conn, ctx: SessionContext, health_score: int) -> None:
    """Move completed session L2 → L3 (session_history + learned_patterns)."""
    domain = (ctx.dcp.domain if ctx.dcp else "") or ""
    lifecycle = (ctx.dcp.lifecycle_stage if ctx.dcp else "") or ""
    agents_used = [
        a.get("agent_id", "") for a in (ctx.plan.agents if ctx.plan else [])
    ]
    data = ctx.to_dict()

    cur = conn.cursor()

    # Insert into session_history
    cur.execute(
        """
        INSERT INTO session_history
            (run_id, session_id, app_id, solution_id, user_query,
             domain, lifecycle_stage, agents_used, health_score,
             hitl_count, context_data)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (run_id) DO NOTHING
        """,
        (
            ctx.run_id, ctx.session_id, ctx.app_id, ctx.solution_id,
            ctx.user_query, domain, lifecycle, agents_used,
            health_score, len(ctx.hitl_packets),
            json.dumps(data, default=str),
        ),
    )

    # Extract and upsert learned patterns
    _extract_patterns(cur, ctx, domain)

    # Clean up L2 entry
    cur.execute(
        "DELETE FROM session_context WHERE session_id = %s",
        (ctx.session_id,),
    )

    conn.commit()
    cur.close()


def _extract_patterns(cur, ctx: SessionContext, domain: str) -> None:
    """Extract repeatable patterns from a completed session into learned_patterns."""
    # Pattern 1: agent_selection — which agents for this domain+lifecycle
    if ctx.plan and ctx.plan.agents and ctx.dcp:
        lifecycle = ctx.dcp.lifecycle_stage or "unknown"
        agent_ids = sorted(
            a.get("agent_id", "") for a in ctx.plan.agents if a.get("agent_id")
        )
        if agent_ids:
            _upsert_pattern(
                cur,
                pattern_type="agent_selection",
                domain=domain,
                app_id=ctx.app_id,
                pattern_key=lifecycle,
                pattern_value={"agents": agent_ids, "lifecycle": lifecycle},
            )

    # Pattern 2: hitl_resolution — how ambiguities were resolved
    for hp in ctx.hitl_packets:
        if hp.question and hp.response:
            _upsert_pattern(
                cur,
                pattern_type="hitl_resolution",
                domain=domain,
                app_id=ctx.app_id,
                pattern_key=hp.question[:200],
                pattern_value={
                    "question": hp.question,
                    "response": hp.response,
                    "resolved_as": hp.resolved_as,
                },
            )

    # Pattern 3: app_resolution — query patterns that map to app_ids
    if ctx.app_id != "base" and ctx.user_query:
        _upsert_pattern(
            cur,
            pattern_type="app_resolution",
            domain=domain,
            app_id=ctx.app_id,
            pattern_key=ctx.user_query[:200],
            pattern_value={"app_id": ctx.app_id, "solution_id": ctx.solution_id},
        )

    # Pattern 4: quality_baseline — average quality scores per agent
    if ctx.adaptive_state and ctx.adaptive_state.get("step_states"):
        lifecycle = (ctx.dcp.lifecycle_stage if ctx.dcp else "") or "unknown"
        for _so, ss in ctx.adaptive_state["step_states"].items():
            agent_id = ss.get("agent_id", "")
            quality = ss.get("quality")
            if agent_id and quality and isinstance(quality, dict):
                _upsert_pattern(
                    cur,
                    pattern_type="quality_baseline",
                    domain=domain,
                    app_id=ctx.app_id,
                    pattern_key=f"{lifecycle}/{agent_id}",
                    pattern_value={
                        "avg_quality": quality.get("overall", 0.0),
                        "sample_count": 1,
                    },
                )

    # Pattern 5: adaptation_outcome — how adaptations resolved per type
    if ctx.adaptation_events:
        lifecycle = (ctx.dcp.lifecycle_stage if ctx.dcp else "") or "unknown"
        by_type: dict[str, list[str]] = {}
        for evt in ctx.adaptation_events:
            etype = evt.get("event_type", "UNKNOWN")
            agent_id = evt.get("agent_id", "")
            by_type.setdefault(etype, []).append(agent_id)
        for etype, agents in by_type.items():
            _upsert_pattern(
                cur,
                pattern_type="adaptation_outcome",
                domain=domain,
                app_id=ctx.app_id,
                pattern_key=f"{lifecycle}/{etype}",
                pattern_value={"count": len(agents), "agents": agents},
            )

    # Pattern 6: rl_decision_outcome — RL action/reward pairs per step
    rl_decisions = (
        getattr(ctx, "rl_decisions", None)
        or ctx.worker_outputs.get("rl_decisions", [])
    )
    if rl_decisions:
        for decision in rl_decisions:
            step_label = decision.get("step_label", "")
            action_taken = decision.get("action_taken", "")
            if step_label and action_taken:
                _upsert_pattern(
                    cur,
                    pattern_type="rl_decision_outcome",
                    domain=domain,
                    app_id=ctx.app_id,
                    pattern_key=f"{step_label}/{action_taken}",
                    pattern_value={
                        "category": decision.get("category", ""),
                        "action": action_taken,
                        "reward": decision.get("reward"),
                        "quality": decision.get("quality", 0.0),
                    },
                )


def _upsert_pattern(
    cur,
    pattern_type: str,
    domain: str,
    app_id: str,
    pattern_key: str,
    pattern_value: dict,
) -> None:
    """Insert or increment a learned pattern."""
    cur.execute(
        """
        INSERT INTO learned_patterns
            (pattern_type, domain, app_id, pattern_key, pattern_value)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (pattern_type, domain, app_id, pattern_key)
        DO UPDATE SET
            frequency   = learned_patterns.frequency + 1,
            last_seen   = NOW(),
            pattern_value = EXCLUDED.pattern_value
        """,
        (pattern_type, domain, app_id, pattern_key,
         json.dumps(pattern_value, default=str)),
    )


def recall_patterns(
    conn,
    domain: str,
    app_id: str = "base",
    pattern_type: str | None = None,
) -> list[dict]:
    """Query L3 learned patterns by domain and optionally pattern_type."""
    cur = conn.cursor()
    if pattern_type:
        cur.execute(
            """
            SELECT pattern_type, pattern_key, pattern_value, frequency, last_seen
            FROM learned_patterns
            WHERE domain = %s AND app_id = %s AND pattern_type = %s
            ORDER BY frequency DESC, last_seen DESC
            """,
            (domain, app_id, pattern_type),
        )
    else:
        cur.execute(
            """
            SELECT pattern_type, pattern_key, pattern_value, frequency, last_seen
            FROM learned_patterns
            WHERE domain = %s AND app_id = %s
            ORDER BY frequency DESC, last_seen DESC
            """,
            (domain, app_id),
        )
    rows = cur.fetchall()
    cur.close()
    return [
        {
            "pattern_type": r[0],
            "pattern_key": r[1],
            "pattern_value": r[2] if isinstance(r[2], dict) else json.loads(r[2]),
            "frequency": r[3],
            "last_seen": str(r[4]),
        }
        for r in rows
    ]


def cleanup_expired(conn) -> int:
    """Delete expired L2 sessions. Returns count deleted."""
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM session_context WHERE expires_at < NOW()"
    )
    count = cur.rowcount
    conn.commit()
    cur.close()
    return count
