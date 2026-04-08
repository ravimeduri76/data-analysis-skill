"""PostgreSQL persistence for DESO RL feedback loop.

4 tables in the existing `domain` database (DOMAIN_DB_URL):
- rl_experience_buffer: SARS tuples
- rl_episodes: per-run summaries
- rl_policy_versions: policy checkpoints
- rl_reward_config: configurable reward weights

All writes are best-effort — callers should wrap in try/except.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from hooks.deso.rl.rl_state import (
    EpisodeSummary,
    Experience,
    PolicyVersion,
    StateSnapshot,
)

log = logging.getLogger(__name__)


# ── Connection ───────────────────────────────────────────────────────────────


def get_rl_conn() -> Any:
    """Get a PostgreSQL connection for RL tables (same DB as domain_graph)."""
    try:
        import psycopg2

        url = os.environ.get("DOMAIN_DB_URL", "")
        if not url:
            # Try loading from .env
            env_path = os.path.expanduser("~/Data_Analysis/.env")
            if os.path.isfile(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("DOMAIN_DB_URL="):
                            url = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
        if not url:
            url = "postgresql://postgres:postgres@localhost:5432/domain"
        return psycopg2.connect(url)
    except Exception as exc:
        log.debug("Could not connect to RL DB: %s", exc)
        return None


# ── DDL ──────────────────────────────────────────────────────────────────────


_DDL = """
CREATE TABLE IF NOT EXISTS rl_experience_buffer (
    id              TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    step_label      TEXT NOT NULL,
    action_category TEXT NOT NULL,
    action_taken    TEXT NOT NULL,
    alternatives    TEXT[],
    reward          FLOAT,
    reward_components JSONB DEFAULT '{}',
    state_structured  JSONB,
    state_embedding   float4[],
    next_state_structured JSONB,
    next_state_embedding  float4[],
    done            BOOLEAN DEFAULT FALSE,
    outcome_metadata JSONB DEFAULT '{}',
    policy_version  INT DEFAULT 0,
    app_id          TEXT DEFAULT 'base',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rl_exp_run ON rl_experience_buffer (run_id);
CREATE INDEX IF NOT EXISTS idx_rl_exp_cat ON rl_experience_buffer (action_category);
CREATE INDEX IF NOT EXISTS idx_rl_exp_created ON rl_experience_buffer (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rl_exp_app ON rl_experience_buffer (app_id);

CREATE TABLE IF NOT EXISTS rl_episodes (
    run_id          TEXT PRIMARY KEY,
    total_reward    FLOAT DEFAULT 0.0,
    num_decisions   INT DEFAULT 0,
    actions_taken   TEXT[],
    avg_quality     FLOAT DEFAULT 0.0,
    total_latency_ms INT DEFAULT 0,
    total_cost_usd  FLOAT DEFAULT 0.0,
    health_score    INT DEFAULT 0,
    policy_version  INT DEFAULT 0,
    app_id          TEXT DEFAULT 'base',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rl_policy_versions (
    version         INT PRIMARY KEY,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    experience_count INT DEFAULT 0,
    avg_reward      FLOAT DEFAULT 0.0,
    parameters      JSONB DEFAULT '{}',
    method          TEXT DEFAULT 'thompson_sampling',
    is_active       BOOLEAN DEFAULT FALSE,
    app_id          TEXT DEFAULT 'base'
);

CREATE TABLE IF NOT EXISTS rl_reward_config (
    id              SERIAL PRIMARY KEY,
    config_name     TEXT UNIQUE NOT NULL,
    weights         JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


_MIGRATE_APP_ID = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'rl_experience_buffer' AND column_name = 'app_id'
    ) THEN
        ALTER TABLE rl_experience_buffer ADD COLUMN app_id TEXT DEFAULT 'base';
        CREATE INDEX IF NOT EXISTS idx_rl_exp_app ON rl_experience_buffer (app_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'rl_episodes' AND column_name = 'app_id'
    ) THEN
        ALTER TABLE rl_episodes ADD COLUMN app_id TEXT DEFAULT 'base';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'rl_policy_versions' AND column_name = 'app_id'
    ) THEN
        ALTER TABLE rl_policy_versions ADD COLUMN app_id TEXT DEFAULT 'base';
    END IF;
END $$;
"""


def init_rl_db(conn: Any) -> None:
    """Create RL tables if they don't exist. Idempotent."""
    cur = conn.cursor()
    cur.execute(_DDL)
    cur.execute(_MIGRATE_APP_ID)
    conn.commit()
    cur.close()


# ── Experience CRUD ──────────────────────────────────────────────────────────


def save_experience(conn: Any, exp: Experience) -> None:
    """Insert a single experience into rl_experience_buffer."""
    cur = conn.cursor()
    state_dict = exp.state.to_dict() if exp.state else None
    next_state_dict = exp.next_state.to_dict() if exp.next_state else None
    state_emb = exp.state.domain_embedding if exp.state and exp.state.domain_embedding else None
    next_state_emb = exp.next_state.domain_embedding if exp.next_state and exp.next_state.domain_embedding else None

    cur.execute(
        """
        INSERT INTO rl_experience_buffer
            (id, run_id, step_label, action_category, action_taken,
             alternatives, reward, reward_components,
             state_structured, state_embedding,
             next_state_structured, next_state_embedding,
             done, outcome_metadata, policy_version, app_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                %s::jsonb, %s, %s::jsonb, %s,
                %s, %s::jsonb, %s, %s)
        ON CONFLICT (id) DO NOTHING
        """,
        (
            exp.id, exp.run_id, exp.step_label,
            exp.action_category, exp.action_taken,
            exp.alternatives, exp.reward,
            json.dumps(exp.reward_components, default=str),
            json.dumps(state_dict, default=str) if state_dict else None,
            state_emb,
            json.dumps(next_state_dict, default=str) if next_state_dict else None,
            next_state_emb,
            exp.done,
            json.dumps(exp.outcome_metadata, default=str),
            exp.policy_version,
            exp.app_id,
        ),
    )
    conn.commit()
    cur.close()


def save_experience_batch(conn: Any, exps: list[Experience]) -> None:
    """Batch-insert experiences in a single transaction."""
    if not exps:
        return
    cur = conn.cursor()
    for exp in exps:
        state_dict = exp.state.to_dict() if exp.state else None
        next_state_dict = exp.next_state.to_dict() if exp.next_state else None
        state_emb = exp.state.domain_embedding if exp.state and exp.state.domain_embedding else None
        next_state_emb = exp.next_state.domain_embedding if exp.next_state and exp.next_state.domain_embedding else None

        cur.execute(
            """
            INSERT INTO rl_experience_buffer
                (id, run_id, step_label, action_category, action_taken,
                 alternatives, reward, reward_components,
                 state_structured, state_embedding,
                 next_state_structured, next_state_embedding,
                 done, outcome_metadata, policy_version, app_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                    %s::jsonb, %s, %s::jsonb, %s,
                    %s, %s::jsonb, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                exp.id, exp.run_id, exp.step_label,
                exp.action_category, exp.action_taken,
                exp.alternatives, exp.reward,
                json.dumps(exp.reward_components, default=str),
                json.dumps(state_dict, default=str) if state_dict else None,
                state_emb,
                json.dumps(next_state_dict, default=str) if next_state_dict else None,
                next_state_emb,
                exp.done,
                json.dumps(exp.outcome_metadata, default=str),
                exp.policy_version,
                exp.app_id,
            ),
        )
    conn.commit()
    cur.close()


# ── Episode CRUD ─────────────────────────────────────────────────────────────


def save_episode(conn: Any, summary: EpisodeSummary) -> None:
    """Insert or update an episode summary."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rl_episodes
            (run_id, total_reward, num_decisions, actions_taken,
             avg_quality, health_score, policy_version, app_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            total_reward = EXCLUDED.total_reward,
            num_decisions = EXCLUDED.num_decisions,
            actions_taken = EXCLUDED.actions_taken,
            avg_quality = EXCLUDED.avg_quality,
            health_score = EXCLUDED.health_score,
            policy_version = EXCLUDED.policy_version,
            app_id = EXCLUDED.app_id
        """,
        (
            summary.run_id, summary.total_reward, summary.num_decisions,
            summary.actions_taken, summary.avg_quality,
            summary.health_score, summary.policy_version, summary.app_id,
        ),
    )
    conn.commit()
    cur.close()


# ── Policy CRUD ──────────────────────────────────────────────────────────────


def save_policy_version(conn: Any, pv: PolicyVersion) -> None:
    """Insert a policy version."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rl_policy_versions
            (version, trained_at, experience_count, avg_reward,
             parameters, method, is_active, app_id)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
        ON CONFLICT (version) DO UPDATE SET
            trained_at = EXCLUDED.trained_at,
            experience_count = EXCLUDED.experience_count,
            avg_reward = EXCLUDED.avg_reward,
            parameters = EXCLUDED.parameters,
            method = EXCLUDED.method,
            is_active = EXCLUDED.is_active,
            app_id = EXCLUDED.app_id
        """,
        (
            pv.version, pv.trained_at or datetime.now(timezone.utc).isoformat(),
            pv.experience_count, pv.avg_reward,
            json.dumps(pv.parameters, default=str),
            pv.method, pv.is_active, pv.app_id,
        ),
    )
    conn.commit()
    cur.close()


def load_active_policy(conn: Any, app_id: str = "base") -> PolicyVersion | None:
    """Load the currently active policy version for an app, or None."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT version, trained_at, experience_count, avg_reward,
               parameters, method, is_active, app_id
        FROM rl_policy_versions
        WHERE is_active = TRUE AND (app_id = %s OR app_id = 'base')
        ORDER BY version DESC
        LIMIT 1
        """,
        (app_id,),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    return PolicyVersion(
        version=row[0],
        trained_at=str(row[1]) if row[1] else "",
        experience_count=row[2] or 0,
        avg_reward=row[3] or 0.0,
        parameters=row[4] if isinstance(row[4], dict) else json.loads(row[4] or "{}"),
        method=row[5] or "thompson_sampling",
        is_active=bool(row[6]),
        app_id=row[7] or "base",
    )


# ── Experience queries ───────────────────────────────────────────────────────


def load_experiences(
    conn: Any,
    limit: int = 500,
    min_reward: float | None = None,
    since_version: int | None = None,
    app_id: str = "base",
) -> list[Experience]:
    """Load experiences from the buffer, newest first, scoped by app_id."""
    cur = conn.cursor()
    conditions = ["(app_id = %s OR app_id = 'base')"]
    params: list[Any] = [app_id]

    if min_reward is not None:
        conditions.append("reward >= %s")
        params.append(min_reward)
    if since_version is not None:
        conditions.append("policy_version >= %s")
        params.append(since_version)

    where = "WHERE " + " AND ".join(conditions)

    cur.execute(
        f"""
        SELECT id, run_id, step_label, action_category, action_taken,
               alternatives, reward, reward_components,
               state_structured, next_state_structured,
               done, outcome_metadata, policy_version, app_id
        FROM rl_experience_buffer
        {where}
        ORDER BY created_at DESC
        LIMIT %s
        """,
        params + [limit],
    )
    rows = cur.fetchall()
    cur.close()

    results: list[Experience] = []
    for row in rows:
        state_dict = row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}")
        next_state_dict = row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}")
        reward_comp = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
        outcome_meta = row[11] if isinstance(row[11], dict) else json.loads(row[11] or "{}")

        exp = Experience(
            id=row[0],
            run_id=row[1],
            step_label=row[2],
            action_category=row[3],
            action_taken=row[4],
            alternatives=list(row[5] or []),
            reward=row[6],
            reward_components=reward_comp,
            done=bool(row[10]),
            outcome_metadata=outcome_meta,
            policy_version=row[12] or 0,
            app_id=row[13] or "base",
        )
        if state_dict:
            exp.state = StateSnapshot.from_dict(state_dict)
        if next_state_dict:
            exp.next_state = StateSnapshot.from_dict(next_state_dict)
        results.append(exp)
    return results


# ── Stats & Cleanup ──────────────────────────────────────────────────────────


def get_rl_stats(conn: Any) -> dict[str, Any]:
    """Aggregate statistics for the RL subsystem."""
    cur = conn.cursor()
    stats: dict[str, Any] = {}

    cur.execute("SELECT count(*) FROM rl_experience_buffer")
    stats["total_experiences"] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM rl_episodes")
    stats["total_episodes"] = cur.fetchone()[0]

    cur.execute("SELECT count(*) FROM rl_policy_versions")
    stats["total_policy_versions"] = cur.fetchone()[0]

    cur.execute(
        "SELECT avg(reward) FROM rl_experience_buffer WHERE reward IS NOT NULL"
    )
    row = cur.fetchone()
    stats["avg_reward"] = float(row[0]) if row[0] is not None else 0.0

    cur.execute(
        """
        SELECT action_category, action_taken, count(*), avg(reward)
        FROM rl_experience_buffer
        WHERE reward IS NOT NULL
        GROUP BY action_category, action_taken
        ORDER BY count(*) DESC
        """
    )
    stats["action_distribution"] = [
        {
            "category": r[0],
            "action": r[1],
            "count": r[2],
            "avg_reward": float(r[3]) if r[3] else 0.0,
        }
        for r in cur.fetchall()
    ]

    cur.close()
    return stats


def cleanup_old_experiences(conn: Any, keep_last_n: int = 5000) -> int:
    """Delete oldest experiences beyond keep_last_n. Returns count deleted."""
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM rl_experience_buffer
        WHERE id NOT IN (
            SELECT id FROM rl_experience_buffer
            ORDER BY created_at DESC
            LIMIT %s
        )
        """,
        (keep_last_n,),
    )
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    return deleted
