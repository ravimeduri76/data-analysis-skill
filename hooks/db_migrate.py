"""
db_migrate.py — Idempotent schema migration for DESO multi-application support.

Adds app_id and solution_id discriminator columns to existing knowledge graph
tables, enabling application-scoped queries while preserving backward compatibility.

All existing data receives DEFAULT 'base' — no UPDATE needed.

Usage:
    from hooks.db_migrate import run_migration
    run_migration()          # uses DOMAIN_DB_URL from environment
    run_migration(conn)      # uses provided connection
"""

from __future__ import annotations

import os
from typing import Optional


def _load_env() -> None:
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


_MIGRATION_SQL = """
-- domain_graph tables
ALTER TABLE domain_models ADD COLUMN IF NOT EXISTS app_id VARCHAR(50) DEFAULT 'base';
ALTER TABLE domain_models ADD COLUMN IF NOT EXISTS solution_id VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_domain_models_app ON domain_models (app_id);

-- intent_graph tables
ALTER TABLE intent_nodes ADD COLUMN IF NOT EXISTS app_id VARCHAR(50) DEFAULT 'base';
ALTER TABLE intent_nodes ADD COLUMN IF NOT EXISTS solution_id VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_intent_nodes_app ON intent_nodes (app_id);

-- execution_graph: agent_cards
ALTER TABLE agent_cards ADD COLUMN IF NOT EXISTS app_id VARCHAR(50) DEFAULT 'base';
CREATE INDEX IF NOT EXISTS idx_agent_cards_app ON agent_cards (app_id);

-- execution_graph: execution_plans
ALTER TABLE execution_plans ADD COLUMN IF NOT EXISTS app_id VARCHAR(50) DEFAULT 'base';
ALTER TABLE execution_plans ADD COLUMN IF NOT EXISTS solution_id VARCHAR(100);
CREATE INDEX IF NOT EXISTS idx_exec_plans_app ON execution_plans (app_id);

-- execution_graph: execution_examples
ALTER TABLE execution_examples ADD COLUMN IF NOT EXISTS app_id VARCHAR(50) DEFAULT 'base';
CREATE INDEX IF NOT EXISTS idx_exec_examples_app ON execution_examples (app_id);

-- semantic matching: solution embeddings (for resolve_app_from_query)
CREATE TABLE IF NOT EXISTS solution_embeddings (
    id          SERIAL PRIMARY KEY,
    app_id      VARCHAR(50) NOT NULL,
    solution_id VARCHAR(100) NOT NULL,
    embed_text  TEXT NOT NULL,
    embedding   FLOAT4[] NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (app_id, solution_id)
);

-- semantic matching: domain keyword embeddings (for domain_detector.py)
CREATE TABLE IF NOT EXISTS domain_keyword_embeddings (
    id          SERIAL PRIMARY KEY,
    domain_name VARCHAR(200) NOT NULL,
    category    VARCHAR(50) NOT NULL,
    embed_text  TEXT NOT NULL,
    embedding   FLOAT4[] NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (domain_name, category)
);

-- semantic matching: domain alias embeddings (for _resolve_domain)
CREATE TABLE IF NOT EXISTS domain_alias_embeddings (
    id         SERIAL PRIMARY KEY,
    alias_text VARCHAR(300) NOT NULL UNIQUE,
    canonical  VARCHAR(200) NOT NULL,
    embedding  FLOAT4[] NOT NULL
);

-- semantic matching: plan pattern embeddings (for _pattern_match)
CREATE TABLE IF NOT EXISTS plan_pattern_embeddings (
    id           SERIAL PRIMARY KEY,
    plan_id      INT NOT NULL,
    pattern_text TEXT NOT NULL,
    embedding    FLOAT4[] NOT NULL,
    UNIQUE (plan_id, pattern_text)
);
"""


def run_migration(conn=None) -> None:
    """Run the multi-app migration (idempotent).

    If no connection is provided, creates one from DOMAIN_DB_URL.
    """
    close_conn = False
    if conn is None:
        _load_env()
        url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
        import psycopg2
        conn = psycopg2.connect(url)
        close_conn = True

    try:
        with conn.cursor() as cur:
            cur.execute(_MIGRATION_SQL)
        conn.commit()
    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    run_migration()
    print("Migration complete: app_id/solution_id columns added.")
