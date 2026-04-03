"""
execution_graph.py -- Execution Graph: plan template storage, LLM-based reasoning,
and semantic matching for the DESO Solution Architect (SA3).

Stores execution plan templates in PostgreSQL (same domain DB as intent_graph.py
and domain_graph.py). Plans are composed of ordered steps that reference agent cards
(worker registry). Plans are matched to resolved intents via a three-tier waterfall:

  1. LLM Reasoning  — Claude API with few-shot examples (if api_key provided)
  2. Pattern Match   — intent_patterns array on execution_plans table
  3. Semantic Search  — BGE-M3 cosine similarity over plan embeddings

Each plan is a DAG of execution steps, with typed dependencies and optional HITL gates.

Public API
----------
get_execution_conn()                                              -> connection
init_execution_db(conn)                                           -> None

# Agent Cards
upsert_agent_card(conn, agent_id, name, type_, ...)               -> int
get_agent_card(conn, agent_id)                                    -> dict|None
list_agent_cards(conn, type_filter, domain_filter)                -> list[dict]

# Plans
upsert_execution_plan(conn, name, label, description, ...)        -> int
get_execution_plan(conn, name)                                    -> dict|None  (includes steps + deps)
list_execution_plans(conn)                                        -> list[dict]

# Steps + Dependencies
add_execution_step(conn, plan_name, step_order, agent_id, ...)    -> int
add_execution_dependency(conn, plan_name, from_agent_id, ...)     -> int

# Few-Shot Examples
add_execution_example(conn, example_name, intent_context, ...)    -> int
get_execution_examples(conn, limit)                               -> list[dict]

# Embedding (BGE-M3, same pattern as intent_graph)
embed_execution_plans(conn)                                       -> int
embed_agent_cards(conn)                                           -> int
backfill_execution_embeddings(conn)                               -> int

# Semantic Search (fallback)
find_candidate_plans(conn, query, limit, domain_filter, ...)      -> list[dict]

# Main Reasoning Engine
resolve_execution_plan(conn, intent_context, dcp, api_key=None)   -> dict
render_execution_plan(plan)                                       -> str
render_execution_plan_as_ap(plan, is_context)                     -> str

Database
--------
PostgreSQL: domain DB  (same as intent_graph.py / domain_graph.py)
DOMAIN_DB_URL env var (default: postgresql://postgres:memorystorepg@localhost:5432/domain)
Reads from C:\\Users\\Ravi\\Data_Analysis\\.env if present.
"""

from __future__ import annotations

import json
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Environment + connection (reuses intent_graph pattern)
# ---------------------------------------------------------------------------

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


def get_execution_conn():
    """Return a psycopg2 connection to the domain DB (where execution tables live)."""
    _load_env()
    url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
    import psycopg2
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS agent_cards (
    id               SERIAL PRIMARY KEY,
    agent_id         VARCHAR(20) NOT NULL UNIQUE,
    name             VARCHAR(200) NOT NULL,
    type             VARCHAR(50) NOT NULL DEFAULT 'worker',
    lifecycle_stage  VARCHAR(100),
    domain           VARCHAR(200) NOT NULL DEFAULT 'Any',
    capability       TEXT,
    skill_workflow   VARCHAR(200),
    ingress_schema   JSONB,
    egress_schema    JSONB,
    runtime_hint     VARCHAR(50) DEFAULT 'sync',
    hitl_required    BOOLEAN DEFAULT FALSE,
    notes            TEXT,
    embedding        FLOAT4[],
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS execution_plans (
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(200) NOT NULL UNIQUE,
    label            VARCHAR(300) NOT NULL,
    description      TEXT,
    intent_patterns  TEXT[],
    lifecycle_stages TEXT[],
    domains          TEXT[],
    requires_dataset BOOLEAN DEFAULT TRUE,
    conditions       JSONB,
    embedding        FLOAT4[],
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS execution_steps (
    id               SERIAL PRIMARY KEY,
    plan_id          INT NOT NULL REFERENCES execution_plans(id) ON DELETE CASCADE,
    step_order       INT NOT NULL,
    agent_card_id    INT NOT NULL REFERENCES agent_cards(id) ON DELETE CASCADE,
    purpose          TEXT,
    ingress_map      JSONB,
    egress_keys      TEXT[],
    is_optional      BOOLEAN DEFAULT FALSE,
    condition        TEXT,
    hitl_gate        VARCHAR(20),
    hitl_reason      TEXT,
    UNIQUE (plan_id, step_order, agent_card_id)
);

CREATE TABLE IF NOT EXISTS execution_dependencies (
    id               SERIAL PRIMARY KEY,
    plan_id          INT NOT NULL REFERENCES execution_plans(id) ON DELETE CASCADE,
    from_step_id     INT NOT NULL REFERENCES execution_steps(id) ON DELETE CASCADE,
    to_step_id       INT NOT NULL REFERENCES execution_steps(id) ON DELETE CASCADE,
    dep_type         VARCHAR(50) NOT NULL DEFAULT 'sequence',
    description      TEXT
);

CREATE TABLE IF NOT EXISTS execution_examples (
    id               SERIAL PRIMARY KEY,
    example_name     VARCHAR(200) NOT NULL UNIQUE,
    intent_context   JSONB NOT NULL,
    domain_context   JSONB NOT NULL,
    selected_plan    VARCHAR(200) NOT NULL,
    reasoning        TEXT NOT NULL,
    plan_overrides   JSONB,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_cards_type     ON agent_cards (type);
CREATE INDEX IF NOT EXISTS idx_agent_cards_domain   ON agent_cards (domain);
CREATE INDEX IF NOT EXISTS idx_exec_plans_name      ON execution_plans (name);
CREATE INDEX IF NOT EXISTS idx_exec_steps_plan      ON execution_steps (plan_id);
CREATE INDEX IF NOT EXISTS idx_exec_steps_agent     ON execution_steps (agent_card_id);
CREATE INDEX IF NOT EXISTS idx_exec_deps_plan       ON execution_dependencies (plan_id);
CREATE INDEX IF NOT EXISTS idx_exec_deps_from       ON execution_dependencies (from_step_id);
CREATE INDEX IF NOT EXISTS idx_exec_deps_to         ON execution_dependencies (to_step_id);
CREATE INDEX IF NOT EXISTS idx_exec_examples_plan   ON execution_examples (selected_plan);
"""


def init_execution_db(conn) -> None:
    """Create all execution graph tables and indexes (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_INIT_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Agent Card CRUD
# ---------------------------------------------------------------------------

def upsert_agent_card(
    conn,
    agent_id: str,
    name: str,
    type_: str = "worker",
    lifecycle_stage: Optional[str] = None,
    domain: str = "Any",
    capability: Optional[str] = None,
    skill_workflow: Optional[str] = None,
    ingress_schema: Optional[dict] = None,
    egress_schema: Optional[dict] = None,
    runtime_hint: str = "sync",
    hitl_required: bool = False,
    notes: Optional[str] = None,
) -> int:
    """Insert or update an agent card. Returns the card id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_cards
                (agent_id, name, type, lifecycle_stage, domain, capability,
                 skill_workflow, ingress_schema, egress_schema, runtime_hint,
                 hitl_required, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (agent_id) DO UPDATE
              SET name            = EXCLUDED.name,
                  type            = EXCLUDED.type,
                  lifecycle_stage = EXCLUDED.lifecycle_stage,
                  domain          = EXCLUDED.domain,
                  capability      = EXCLUDED.capability,
                  skill_workflow  = EXCLUDED.skill_workflow,
                  ingress_schema  = EXCLUDED.ingress_schema,
                  egress_schema   = EXCLUDED.egress_schema,
                  runtime_hint    = EXCLUDED.runtime_hint,
                  hitl_required   = EXCLUDED.hitl_required,
                  notes           = EXCLUDED.notes
            RETURNING id
            """,
            (
                agent_id, name, type_, lifecycle_stage, domain, capability,
                skill_workflow,
                json.dumps(ingress_schema) if ingress_schema else None,
                json.dumps(egress_schema) if egress_schema else None,
                runtime_hint, hitl_required, notes,
            ),
        )
        return cur.fetchone()[0]


def get_agent_card(conn, agent_id: str) -> Optional[dict]:
    """Fetch a single agent card by agent_id. Returns None if not found."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, agent_id, name, type, lifecycle_stage, domain, capability,
                   skill_workflow, ingress_schema, egress_schema, runtime_hint,
                   hitl_required, notes
            FROM agent_cards WHERE agent_id = %s
            """,
            (agent_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "agent_id": row[1], "name": row[2], "type": row[3],
        "lifecycle_stage": row[4], "domain": row[5], "capability": row[6],
        "skill_workflow": row[7],
        "ingress_schema": row[8] if isinstance(row[8], dict) else (json.loads(row[8]) if row[8] else None),
        "egress_schema": row[9] if isinstance(row[9], dict) else (json.loads(row[9]) if row[9] else None),
        "runtime_hint": row[10], "hitl_required": row[11], "notes": row[12],
    }


def list_agent_cards(
    conn,
    type_filter: Optional[str] = None,
    domain_filter: Optional[str] = None,
) -> list[dict]:
    """List agent cards with optional type/domain filters."""
    conditions: list[str] = []
    params: list = []
    if type_filter:
        conditions.append("type = %s")
        params.append(type_filter)
    if domain_filter:
        conditions.append("(domain = %s OR domain = 'Any')")
        params.append(domain_filter)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, agent_id, name, type, lifecycle_stage, domain, capability,
                   skill_workflow, runtime_hint, hitl_required, notes
            FROM agent_cards {where}
            ORDER BY agent_id
            """,
            params,
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0], "agent_id": r[1], "name": r[2], "type": r[3],
            "lifecycle_stage": r[4], "domain": r[5], "capability": r[6],
            "skill_workflow": r[7], "runtime_hint": r[8],
            "hitl_required": r[9], "notes": r[10],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Execution Plan CRUD
# ---------------------------------------------------------------------------

def upsert_execution_plan(
    conn,
    name: str,
    label: str,
    description: str,
    intent_patterns: Optional[list[str]] = None,
    lifecycle_stages: Optional[list[str]] = None,
    domains: Optional[list[str]] = None,
    requires_dataset: bool = True,
    conditions: Optional[dict] = None,
) -> int:
    """Insert or update an execution plan template. Returns plan id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO execution_plans
                (name, label, description, intent_patterns, lifecycle_stages,
                 domains, requires_dataset, conditions)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET label            = EXCLUDED.label,
                  description      = EXCLUDED.description,
                  intent_patterns  = EXCLUDED.intent_patterns,
                  lifecycle_stages = EXCLUDED.lifecycle_stages,
                  domains          = EXCLUDED.domains,
                  requires_dataset = EXCLUDED.requires_dataset,
                  conditions       = EXCLUDED.conditions
            RETURNING id
            """,
            (
                name, label, description,
                intent_patterns, lifecycle_stages, domains,
                requires_dataset,
                json.dumps(conditions) if conditions else None,
            ),
        )
        return cur.fetchone()[0]


def get_execution_plan(conn, name: str) -> Optional[dict]:
    """Fetch a plan by name, including its steps and dependencies."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, label, description, intent_patterns, lifecycle_stages,
                   domains, requires_dataset, conditions
            FROM execution_plans WHERE name = %s
            """,
            (name,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    plan_id = row[0]
    plan = {
        "id": plan_id, "name": row[1], "label": row[2], "description": row[3],
        "intent_patterns": row[4] or [], "lifecycle_stages": row[5] or [],
        "domains": row[6] or [], "requires_dataset": row[7],
        "conditions": row[8] if isinstance(row[8], dict) else (json.loads(row[8]) if row[8] else None),
    }
    # Fetch steps
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.step_order, ac.agent_id, ac.name, s.purpose,
                   s.ingress_map, s.egress_keys, s.is_optional, s.condition,
                   s.hitl_gate, s.hitl_reason
            FROM execution_steps s
            JOIN agent_cards ac ON ac.id = s.agent_card_id
            WHERE s.plan_id = %s
            ORDER BY s.step_order, ac.agent_id
            """,
            (plan_id,),
        )
        steps = [
            {
                "step_id": r[0], "step_order": r[1], "agent_id": r[2],
                "agent_name": r[3], "purpose": r[4],
                "ingress_map": r[5] if isinstance(r[5], dict) else (json.loads(r[5]) if r[5] else None),
                "egress_keys": r[6] or [], "is_optional": r[7],
                "condition": r[8], "hitl_gate": r[9], "hitl_reason": r[10],
            }
            for r in cur.fetchall()
        ]
    plan["steps"] = steps
    # Fetch dependencies
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.id, d.from_step_id, d.to_step_id, d.dep_type, d.description
            FROM execution_dependencies d
            WHERE d.plan_id = %s
            """,
            (plan_id,),
        )
        deps = [
            {
                "dep_id": r[0], "from_step_id": r[1], "to_step_id": r[2],
                "dep_type": r[3], "description": r[4],
            }
            for r in cur.fetchall()
        ]
    plan["dependencies"] = deps
    return plan


def list_execution_plans(conn) -> list[dict]:
    """List all execution plan templates (without steps/deps for brevity)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, label, description, intent_patterns, lifecycle_stages,
                   domains, requires_dataset
            FROM execution_plans
            ORDER BY name
            """
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0], "name": r[1], "label": r[2], "description": r[3],
            "intent_patterns": r[4] or [], "lifecycle_stages": r[5] or [],
            "domains": r[6] or [], "requires_dataset": r[7],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Steps + Dependencies
# ---------------------------------------------------------------------------

def add_execution_step(
    conn,
    plan_name: str,
    step_order: int,
    agent_id: str,
    purpose: str,
    ingress_map: Optional[dict] = None,
    egress_keys: Optional[list[str]] = None,
    is_optional: bool = False,
    condition: Optional[str] = None,
    hitl_gate: Optional[str] = None,
    hitl_reason: Optional[str] = None,
) -> int:
    """Add a step to an execution plan. Returns step id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM execution_plans WHERE name = %s", (plan_name,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Plan not found: {plan_name!r}")
        plan_id = row[0]

        cur.execute("SELECT id FROM agent_cards WHERE agent_id = %s", (agent_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Agent card not found: {agent_id!r}")
        agent_card_id = row[0]

        cur.execute(
            """
            INSERT INTO execution_steps
                (plan_id, step_order, agent_card_id, purpose, ingress_map,
                 egress_keys, is_optional, condition, hitl_gate, hitl_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (plan_id, step_order, agent_card_id) DO UPDATE
              SET purpose     = EXCLUDED.purpose,
                  ingress_map = EXCLUDED.ingress_map,
                  egress_keys = EXCLUDED.egress_keys,
                  is_optional = EXCLUDED.is_optional,
                  condition   = EXCLUDED.condition,
                  hitl_gate   = EXCLUDED.hitl_gate,
                  hitl_reason = EXCLUDED.hitl_reason
            RETURNING id
            """,
            (
                plan_id, step_order, agent_card_id, purpose,
                json.dumps(ingress_map) if ingress_map else None,
                egress_keys, is_optional, condition, hitl_gate, hitl_reason,
            ),
        )
        return cur.fetchone()[0]


def add_execution_dependency(
    conn,
    plan_name: str,
    from_agent_id: str,
    to_agent_id: str,
    from_step_order: int = 1,
    to_step_order: int = 2,
    dep_type: str = "sequence",
    description: Optional[str] = None,
) -> int:
    """Add a dependency edge between two steps in a plan. Returns dependency id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM execution_plans WHERE name = %s", (plan_name,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Plan not found: {plan_name!r}")
        plan_id = row[0]

        # Find from_step
        cur.execute(
            """
            SELECT s.id FROM execution_steps s
            JOIN agent_cards ac ON ac.id = s.agent_card_id
            WHERE s.plan_id = %s AND ac.agent_id = %s AND s.step_order = %s
            """,
            (plan_id, from_agent_id, from_step_order),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"From step not found: plan={plan_name}, agent={from_agent_id}, order={from_step_order}")
        from_step_id = row[0]

        # Find to_step
        cur.execute(
            """
            SELECT s.id FROM execution_steps s
            JOIN agent_cards ac ON ac.id = s.agent_card_id
            WHERE s.plan_id = %s AND ac.agent_id = %s AND s.step_order = %s
            """,
            (plan_id, to_agent_id, to_step_order),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"To step not found: plan={plan_name}, agent={to_agent_id}, order={to_step_order}")
        to_step_id = row[0]

        cur.execute(
            """
            INSERT INTO execution_dependencies
                (plan_id, from_step_id, to_step_id, dep_type, description)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (plan_id, from_step_id, to_step_id, dep_type, description),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Few-Shot Examples
# ---------------------------------------------------------------------------

def add_execution_example(
    conn,
    example_name: str,
    intent_context: dict,
    domain_context: dict,
    selected_plan: str,
    reasoning: str,
    plan_overrides: Optional[dict] = None,
) -> int:
    """Add a few-shot reasoning example. Returns example id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO execution_examples
                (example_name, intent_context, domain_context, selected_plan,
                 reasoning, plan_overrides)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (example_name) DO UPDATE
              SET intent_context = EXCLUDED.intent_context,
                  domain_context = EXCLUDED.domain_context,
                  selected_plan  = EXCLUDED.selected_plan,
                  reasoning      = EXCLUDED.reasoning,
                  plan_overrides = EXCLUDED.plan_overrides
            RETURNING id
            """,
            (
                example_name,
                json.dumps(intent_context, default=str),
                json.dumps(domain_context, default=str),
                selected_plan, reasoning,
                json.dumps(plan_overrides, default=str) if plan_overrides else None,
            ),
        )
        return cur.fetchone()[0]


def get_execution_examples(conn, limit: int = 20) -> list[dict]:
    """Fetch few-shot examples, most recent first."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, example_name, intent_context, domain_context,
                   selected_plan, reasoning, plan_overrides
            FROM execution_examples
            ORDER BY id
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    results = []
    for r in rows:
        ic = r[2] if isinstance(r[2], dict) else (json.loads(r[2]) if r[2] else {})
        dc = r[3] if isinstance(r[3], dict) else (json.loads(r[3]) if r[3] else {})
        po = r[6] if isinstance(r[6], dict) else (json.loads(r[6]) if r[6] else None)
        results.append({
            "id": r[0], "example_name": r[1],
            "intent_context": ic, "domain_context": dc,
            "selected_plan": r[4], "reasoning": r[5],
            "plan_overrides": po,
        })
    return results


# ---------------------------------------------------------------------------
# Embedding (BGE-M3, same pattern as intent_graph)
# ---------------------------------------------------------------------------

_embed_cache: dict = {}


def _get_embed_model():
    """Load BGE-M3 (cached). Returns None if sentence-transformers unavailable."""
    if "model" not in _embed_cache:
        try:
            from sentence_transformers import SentenceTransformer
            import warnings
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _embed_cache["model"] = SentenceTransformer("BAAI/bge-m3")
        except Exception:
            _embed_cache["model"] = None
    return _embed_cache["model"]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    if model is None:
        return [[] for _ in texts]
    import numpy as np
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def _plan_to_embed_text(row: dict) -> str:
    """Build rich embedding text from a plan dict."""
    parts = [f"plan: {row.get('label', '')}"]
    if row.get("description"):
        parts.append(row["description"])
    if row.get("intent_patterns"):
        parts.append(f"intents: {', '.join(row['intent_patterns'])}")
    if row.get("lifecycle_stages"):
        parts.append(f"lifecycle: {', '.join(row['lifecycle_stages'])}")
    if row.get("domains"):
        parts.append(f"domains: {', '.join(row['domains'])}")
    return ". ".join(parts)


def _card_to_embed_text(row: dict) -> str:
    """Build rich embedding text from an agent card dict."""
    parts = [f"agent: {row.get('name', '')} ({row.get('agent_id', '')})"]
    if row.get("capability"):
        parts.append(row["capability"])
    if row.get("domain") and row["domain"] != "Any":
        parts.append(f"domain: {row['domain']}")
    if row.get("lifecycle_stage"):
        parts.append(f"lifecycle: {row['lifecycle_stage']}")
    return ". ".join(parts)


def embed_execution_plans(conn) -> int:
    """Embed all execution_plans that have NULL embedding. Returns count."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, label, description, intent_patterns,
                   lifecycle_stages, domains
            FROM execution_plans WHERE embedding IS NULL
            """
        )
        rows = cur.fetchall()
    if not rows:
        return 0
    plan_dicts = [
        {"id": r[0], "name": r[1], "label": r[2], "description": r[3],
         "intent_patterns": r[4] or [], "lifecycle_stages": r[5] or [],
         "domains": r[6] or []}
        for r in rows
    ]
    texts = [_plan_to_embed_text(d) for d in plan_dicts]
    embeds = _embed_texts(texts)
    with conn.cursor() as cur:
        for d, emb in zip(plan_dicts, embeds):
            cur.execute(
                "UPDATE execution_plans SET embedding = %s WHERE id = %s",
                (emb if emb else None, d["id"]),
            )
    conn.commit()
    return len(plan_dicts)


def embed_agent_cards(conn) -> int:
    """Embed all agent_cards that have NULL embedding. Returns count."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, agent_id, name, capability, domain, lifecycle_stage
            FROM agent_cards WHERE embedding IS NULL
            """
        )
        rows = cur.fetchall()
    if not rows:
        return 0
    card_dicts = [
        {"id": r[0], "agent_id": r[1], "name": r[2], "capability": r[3],
         "domain": r[4], "lifecycle_stage": r[5]}
        for r in rows
    ]
    texts = [_card_to_embed_text(d) for d in card_dicts]
    embeds = _embed_texts(texts)
    with conn.cursor() as cur:
        for d, emb in zip(card_dicts, embeds):
            cur.execute(
                "UPDATE agent_cards SET embedding = %s WHERE id = %s",
                (emb if emb else None, d["id"]),
            )
    conn.commit()
    return len(card_dicts)


def backfill_execution_embeddings(conn) -> int:
    """Embed any unembedded plans and agent cards. Returns total count."""
    return embed_execution_plans(conn) + embed_agent_cards(conn)


# ---------------------------------------------------------------------------
# Semantic search — find candidate plans
# ---------------------------------------------------------------------------

def find_candidate_plans(
    conn,
    query: str,
    limit: int = 5,
    domain_filter: Optional[str] = None,
    lifecycle_filter: Optional[str] = None,
) -> list[dict]:
    """Embed query and return top-k execution plans ranked by cosine similarity.

    Falls back to keyword search over plan names/descriptions if BGE-M3 unavailable.
    """
    model = _get_embed_model()
    if model is None:
        return _keyword_plan_search(conn, query, limit, domain_filter)

    try:
        import numpy as np
    except ImportError:
        return _keyword_plan_search(conn, query, limit, domain_filter)

    q_vec = model.encode([query], normalize_embeddings=True)[0]

    conditions: list[str] = []
    params: list = []
    if domain_filter:
        conditions.append("(%s = ANY(domains) OR 'Any' = ANY(domains))")
        params.append(domain_filter)
    if lifecycle_filter:
        conditions.append("%s = ANY(lifecycle_stages)")
        params.append(lifecycle_filter)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, label, description, intent_patterns,
                   lifecycle_stages, domains, requires_dataset, embedding
            FROM execution_plans
            {where}
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return _keyword_plan_search(conn, query, limit, domain_filter)

    results = []
    for row in rows:
        emb = row[8]
        if not emb:
            continue
        vec = np.array(emb, dtype=np.float32)
        score = float(np.dot(q_vec, vec))
        results.append({
            "id": row[0], "name": row[1], "label": row[2],
            "description": row[3] or "", "intent_patterns": row[4] or [],
            "lifecycle_stages": row[5] or [], "domains": row[6] or [],
            "requires_dataset": row[7], "score": score,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def _keyword_plan_search(
    conn,
    query: str,
    limit: int,
    domain_filter: Optional[str],
) -> list[dict]:
    """Keyword fallback: search plan names, labels, descriptions."""
    tokens = [t.lower() for t in query.split() if len(t) > 2]
    if not tokens:
        return []
    like_clauses = " OR ".join(
        ["LOWER(name) LIKE %s OR LOWER(label) LIKE %s OR LOWER(description) LIKE %s"] * len(tokens)
    )
    like_params = [v for t in tokens for v in (f"%{t}%", f"%{t}%", f"%{t}%")]
    domain_clause = ""
    extra_params: list = []
    if domain_filter:
        domain_clause = "AND (%s = ANY(domains) OR 'Any' = ANY(domains))"
        extra_params.append(domain_filter)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, label, description, intent_patterns,
                   lifecycle_stages, domains, requires_dataset
            FROM execution_plans
            WHERE ({like_clauses}) {domain_clause}
            LIMIT %s
            """,
            like_params + extra_params + [limit],
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0], "name": r[1], "label": r[2], "description": r[3] or "",
            "intent_patterns": r[4] or [], "lifecycle_stages": r[5] or [],
            "domains": r[6] or [], "requires_dataset": r[7], "score": 1.0,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# LLM Reasoning Engine
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are SA3's execution plan resolver. Given an intent context and domain context,
select the best execution plan template from the available plans and optionally apply overrides.

Rules:
- Pick the plan whose intent_patterns best match the resolved intent.
- If no plan matches well, return plan_name=null.
- Consider domain, lifecycle_stage, and whether a dataset is available.
- Apply overrides to add, remove, or modify steps when the template needs adjustment.
- Confidence: 0.95 for exact intent_pattern match, 0.85 for close match, 0.70 for reasonable but uncertain.
- If the intent is ambiguous or cross-domain, lower confidence and add warnings.

Return ONLY valid JSON (no markdown, no explanation outside JSON):
{
  "plan_name": "<plan name or null>",
  "reasoning": "<1-2 sentence explanation>",
  "overrides": {
    "add_steps": [{"agent_id": "WA_", "step_order": N, "purpose": "..."}],
    "remove_agents": ["WA_"],
    "modify_steps": [{"agent_id": "WA_", "changes": {"is_optional": true}}]
  },
  "confidence": 0.0-1.0,
  "warnings": ["..."]
}"""


def _build_llm_prompt(
    intent_context: dict,
    dcp: dict,
    agent_cards: list[dict],
    plans: list[dict],
    examples: list[dict],
) -> list[dict]:
    """Build Claude API messages for plan resolution."""
    # Abbreviated agent cards
    cards_text = "\n".join(
        f"- {c['agent_id']}: {c['name']} | {c.get('capability', '')[:100]} | "
        f"domain={c.get('domain', 'Any')} | hitl={c.get('hitl_required', False)}"
        for c in agent_cards
    )
    # Abbreviated plans
    plans_text = "\n".join(
        f"- {p['name']}: {p.get('label', '')} | intents={p.get('intent_patterns', [])} | "
        f"lifecycle={p.get('lifecycle_stages', [])} | dataset={p.get('requires_dataset', True)}"
        for p in plans
    )
    # Few-shot examples
    examples_text = ""
    for ex in examples:
        examples_text += (
            f"\n--- Example: {ex['example_name']} ---\n"
            f"Intent: {json.dumps(ex.get('intent_context', {}), default=str)}\n"
            f"Domain: {json.dumps(ex.get('domain_context', {}), default=str)}\n"
            f"Selected: {ex['selected_plan']}\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"Overrides: {json.dumps(ex.get('plan_overrides'), default=str)}\n"
        )

    user_msg = (
        f"## Available Agent Cards\n{cards_text}\n\n"
        f"## Available Plans\n{plans_text}\n\n"
        f"## Few-Shot Examples\n{examples_text}\n\n"
        f"## Current Intent Context\n{json.dumps(intent_context, default=str)}\n\n"
        f"## Current Domain Context (DCP)\n{json.dumps(dcp, default=str)}\n\n"
        f"Select the best execution plan. Return JSON only."
    )

    return [
        {"role": "user", "content": user_msg},
    ]


def _call_llm(api_key: str, messages: list[dict]) -> Optional[dict]:
    """Call LLM for plan resolution. Returns parsed JSON or None."""
    from hooks.llm_client import call_llm

    # messages list has a single user message; extract it
    user_content = messages[0]["content"] if messages else ""
    text = call_llm(
        user_content,
        max_tokens=1024,
        system=_LLM_SYSTEM_PROMPT,
        expect_json=True,
    )
    if text is None:
        return None
    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except Exception:
        return None


def _pattern_match(
    conn,
    intent_context: dict,
) -> Optional[dict]:
    """Pattern matching: check intent_patterns arrays for direct match."""
    matched = intent_context.get("matched_node")
    if not matched:
        return None
    intent_name = matched.get("name", "")
    if not intent_name:
        return None

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name FROM execution_plans
            WHERE %s = ANY(intent_patterns)
            LIMIT 1
            """,
            (intent_name,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    plan = get_execution_plan(conn, row[0])
    if plan is None:
        return None
    return {
        "plan": plan,
        "confidence": 0.80,
        "reasoning": f"Direct pattern match: intent '{intent_name}' found in plan '{plan['name']}' intent_patterns",
        "overrides": None,
        "warnings": [],
        "method": "pattern_match",
    }


def _semantic_match(
    conn,
    intent_context: dict,
    dcp: dict,
) -> Optional[dict]:
    """Semantic similarity fallback over plan embeddings."""
    matched = intent_context.get("matched_node")
    query_parts = []
    if matched:
        query_parts.append(matched.get("label", ""))
        query_parts.append(matched.get("description", ""))
    if dcp.get("summary"):
        query_parts.append(dcp["summary"])
    query = ". ".join(p for p in query_parts if p)
    if not query:
        return None

    domain = dcp.get("domain")
    candidates = find_candidate_plans(conn, query, limit=3, domain_filter=domain)
    if not candidates or candidates[0]["score"] < 0.60:
        return None

    top = candidates[0]
    plan = get_execution_plan(conn, top["name"])
    if plan is None:
        return None
    return {
        "plan": plan,
        "confidence": round(min(top["score"], 0.75), 2),
        "reasoning": f"Semantic match: plan '{plan['name']}' scored {top['score']:.3f} against intent description",
        "overrides": None,
        "warnings": ["Matched by semantic similarity — verify plan suitability"],
        "method": "semantic",
    }


def _apply_overrides(plan: dict, overrides: Optional[dict]) -> dict:
    """Apply LLM overrides to a plan's steps (add/remove/modify)."""
    if not overrides:
        return plan
    steps = list(plan.get("steps", []))

    # Remove agents
    remove_agents = overrides.get("remove_agents") or []
    if remove_agents:
        steps = [s for s in steps if s["agent_id"] not in remove_agents]

    # Add steps
    add_steps = overrides.get("add_steps") or []
    for add in add_steps:
        steps.append({
            "step_id": None, "step_order": add.get("step_order", 99),
            "agent_id": add["agent_id"], "agent_name": add.get("agent_name", ""),
            "purpose": add.get("purpose", ""), "ingress_map": None,
            "egress_keys": [], "is_optional": add.get("is_optional", False),
            "condition": None, "hitl_gate": add.get("hitl_gate"),
            "hitl_reason": add.get("hitl_reason"),
        })

    # Modify steps
    modify_steps = overrides.get("modify_steps") or []
    for mod in modify_steps:
        aid = mod.get("agent_id")
        changes = mod.get("changes", {})
        for s in steps:
            if s["agent_id"] == aid:
                s.update(changes)

    steps.sort(key=lambda s: s.get("step_order", 0))
    plan["steps"] = steps
    return plan


def resolve_execution_plan(
    conn,
    intent_context: dict,
    dcp: dict,
    api_key: Optional[str] = None,
) -> dict:
    """Main reasoning engine: resolve an intent to an execution plan.

    Waterfall:
      1. LLM reasoning (if api_key provided)
      2. Pattern matching (intent_patterns)
      3. Semantic similarity
      4. Fallback (no plan found)

    Returns:
        {
            "plan": dict|None,        # full plan with steps/deps
            "confidence": float,
            "reasoning": str,
            "overrides": dict|None,
            "warnings": list[str],
            "method": str,            # "llm"|"pattern_match"|"semantic"|"fallback"
        }
    """
    fallback = {
        "plan": None, "confidence": 0.0,
        "reasoning": "No execution plan matched — SA3 should use hardcoded rules",
        "overrides": None, "warnings": [], "method": "fallback",
    }

    # Step 1: LLM Reasoning
    if api_key:
        try:
            cards = list_agent_cards(conn)
            plans = list_execution_plans(conn)
            examples = get_execution_examples(conn, limit=10)
            messages = _build_llm_prompt(intent_context, dcp, cards, plans, examples)
            llm_result = _call_llm(api_key, messages)
            if llm_result and llm_result.get("plan_name"):
                plan = get_execution_plan(conn, llm_result["plan_name"])
                if plan:
                    overrides = llm_result.get("overrides")
                    plan = _apply_overrides(plan, overrides)
                    return {
                        "plan": plan,
                        "confidence": llm_result.get("confidence", 0.85),
                        "reasoning": llm_result.get("reasoning", "LLM selected plan"),
                        "overrides": overrides,
                        "warnings": llm_result.get("warnings", []),
                        "method": "llm",
                    }
        except Exception:
            pass  # Fall through to pattern match

    # Step 2: Pattern Matching
    try:
        result = _pattern_match(conn, intent_context)
        if result:
            return result
    except Exception:
        pass

    # Step 3: Semantic Similarity
    try:
        result = _semantic_match(conn, intent_context, dcp)
        if result:
            return result
    except Exception:
        pass

    # Step 4: Fallback
    return fallback


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_execution_plan(plan_result: dict) -> str:
    """Render a resolved execution plan as an ASCII-safe string for LLM prompts."""
    if not plan_result or not plan_result.get("plan"):
        return "[Execution Graph: no plan resolved — using hardcoded rules]"

    plan = plan_result["plan"]
    lines = [
        "=== EXECUTION GRAPH PLAN ===",
        f"Plan          : {plan['label']} ({plan['name']})",
        f"Description   : {plan.get('description', '')}",
        f"Confidence    : {plan_result.get('confidence', 0):.2f}",
        f"Method        : {plan_result.get('method', 'unknown')}",
        f"Reasoning     : {plan_result.get('reasoning', '')}",
        f"Dataset Req   : {'yes' if plan.get('requires_dataset') else 'no'}",
        "",
    ]

    steps = plan.get("steps", [])
    if steps:
        lines.append("-- Steps --")
        current_order = None
        for s in steps:
            if s["step_order"] != current_order:
                current_order = s["step_order"]
                lines.append(f"  Step {current_order}:")
            opt = " (optional)" if s.get("is_optional") else ""
            hitl = ""
            if s.get("hitl_gate"):
                hitl = f" [HITL {s['hitl_gate']}: {s.get('hitl_reason', '')}]"
            lines.append(
                f"    [{s['agent_id']}] {s.get('agent_name', '')} — {s.get('purpose', '')}{opt}{hitl}"
            )
        lines.append("")

    deps = plan.get("dependencies", [])
    if deps:
        lines.append("-- Dependencies --")
        for d in deps:
            lines.append(f"    step {d['from_step_id']} -> step {d['to_step_id']} ({d['dep_type']}): {d.get('description', '')}")
        lines.append("")

    warnings = plan_result.get("warnings", [])
    if warnings:
        lines.append("-- Warnings --")
        for w in warnings:
            lines.append(f"  ! {w}")
        lines.append("")

    if plan_result.get("overrides"):
        lines.append(f"-- Overrides Applied: {json.dumps(plan_result['overrides'], default=str)[:200]} --")
        lines.append("")

    lines.append("=== END EXECUTION PLAN ===")
    raw = "\n".join(lines)
    return raw.encode("ascii", errors="replace").decode("ascii")


def render_execution_plan_as_ap(plan_result: dict, is_context: dict) -> str:
    """Render a resolved plan in SA3's Agent Plan (AP) format.

    Parameters
    ----------
    plan_result : dict from resolve_execution_plan
    is_context  : Intent Specification dict from SA2
    """
    if not plan_result or not plan_result.get("plan"):
        return "[No execution plan resolved — SA3 must build AP manually]"

    plan = plan_result["plan"]
    steps = plan.get("steps", [])
    goal = is_context.get("explicit_goal", plan.get("description", ""))

    lines = [
        "Agent Plan",
        "----------" + "-" * len("Agent Plan"),
        f"Goal: {goal}",
        f"Source: Execution Graph — {plan['name']} (confidence={plan_result.get('confidence', 0):.2f}, method={plan_result.get('method', '')})",
        "",
        "Agents:",
    ]

    # Group steps by step_order to detect parallel
    from collections import defaultdict
    order_groups: dict[int, list[dict]] = defaultdict(list)
    for s in steps:
        order_groups[s["step_order"]].append(s)

    agent_idx = 1
    agent_labels: dict[str, str] = {}  # agent_id -> A{n} label
    prev_labels: list[str] = []

    for order in sorted(order_groups.keys()):
        group = order_groups[order]
        current_labels = []
        for s in group:
            label = f"A{agent_idx}"
            agent_labels[s["agent_id"]] = label
            current_labels.append(label)

            # Determine run order
            if not prev_labels:
                runs = "first"
            elif len(group) > 1:
                parallel_peers = [f"A{agent_idx + j}" for j in range(len(group)) if j != group.index(s)]
                runs = f"parallel with {', '.join(parallel_peers)}"
                if prev_labels:
                    runs += f" | after {', '.join(prev_labels)}"
            else:
                runs = f"after {', '.join(prev_labels)}"

            opt = " (OPTIONAL)" if s.get("is_optional") else ""
            lines.append(f"  [{label}] {s['agent_id']}: {s.get('agent_name', '')} -- role{opt}")
            lines.append(f"       Purpose : {s.get('purpose', '')}")
            ingress = s.get("ingress_map")
            if ingress:
                lines.append(f"       Ingress : {json.dumps(ingress, default=str)[:120]}")
            else:
                lines.append(f"       Ingress : $input (from prior step or DCP)")
            egress_keys = s.get("egress_keys", [])
            lines.append(f"       Egress  : {', '.join(egress_keys) if egress_keys else 'standard output'}")
            lines.append(f"       Runs    : {runs}")
            lines.append("")
            agent_idx += 1

        prev_labels = current_labels

    # HITL Gates
    hitl_lines = []
    for s in steps:
        if s.get("hitl_gate"):
            lbl = agent_labels.get(s["agent_id"], s["agent_id"])
            hitl_lines.append(f"  {s['hitl_gate'].capitalize()} {lbl} : {s.get('hitl_reason', 'User confirmation required')}")
    lines.append("HITL Gates:")
    if hitl_lines:
        lines.extend(hitl_lines)
    else:
        lines.append("  None required")
    lines.append("")

    # Blocking risks
    warnings = plan_result.get("warnings", [])
    risk_signals = is_context.get("risk_signals", [])
    missing_inputs = is_context.get("missing_inputs", [])
    lines.append("Blocking risks:")
    if warnings or risk_signals or missing_inputs:
        for w in warnings:
            lines.append(f"  - {w}")
        for r in risk_signals:
            lines.append(f"  - (IS risk) {r}")
        for m in missing_inputs:
            lines.append(f"  - (missing) {m}")
    else:
        lines.append("  None identified")

    raw = "\n".join(lines)
    return raw.encode("ascii", errors="replace").decode("ascii")
