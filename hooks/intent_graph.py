"""
intent_graph.py — Intent Knowledge Graph: hierarchical intent storage and traversal.

Stores user intents as a typed graph in PostgreSQL (same domain DB as domain_graph.py).
Intents form a three-level hierarchy:

  abstract  — high-level goal classes  (e.g. patient_risk_management)
  mid       — subject areas             (e.g. survival_analysis)
  specific  — concrete operations       (e.g. cox_regression)

Nodes are connected by typed triplets:
  IS_A             — X is a subtype of Y  (defines the hierarchy)
  REQUIRES         — X needs Y as prerequisite
  ENABLES          — completing X enables Y
  PRECEDES         — X typically runs before Y
  COMPETES_WITH    — X and Y are alternative approaches
  PART_OF          — X is a component of Y

Each node is embedded with BGE-M3 (1024-dim, L2-normalised, stored as float4[]).
Cosine similarity is computed in numpy — no pgvector required.

Given a user query, resolve_intent() returns:
  - matched_node         — best-matching intent node
  - confidence           — cosine similarity score
  - abstract_parents     — upward IS_A traversal  → latent goals
  - specific_children    — downward IS_A traversal → execution paths
  - sibling_intents      — peers sharing the same parent
  - ambiguity_candidates — other candidates within 0.05 of top score
  - requires_clarification — true when ambiguity_candidates is non-empty

Public API
----------
get_intent_conn()                                          -> connection
init_intent_db(conn)                                       -> None
upsert_intent_node(conn, ..., app_id, solution_id)        -> int   (node id)
add_intent_triplet(conn, subject_name, predicate,
                   object_name, weight)                    -> int   (triplet id)
add_intent_alias(conn, intent_name, alias, alias_type)     -> None
embed_intent_nodes(conn)                                   -> int   (count embedded)
find_candidate_intents(conn, query, limit, domain_filter,
                       app_id)                             -> list[dict]
traverse_up(conn, node_id, max_depth, app_id)              -> list[dict]
traverse_down(conn, node_id, max_depth, app_id)            -> list[dict]
resolve_intent(conn, query, domain, app_id)                -> dict  (IntentContext)
render_intent_context(ctx)                                 -> str   (ASCII-safe)
backfill_intent_embeddings(conn)                           -> int

Database
--------
PostgreSQL: domain DB  (same as domain_graph.py)
DOMAIN_DB_URL env var (default: postgresql://postgres:memorystorepg@localhost:5432/domain)
Reads from C:\\Users\\Ravi\\Data_Analysis\\.env if present.
"""

from __future__ import annotations

import os
from typing import Optional

# ---------------------------------------------------------------------------
# Environment + connection
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


def get_intent_conn():
    """Return a psycopg2 connection to the domain DB (where intent tables live)."""
    _load_env()
    url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
    import psycopg2
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS intent_nodes (
    id               SERIAL PRIMARY KEY,
    name             VARCHAR(200) NOT NULL UNIQUE,
    label            VARCHAR(300) NOT NULL,
    description      TEXT,
    level            VARCHAR(50)  NOT NULL DEFAULT 'specific',
    domain           VARCHAR(200) NOT NULL DEFAULT 'any',
    lifecycle_stage  VARCHAR(100),
    embedding        FLOAT4[],
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intent_triplets (
    id          SERIAL PRIMARY KEY,
    subject_id  INT NOT NULL REFERENCES intent_nodes(id) ON DELETE CASCADE,
    predicate   VARCHAR(100) NOT NULL,
    object_id   INT NOT NULL REFERENCES intent_nodes(id) ON DELETE CASCADE,
    weight      FLOAT NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (subject_id, predicate, object_id)
);

CREATE TABLE IF NOT EXISTS intent_aliases (
    id          SERIAL PRIMARY KEY,
    intent_id   INT NOT NULL REFERENCES intent_nodes(id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    alias_type  VARCHAR(50) NOT NULL DEFAULT 'keyword',
    UNIQUE (intent_id, alias)
);

CREATE INDEX IF NOT EXISTS idx_intent_nodes_level   ON intent_nodes (level);
CREATE INDEX IF NOT EXISTS idx_intent_nodes_domain  ON intent_nodes (domain);
CREATE INDEX IF NOT EXISTS idx_intent_triplets_subj ON intent_triplets (subject_id);
CREATE INDEX IF NOT EXISTS idx_intent_triplets_obj  ON intent_triplets (object_id);
CREATE INDEX IF NOT EXISTS idx_intent_triplets_pred ON intent_triplets (predicate);
CREATE INDEX IF NOT EXISTS idx_intent_aliases_iid   ON intent_aliases (intent_id);
"""


def init_intent_db(conn) -> None:
    """Create all intent graph tables and indexes (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_INIT_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def upsert_intent_node(
    conn,
    name: str,
    label: str,
    description: str,
    level: str,
    domain: str,
    lifecycle_stage: Optional[str] = None,
    app_id: str = "base",
    solution_id: Optional[str] = None,
) -> int:
    """Insert or update an intent node. Returns the node id.

    Parameters
    ----------
    app_id      : Application scope (default 'base')
    solution_id : Solution scope within the app (default None = app-wide)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO intent_nodes (name, label, description, level, domain,
                                      lifecycle_stage, app_id, solution_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET label           = EXCLUDED.label,
                  description     = EXCLUDED.description,
                  level           = EXCLUDED.level,
                  domain          = EXCLUDED.domain,
                  lifecycle_stage = EXCLUDED.lifecycle_stage,
                  app_id          = EXCLUDED.app_id,
                  solution_id     = EXCLUDED.solution_id
            RETURNING id
            """,
            (name, label, description, level, domain, lifecycle_stage,
             app_id, solution_id),
        )
        return cur.fetchone()[0]


def add_intent_triplet(
    conn,
    subject_name: str,
    predicate: str,
    object_name: str,
    weight: float = 1.0,
) -> int:
    """Add a typed triplet between two nodes (by name). Returns triplet id."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM intent_nodes WHERE name = %s", (subject_name,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Subject node not found: {subject_name!r}")
        subject_id = row[0]

        cur.execute("SELECT id FROM intent_nodes WHERE name = %s", (object_name,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Object node not found: {object_name!r}")
        object_id = row[0]

        cur.execute(
            """
            INSERT INTO intent_triplets (subject_id, predicate, object_id, weight)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (subject_id, predicate, object_id) DO UPDATE
              SET weight = EXCLUDED.weight
            RETURNING id
            """,
            (subject_id, predicate, object_id, weight),
        )
        return cur.fetchone()[0]


def add_intent_alias(
    conn,
    intent_name: str,
    alias: str,
    alias_type: str = "keyword",
) -> None:
    """Add a keyword / phrase / synonym alias for an intent node."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM intent_nodes WHERE name = %s", (intent_name,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Intent node not found: {intent_name!r}")
        intent_id = row[0]
        cur.execute(
            """
            INSERT INTO intent_aliases (intent_id, alias, alias_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (intent_id, alias) DO NOTHING
            """,
            (intent_id, alias.lower(), alias_type),
        )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_embed_cache: dict = {}


def _get_embed_model():
    """Load BGE-M3 (cached). Returns None if sentence-transformers unavailable."""
    if "model" not in _embed_cache:
        from hooks.embedding_registry import get_embed_model
        _embed_cache["model"] = get_embed_model()
    return _embed_cache["model"]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    if model is None:
        return [[] for _ in texts]
    import numpy as np
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def _node_to_embed_text(row: tuple) -> str:
    """Build rich embedding text from (name, label, description, level, domain, lifecycle_stage)."""
    name, label, description, level, domain, lifecycle_stage = row
    parts = [f"intent: {label}"]
    if description:
        parts.append(description)
    parts.append(f"level: {level}")
    if domain and domain != "any":
        parts.append(f"domain: {domain}")
    if lifecycle_stage:
        parts.append(f"lifecycle: {lifecycle_stage}")
    return ". ".join(parts)


def embed_intent_nodes(conn) -> int:
    """Embed all intent_nodes that have NULL embedding. Returns count embedded."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, label, description, level, domain, lifecycle_stage
            FROM intent_nodes
            WHERE embedding IS NULL
            """
        )
        rows = cur.fetchall()

    if not rows:
        return 0

    ids   = [r[0] for r in rows]
    texts = [_node_to_embed_text(r[1:]) for r in rows]
    embeds = _embed_texts(texts)

    with conn.cursor() as cur:
        for node_id, embedding in zip(ids, embeds):
            cur.execute(
                "UPDATE intent_nodes SET embedding = %s WHERE id = %s",
                (embedding if embedding else None, node_id),
            )
    conn.commit()
    return len(ids)


def backfill_intent_embeddings(conn) -> int:
    """Alias for embed_intent_nodes — embeds any nodes missing embeddings."""
    return embed_intent_nodes(conn)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def find_candidate_intents(
    conn,
    query: str,
    limit: int = 8,
    domain_filter: Optional[str] = None,
    app_id: str = "base",
) -> list[dict]:
    """Embed query and return top-k intent nodes ranked by cosine similarity.

    Falls back to alias keyword search if BGE-M3 is unavailable.

    Parameters
    ----------
    app_id : Application scope — sees own data + 'base' data (default 'base')

    Returns list of dicts: {id, name, label, description, level, domain,
                             lifecycle_stage, score}
    """
    model = _get_embed_model()
    if model is None:
        return _alias_keyword_search(conn, query, limit, domain_filter, app_id)

    try:
        import numpy as np
    except ImportError:
        return _alias_keyword_search(conn, query, limit, domain_filter, app_id)

    q_vec = model.encode([query], normalize_embeddings=True)[0]

    # App-scoped: own + base data
    conditions = ["(app_id = %s OR app_id = 'base')"]
    params: list = [app_id]
    if domain_filter and domain_filter.lower() != "any":
        # Include the requested domain + all domains declared by the app manifest
        domain_set = {domain_filter}
        if app_id != "base":
            try:
                from app_registry import load_app_manifest
                app_manifest = load_app_manifest(app_id)
                domain_set.update(app_manifest.get("domains", []))
            except (ImportError, FileNotFoundError):
                pass
        domain_placeholders = ", ".join(["%s"] * len(domain_set))
        conditions.append(f"(domain IN ({domain_placeholders}) OR domain = 'any')")
        params.extend(sorted(domain_set))
    where = "WHERE " + " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, name, label, description, level, domain, lifecycle_stage, embedding
            FROM intent_nodes
            {where}
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return _alias_keyword_search(conn, query, limit, domain_filter, app_id)

    results = []
    for row in rows:
        node_id, name, label, description, level, domain, lifecycle_stage, embedding = row
        if not embedding:
            continue
        vec = np.array(embedding, dtype=np.float32)
        score = float(np.dot(q_vec, vec))
        results.append({
            "id":              node_id,
            "name":            name,
            "label":           label,
            "description":     description or "",
            "level":           level,
            "domain":          domain,
            "lifecycle_stage": lifecycle_stage or "",
            "score":           score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # Boost alias matches — if the query (lowercased) appears in any alias for a node,
    # add a 0.05 bonus so exact keyword matches aren't buried by semantic near-misses.
    alias_boosts = _get_alias_boosts(conn, query)
    for r in results:
        if r["id"] in alias_boosts:
            r["score"] = min(1.0, r["score"] + 0.05)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:limit]


def _get_alias_boosts(conn, query: str) -> set[int]:
    """Return set of node IDs whose aliases match the query (substring, case-insensitive)."""
    tokens = [t.lower() for t in query.split() if len(t) > 2]
    if not tokens:
        return set()
    with conn.cursor() as cur:
        placeholders = " OR ".join(["alias LIKE %s"] * len(tokens))
        cur.execute(
            f"SELECT DISTINCT intent_id FROM intent_aliases WHERE {placeholders}",
            [f"%{t}%" for t in tokens],
        )
        return {r[0] for r in cur.fetchall()}


def _alias_keyword_search(
    conn,
    query: str,
    limit: int,
    domain_filter: Optional[str],
    app_id: str = "base",
) -> list[dict]:
    """Keyword fallback: search node names, labels, and aliases."""
    tokens = [t.lower() for t in query.split() if len(t) > 2]
    if not tokens:
        return []

    app_clause = "AND (n.app_id = %s OR n.app_id = 'base')"
    domain_clause = ""
    base_params: list = [app_id]
    if domain_filter and domain_filter.lower() != "any":
        domain_clause = "AND (n.domain = %s OR n.domain = 'any')"
        base_params.append(domain_filter)

    like_clauses = " OR ".join(
        ["LOWER(n.name) LIKE %s OR LOWER(n.label) LIKE %s"] * len(tokens)
    )
    like_params = [v for t in tokens for v in (f"%{t}%", f"%{t}%")]

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT n.id, n.name, n.label, n.description,
                            n.level, n.domain, n.lifecycle_stage
            FROM intent_nodes n
            WHERE ({like_clauses}) {app_clause} {domain_clause}
            UNION
            SELECT DISTINCT n.id, n.name, n.label, n.description,
                            n.level, n.domain, n.lifecycle_stage
            FROM intent_nodes n
            JOIN intent_aliases a ON a.intent_id = n.id
            WHERE ({" OR ".join(["a.alias LIKE %s"] * len(tokens))}) {app_clause} {domain_clause}
            LIMIT %s
            """,
            like_params + base_params + [f"%{t}%" for t in tokens] + base_params + [limit],
        )
        rows = cur.fetchall()

    # Score by token overlap ratio instead of flat 1.0
    results = []
    for r in rows:
        node_text = f"{r[1]} {r[2]} {r[3] or ''}".lower()
        matched = sum(1 for t in tokens if t in node_text)
        score = matched / max(len(tokens), 1)
        results.append({
            "id":              r[0],
            "name":            r[1],
            "label":           r[2],
            "description":     r[3] or "",
            "level":           r[4],
            "domain":          r[5],
            "lifecycle_stage": r[6] or "",
            "score":           max(score, 0.1),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def traverse_up(conn, node_id: int, max_depth: int = 4, app_id: str = "base") -> list[dict]:
    """Walk IS_A edges upward from node_id to find all ancestor intents.

    If X IS_A Y, then Y is an ancestor of X.
    Returns list of ancestor dicts ordered by ascending depth (closest first),
    with 'depth' field added.  Scoped to nodes visible to app_id (own + base).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE ancestors AS (
              SELECT t.object_id AS node_id, 1 AS depth
              FROM intent_triplets t
              WHERE t.subject_id = %s AND t.predicate = 'IS_A'

              UNION ALL

              SELECT t.object_id, a.depth + 1
              FROM intent_triplets t
              JOIN ancestors a ON t.subject_id = a.node_id
              WHERE t.predicate = 'IS_A' AND a.depth < %s
            )
            SELECT DISTINCT n.id, n.name, n.label, n.description,
                            n.level, n.domain, n.lifecycle_stage,
                            MIN(a.depth) AS depth
            FROM intent_nodes n
            JOIN ancestors a ON n.id = a.node_id
            WHERE (n.app_id = %s OR n.app_id = 'base')
            GROUP BY n.id, n.name, n.label, n.description, n.level, n.domain, n.lifecycle_stage
            ORDER BY depth ASC
            """,
            (node_id, max_depth, app_id),
        )
        rows = cur.fetchall()

    return [
        {
            "id":              r[0],
            "name":            r[1],
            "label":           r[2],
            "description":     r[3] or "",
            "level":           r[4],
            "domain":          r[5],
            "lifecycle_stage": r[6] or "",
            "depth":           r[7],
        }
        for r in rows
    ]


def traverse_down(conn, node_id: int, max_depth: int = 4, app_id: str = "base") -> list[dict]:
    """Walk IS_A edges downward from node_id to find all descendant intents.

    If X IS_A Y, then X is a descendant of Y.  We traverse in reverse:
    find all nodes whose IS_A chain leads back to node_id.
    Returns list of descendant dicts ordered by ascending depth (immediate children first).
    Scoped to nodes visible to app_id (own + base).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE descendants AS (
              SELECT t.subject_id AS node_id, 1 AS depth
              FROM intent_triplets t
              WHERE t.object_id = %s AND t.predicate = 'IS_A'

              UNION ALL

              SELECT t.subject_id, d.depth + 1
              FROM intent_triplets t
              JOIN descendants d ON t.object_id = d.node_id
              WHERE t.predicate = 'IS_A' AND d.depth < %s
            )
            SELECT DISTINCT n.id, n.name, n.label, n.description,
                            n.level, n.domain, n.lifecycle_stage,
                            MIN(d.depth) AS depth
            FROM intent_nodes n
            JOIN descendants d ON n.id = d.node_id
            WHERE (n.app_id = %s OR n.app_id = 'base')
            GROUP BY n.id, n.name, n.label, n.description, n.level, n.domain, n.lifecycle_stage
            ORDER BY depth ASC
            """,
            (node_id, max_depth, app_id),
        )
        rows = cur.fetchall()

    return [
        {
            "id":              r[0],
            "name":            r[1],
            "label":           r[2],
            "description":     r[3] or "",
            "level":           r[4],
            "domain":          r[5],
            "lifecycle_stage": r[6] or "",
            "depth":           r[7],
        }
        for r in rows
    ]


def _get_related_by_predicate(conn, node_id: int, predicate: str) -> list[dict]:
    """Return nodes connected to node_id via the given predicate (subject → object)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.id, n.name, n.label, n.description, n.level, n.domain, n.lifecycle_stage
            FROM intent_nodes n
            JOIN intent_triplets t ON t.object_id = n.id
            WHERE t.subject_id = %s AND t.predicate = %s
            UNION
            SELECT n.id, n.name, n.label, n.description, n.level, n.domain, n.lifecycle_stage
            FROM intent_nodes n
            JOIN intent_triplets t ON t.subject_id = n.id
            WHERE t.object_id = %s AND t.predicate = %s
            """,
            (node_id, predicate, node_id, predicate),
        )
        rows = cur.fetchall()

    return [
        {
            "id":              r[0],
            "name":            r[1],
            "label":           r[2],
            "description":     r[3] or "",
            "level":           r[4],
            "domain":          r[5],
            "lifecycle_stage": r[6] or "",
        }
        for r in rows
    ]


def _get_siblings(conn, node_id: int, app_id: str = "base") -> list[dict]:
    """Return intent nodes that share a direct IS_A parent with node_id.
    Scoped to nodes visible to app_id (own + base).
    """
    with conn.cursor() as cur:
        # Find direct parents of node_id
        cur.execute(
            "SELECT object_id FROM intent_triplets WHERE subject_id = %s AND predicate = 'IS_A'",
            (node_id,),
        )
        parent_ids = [r[0] for r in cur.fetchall()]

    if not parent_ids:
        return []

    placeholders = ",".join(["%s"] * len(parent_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT n.id, n.name, n.label, n.description, n.level, n.domain, n.lifecycle_stage
            FROM intent_nodes n
            JOIN intent_triplets t ON t.subject_id = n.id
            WHERE t.predicate = 'IS_A'
              AND t.object_id IN ({placeholders})
              AND n.id != %s
              AND (n.app_id = %s OR n.app_id = 'base')
            """,
            parent_ids + [node_id, app_id],
        )
        rows = cur.fetchall()

    return [
        {
            "id":              r[0],
            "name":            r[1],
            "label":           r[2],
            "description":     r[3] or "",
            "level":           r[4],
            "domain":          r[5],
            "lifecycle_stage": r[6] or "",
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Full intent resolution pipeline
# ---------------------------------------------------------------------------

# Default thresholds (used when DB is unavailable or not initialized)
_AMBIGUITY_THRESHOLD    = 0.05   # cosine sim delta within which candidates may be ambiguous
_LOW_CONFIDENCE_FLOOR   = 0.55   # below this score, always flag — match may not exist in graph
_MIN_CONFIDENCE_TO_MATCH = 0.45  # below this, return no match (random noise)


# ---------------------------------------------------------------------------
# WI-1B: Adaptive per-type confidence thresholds (OB1 pattern: adaptive-capture-classification)
# ---------------------------------------------------------------------------
_THRESHOLD_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS intent_thresholds (
    id              SERIAL PRIMARY KEY,
    threshold_name  TEXT UNIQUE NOT NULL,
    value           FLOAT NOT NULL,
    corrections     INT DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

_DEFAULT_THRESHOLDS = {
    "ambiguity_threshold":     0.05,
    "low_confidence_floor":    0.55,
    "min_confidence_to_match": 0.45,
}

_THRESHOLD_FLOOR = 0.01
_THRESHOLD_CEILING = 0.95
_ADJUSTMENT_STEP = 0.005


def init_intent_thresholds(conn) -> None:
    """Create the intent_thresholds table and seed defaults (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_THRESHOLD_TABLE_SQL)
        for name, default_val in _DEFAULT_THRESHOLDS.items():
            cur.execute(
                "INSERT INTO intent_thresholds (threshold_name, value) "
                "VALUES (%s, %s) ON CONFLICT (threshold_name) DO NOTHING",
                (name, default_val),
            )
    conn.commit()


def get_threshold(conn, name: str) -> float:
    """Read an adaptive threshold from DB, falling back to hardcoded default."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM intent_thresholds WHERE threshold_name = %s",
                (name,),
            )
            row = cur.fetchone()
            if row:
                return float(row[0])
    except Exception:
        pass
    return _DEFAULT_THRESHOLDS.get(name, 0.05)


def record_correction(conn, threshold_name: str, was_correct: bool, actual_score: float) -> float:
    """Record a correction and adjust the threshold adaptively.

    If was_correct=False and actual_score was above threshold (false positive):
      raise threshold by _ADJUSTMENT_STEP.
    If was_correct=False and actual_score was below threshold (false negative):
      lower threshold by _ADJUSTMENT_STEP.
    If was_correct=True: no adjustment, just increment counter.

    Returns the new threshold value.
    """
    current = get_threshold(conn, threshold_name)

    if not was_correct:
        if actual_score >= current:
            # False positive: threshold too low, raise it
            new_val = min(_THRESHOLD_CEILING, current + _ADJUSTMENT_STEP)
        else:
            # False negative: threshold too high, lower it
            new_val = max(_THRESHOLD_FLOOR, current - _ADJUSTMENT_STEP)
    else:
        new_val = current  # correct prediction, no adjustment

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE intent_thresholds "
            "SET value = %s, corrections = corrections + 1, updated_at = NOW() "
            "WHERE threshold_name = %s",
            (new_val, threshold_name),
        )
    conn.commit()
    return new_val


def _get_root_ancestor(conn, node_id: int) -> Optional[int]:
    """Return the root ancestor id of a node by walking the IS_A chain to its top.
    Returns node_id itself if the node has no IS_A parents (it is a root).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH RECURSIVE ancestors AS (
              SELECT t.object_id AS node_id
              FROM intent_triplets t
              WHERE t.subject_id = %s AND t.predicate = 'IS_A'

              UNION ALL

              SELECT t.object_id
              FROM intent_triplets t
              JOIN ancestors a ON t.subject_id = a.node_id
              WHERE t.predicate = 'IS_A'
            )
            SELECT node_id FROM ancestors
            WHERE node_id NOT IN (
              SELECT subject_id FROM intent_triplets WHERE predicate = 'IS_A'
            )
            LIMIT 1
            """,
            (node_id,),
        )
        row = cur.fetchone()
    return row[0] if row else node_id


def resolve_intent(
    conn,
    query: str,
    domain: Optional[str] = None,
    app_id: str = "base",
) -> dict:
    """Full intent resolution pipeline.

    1. Embed query → find candidate intent nodes by cosine similarity
    2. Take the top match as matched_node (if above MIN_CONFIDENCE_TO_MATCH)
    3. Check ambiguity — CROSS-BRANCH only:
       a. Candidates within AMBIGUITY_THRESHOLD AND with a different root ancestor
          are genuine ambiguities requiring clarification.
       b. Same-branch candidates (different node, same root) are refinements,
          not ambiguities — they are surfaced as sibling/child context, not HITL triggers.
       c. If top confidence < LOW_CONFIDENCE_FLOOR: always flag as ambiguous
          (the intent likely does not exist in the graph yet).
    4. Traverse upward  → abstract_parents  (latent goals)
    5. Traverse downward → specific_children (execution paths)
    6. Collect siblings  → sibling_intents
    7. Collect cross-cutting relationships (REQUIRES, ENABLES, PRECEDES)
    8. Return structured IntentContext dict

    Returns
    -------
    {
        "matched_node":            dict | None,
        "confidence":              float,
        "abstract_parents":        list[dict],   # ordered closest → most abstract
        "specific_children":       list[dict],   # ordered closest → most specific
        "sibling_intents":         list[dict],
        "same_branch_alternatives":list[dict],   # close candidates in same branch
        "requires_before":         list[dict],   # nodes this intent REQUIRES
        "enables_after":           list[dict],   # nodes this intent ENABLES
        "precedes":                list[dict],   # nodes this intent PRECEDES
        "ambiguity_candidates":    list[dict],   # CROSS-BRANCH candidates within threshold
        "requires_clarification":  bool,
        "clarification_reason":    str | None,   # why clarification is needed
        "graph_used":              bool,
    }
    """
    empty: dict = {
        "matched_node":             None,
        "confidence":               0.0,
        "abstract_parents":         [],
        "specific_children":        [],
        "sibling_intents":          [],
        "same_branch_alternatives": [],
        "requires_before":          [],
        "enables_after":            [],
        "precedes":                 [],
        "ambiguity_candidates":     [],
        "requires_clarification":   False,
        "clarification_reason":     None,
        "graph_used":               False,
    }

    try:
        candidates = find_candidate_intents(conn, query, limit=12, domain_filter=domain,
                                            app_id=app_id)
    except Exception:
        return empty

    if not candidates:
        return empty

    top = candidates[0]
    top_score = top["score"]

    # WI-1B: resolve adaptive thresholds from DB (falls back to module constants)
    min_conf = get_threshold(conn, "min_confidence_to_match")
    ambiguity_thresh = get_threshold(conn, "ambiguity_threshold")
    low_conf_floor = get_threshold(conn, "low_confidence_floor")

    # Too low to be meaningful — graph has no relevant node
    if top_score < min_conf:
        return {**empty, "graph_used": True,
                "requires_clarification": True,
                "clarification_reason": f"No graph node matched with sufficient confidence (best={top_score:.3f}) — intent may not yet be seeded"}

    node_id = top["id"]

    # Get root ancestor of top match for branch comparison
    try:
        top_root = _get_root_ancestor(conn, node_id)
    except Exception:
        top_root = node_id

    # Split candidates into cross-branch (real ambiguity) vs same-branch (refinements)
    cross_branch: list[dict] = []
    same_branch:  list[dict] = []
    for c in candidates[1:]:
        if (top_score - c["score"]) > ambiguity_thresh:
            break  # beyond threshold — stop
        try:
            c_root = _get_root_ancestor(conn, c["id"])
        except Exception:
            c_root = c["id"]
        if c_root != top_root:
            cross_branch.append(c)
        else:
            same_branch.append(c)

    # Determine clarification need and reason
    requires_clarification = False
    clarification_reason: Optional[str] = None

    if top_score < low_conf_floor:
        requires_clarification = True
        clarification_reason = (
            f"Low confidence match ({top_score:.3f} < {low_conf_floor}) — "
            f"intent may not be fully represented in the graph. "
            f"Closest match: {top['label']}. Consider adding a more specific node."
        )
    elif cross_branch:
        requires_clarification = True
        branch_labels = ", ".join(c["label"] for c in cross_branch[:3])
        clarification_reason = (
            f"Query matches intents in different goal branches: "
            f"'{top['label']}' (confidence={top_score:.3f}) vs [{branch_labels}]. "
            f"Ask user to confirm which direction they need."
        )

    # Graph traversal — all best-effort, scoped by app_id
    try:
        parents = traverse_up(conn, node_id, max_depth=4, app_id=app_id)
    except Exception:
        parents = []

    try:
        children = traverse_down(conn, node_id, max_depth=4, app_id=app_id)
    except Exception:
        children = []

    try:
        siblings = _get_siblings(conn, node_id, app_id=app_id)
    except Exception:
        siblings = []

    try:
        requires_before = _get_related_by_predicate(conn, node_id, "REQUIRES")
    except Exception:
        requires_before = []

    try:
        enables_after = _get_related_by_predicate(conn, node_id, "ENABLES")
    except Exception:
        enables_after = []

    try:
        precedes = _get_related_by_predicate(conn, node_id, "PRECEDES")
    except Exception:
        precedes = []

    return {
        "matched_node":             top,
        "confidence":               round(top_score, 4),
        "abstract_parents":         parents,
        "specific_children":        children,
        "sibling_intents":          siblings,
        "same_branch_alternatives": same_branch,
        "requires_before":          requires_before,
        "enables_after":            enables_after,
        "precedes":                 precedes,
        "ambiguity_candidates":     cross_branch,
        "requires_clarification":   requires_clarification,
        "clarification_reason":     clarification_reason,
        "graph_used":               True,
    }


# ---------------------------------------------------------------------------
# WI-1D: Fuzzy entity resolution (OB1 pattern: schema-aware-routing)
# ---------------------------------------------------------------------------

def find_similar_entities(
    conn,
    name: str,
    entity_type: str = "",
    top_k: int = 5,
    min_score: float = 0.3,
) -> list[dict]:
    """Find entities with similar names using BGE-M3 embedding similarity.

    Queries intent_nodes (level='specific') for names semantically similar
    to the input. Used for entity deduplication and disambiguation.

    Returns list of dicts: {id, name, label, score}
    """
    model = _get_embed_model()
    if model is None:
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    q_vec = model.encode([name], normalize_embeddings=True)[0]

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, label, embedding FROM intent_nodes "
            "WHERE level = 'specific' AND embedding IS NOT NULL",
        )
        rows = cur.fetchall()

    results = []
    for node_id, node_name, label, embedding in rows:
        if not embedding:
            continue
        vec = np.array(embedding, dtype=np.float32)
        score = float(np.dot(q_vec, vec))
        if score >= min_score:
            results.append({
                "id": node_id,
                "name": node_name,
                "label": label,
                "score": round(score, 4),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_intent_context(ctx: dict) -> str:
    """Render an IntentContext dict as an ASCII-safe string for LLM prompts."""
    if not ctx.get("graph_used") or ctx.get("matched_node") is None:
        return "[Intent graph: no match found — falling back to keyword-based intent inference]"

    mn = ctx["matched_node"]
    lines = [
        "=== INTENT GRAPH CONTEXT ===",
        f"Matched Intent  : {mn['label']} ({mn['name']})",
        f"Level           : {mn['level']}",
        f"Domain          : {mn['domain']}",
        f"Lifecycle Stage : {mn.get('lifecycle_stage') or 'n/a'}",
        f"Confidence      : {ctx['confidence']:.3f}",
        f"Description     : {mn.get('description') or ''}",
        "",
    ]

    if ctx["abstract_parents"]:
        lines.append("-- Abstract Parents (latent goals, closest first) --")
        for p in ctx["abstract_parents"]:
            lines.append(f"  [depth={p['depth']}] {p['label']} ({p['name']}) — {p.get('description', '')[:80]}")
        lines.append("")

    if ctx["specific_children"]:
        lines.append("-- Specific Children (execution paths, closest first) --")
        for c in ctx["specific_children"][:6]:
            lines.append(f"  [depth={c['depth']}] {c['label']} ({c['name']}) — {c.get('description', '')[:80]}")
        lines.append("")

    if ctx["sibling_intents"]:
        lines.append("-- Sibling Intents (same parent, alternative approaches) --")
        for s in ctx["sibling_intents"][:4]:
            lines.append(f"  {s['label']} ({s['name']})")
        lines.append("")

    if ctx["requires_before"]:
        lines.append("-- Requires (prerequisites) --")
        for r in ctx["requires_before"]:
            lines.append(f"  {r['label']}")
        lines.append("")

    if ctx["enables_after"]:
        lines.append("-- Enables (downstream) --")
        for e in ctx["enables_after"]:
            lines.append(f"  {e['label']}")
        lines.append("")

    if ctx["ambiguity_candidates"]:
        lines.append("-- Ambiguity Candidates (within similarity threshold) --")
        for a in ctx["ambiguity_candidates"]:
            lines.append(f"  {a['label']} ({a['name']}) score={a['score']:.3f}")
        lines.append("")
        lines.append("! requires_clarification: true — consider HITL to confirm intent")
        lines.append("")

    lines.append("=== END INTENT CONTEXT ===")

    raw = "\n".join(lines)
    return raw.encode("ascii", errors="replace").decode("ascii")
