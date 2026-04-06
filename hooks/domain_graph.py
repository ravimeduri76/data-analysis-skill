"""
domain_graph.py — Domain Knowledge Graph: storage and vector traversal.

Stores structured domain models (entities, ontology triplets, terminology,
invariants, properties) in a dedicated PostgreSQL database and enables
semantic graph traversal using BGE-M3 embeddings — same approach as
semantic_store.py (float4[] columns, cosine similarity in numpy, no pgvector
required).

Public API
----------
get_domain_conn()                                                    -> connection  (domain DB)
init_domain_db(conn)                                                 -> None
save_domain_model(conn, model_json, app_id, solution_id)             -> int         (model_id)
traverse_graph(conn, domain, query, top_k, node_types, app_id)      -> list[dict]
get_domain_context(conn, domain, query, top_k, app_id)              -> dict
render_domain_context(ctx)                                           -> str         (ASCII-safe)

Database
--------
PostgreSQL database : domain
DOMAIN_DB_URL env var (default: postgresql://postgres:memorystorepg@localhost:5432/domain)
Reads from C:\\Users\\Ravi\\Data_Analysis\\.env if present.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

# ── domain name aliases ───────────────────────────────────────────────────────
# Maps detector-emitted domain names to the canonical name stored in domain_models.
# Add entries here whenever the detector and the seeded model use different names.

_DOMAIN_ALIASES: dict[str, str] = {
    "Electrical Components":                "Industrial Component",
    "Industrial Equipment":                 "Industrial Component",
    "Mechanical Components":                "Industrial Component",
    "Automotive Aftermarket / Heavy Truck": "Industrial Component",  # TODO: seed dedicated Truck Parts domain model
    "Wellness":                             "Healthcare",
}

# ── connection ────────────────────────────────────────────────────────────────

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


def get_domain_conn():
    """Return a psycopg2 connection to the domain knowledge graph database.

    Raises psycopg2.OperationalError on connection failure (logged as warning).
    """
    _load_env()
    url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
    import psycopg2
    try:
        return psycopg2.connect(url)
    except psycopg2.OperationalError as exc:
        log.warning(
            "Domain graph DB connection failed (graph_available will be false): %s",
            exc,
        )
        raise


# ── schema ────────────────────────────────────────────────────────────────────

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS domain_models (
    id                 SERIAL PRIMARY KEY,
    domain             TEXT NOT NULL,
    sub_domain         TEXT,
    category           TEXT,
    domain_summary     TEXT,
    sub_domain_summary TEXT,
    functional_purpose TEXT,
    system_boundaries  TEXT,
    namespace_prefix   TEXT DEFAULT 'fso',
    created_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS domain_entities (
    id             SERIAL PRIMARY KEY,
    model_id       INT REFERENCES domain_models(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    entity_group   TEXT,   -- primary | supporting_infrastructure | control_and_monitoring | safety_and_protection | governance
    physical_logical TEXT, -- physical | logical | standard
    description    TEXT
);

CREATE TABLE IF NOT EXISTS domain_properties (
    id             SERIAL PRIMARY KEY,
    entity_id      INT REFERENCES domain_entities(id) ON DELETE CASCADE,
    property_group TEXT,
    property_name  TEXT
);

CREATE TABLE IF NOT EXISTS domain_triplets (
    id         SERIAL PRIMARY KEY,
    model_id   INT REFERENCES domain_models(id) ON DELETE CASCADE,
    subject    TEXT NOT NULL,
    predicate  TEXT NOT NULL,
    object     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS domain_terminology (
    id             SERIAL PRIMARY KEY,
    model_id       INT REFERENCES domain_models(id) ON DELETE CASCADE,
    term           TEXT NOT NULL,
    definition     TEXT,
    functional_role TEXT,
    related_terms  TEXT[]
);

CREATE TABLE IF NOT EXISTS domain_invariants (
    id           SERIAL PRIMARY KEY,
    model_id     INT REFERENCES domain_models(id) ON DELETE CASCADE,
    invariant_id TEXT,
    rule         TEXT,
    description  TEXT
);

CREATE TABLE IF NOT EXISTS domain_node_embeddings (
    id         SERIAL PRIMARY KEY,
    model_id   INT REFERENCES domain_models(id) ON DELETE CASCADE,
    node_type  TEXT NOT NULL,   -- entity | triplet | term | property | invariant
    node_key   TEXT NOT NULL,
    node_text  TEXT NOT NULL,
    embedding  FLOAT4[]
);

CREATE INDEX IF NOT EXISTS idx_domain_models_domain  ON domain_models (domain);
CREATE INDEX IF NOT EXISTS idx_domain_entities_model ON domain_entities (model_id);
CREATE INDEX IF NOT EXISTS idx_domain_triplets_model ON domain_triplets (model_id);
CREATE INDEX IF NOT EXISTS idx_domain_embeddings_model ON domain_node_embeddings (model_id);
CREATE INDEX IF NOT EXISTS idx_domain_embeddings_type  ON domain_node_embeddings (node_type);
"""


def init_domain_db(conn) -> None:
    """Create all domain graph tables (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_INIT_SQL)
    conn.commit()


# ── save domain model ─────────────────────────────────────────────────────────

def save_domain_model(
    conn,
    model_json: dict,
    app_id: str = "base",
    solution_id: Optional[str] = None,
) -> int:
    """Persist a domain model JSON (matching the generator prompt output format).

    Returns the new model_id.  Inserts entities, properties, triplets,
    terminology, and invariants in one transaction, then embeds all nodes.

    Parameters
    ----------
    app_id      : Application scope (default 'base')
    solution_id : Solution scope within the app (default None = app-wide)
    """
    overview = model_json.get("domain_overview", {})
    entities_section = model_json.get("entities", {})
    property_model   = model_json.get("property_model", {})
    ontology         = model_json.get("ontology", {})
    terminology_list = model_json.get("terminology", [])
    invariants_list  = model_json.get("invariants", [])

    with conn.cursor() as cur:
        # 1. domain_models row
        cur.execute(
            """
            INSERT INTO domain_models
                (domain, sub_domain, category, domain_summary, sub_domain_summary,
                 functional_purpose, system_boundaries, namespace_prefix,
                 app_id, solution_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                overview.get("domain", ""),
                overview.get("sub_domain", ""),
                overview.get("category", ""),
                overview.get("domain_summary", ""),
                overview.get("sub_domain_summary", ""),
                overview.get("functional_purpose", ""),
                overview.get("system_boundaries", ""),
                ontology.get("namespace_prefix", "fso"),
                app_id,
                solution_id,
            ),
        )
        model_id = cur.fetchone()[0]

        # 2. entities
        entity_id_map: dict[str, int] = {}
        group_map = {
            "primary":                    "primary",
            "supporting_infrastructure":  "supporting_infrastructure",
            "control_and_monitoring":     "control_and_monitoring",
            "safety_and_protection":      "safety_and_protection",
            "governance":                 "governance",
        }
        for group_key, entities in entities_section.items():
            if not isinstance(entities, list):
                continue
            group_label = group_map.get(group_key, group_key)
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                cur.execute(
                    """
                    INSERT INTO domain_entities
                        (model_id, name, entity_group, physical_logical, description)
                    VALUES (%s,%s,%s,%s,%s)
                    RETURNING id
                    """,
                    (
                        model_id,
                        ent.get("name", ""),
                        group_label,
                        ent.get("type", ""),
                        ent.get("description", ""),
                    ),
                )
                entity_id = cur.fetchone()[0]
                entity_id_map[ent.get("name", "")] = entity_id

                # 3. properties for this entity
                props = property_model.get(ent.get("name", ""), {})
                for prop_group, prop_list in props.items():
                    if not isinstance(prop_list, list):
                        continue
                    for prop_name in prop_list:
                        cur.execute(
                            "INSERT INTO domain_properties (entity_id, property_group, property_name) VALUES (%s,%s,%s)",
                            (entity_id, prop_group, prop_name),
                        )

        # 4. ontology triplets
        for triplet in ontology.get("triplets", []):
            if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                cur.execute(
                    "INSERT INTO domain_triplets (model_id, subject, predicate, object) VALUES (%s,%s,%s,%s)",
                    (model_id, triplet[0], triplet[1], triplet[2]),
                )

        # 5. terminology
        for term_entry in terminology_list:
            if not isinstance(term_entry, dict):
                continue
            cur.execute(
                """
                INSERT INTO domain_terminology
                    (model_id, term, definition, functional_role, related_terms)
                VALUES (%s,%s,%s,%s,%s)
                """,
                (
                    model_id,
                    term_entry.get("term", ""),
                    term_entry.get("definition", ""),
                    term_entry.get("functional_role", ""),
                    term_entry.get("related_terms", []),
                ),
            )

        # 6. invariants
        for inv in invariants_list:
            if not isinstance(inv, dict):
                continue
            cur.execute(
                "INSERT INTO domain_invariants (model_id, invariant_id, rule, description) VALUES (%s,%s,%s,%s)",
                (model_id, inv.get("id", ""), inv.get("rule", ""), inv.get("description", "")),
            )

    conn.commit()

    # 7. embed all nodes (best-effort — never raises)
    try:
        _embed_model_nodes(conn, model_id, model_json)
    except Exception:
        pass

    return model_id


# ── embedding helpers ─────────────────────────────────────────────────────────

_embed_model_cache: dict = {}


def _get_embed_model():
    """Load BGE-M3 model (cached). Returns None if sentence-transformers unavailable."""
    if "model" not in _embed_model_cache:
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model_cache["model"] = SentenceTransformer("BAAI/bge-m3")
        except Exception:
            _embed_model_cache["model"] = None
    return _embed_model_cache["model"]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    if model is None:
        return [[] for _ in texts]
    vecs = model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def _build_node_texts(model_json: dict) -> list[tuple[str, str, str]]:
    """Build (node_type, node_key, node_text) tuples for every node in a model."""
    nodes: list[tuple[str, str, str]] = []
    overview = model_json.get("domain_overview", {})
    domain = overview.get("domain", "")

    entities_section = model_json.get("entities", {})
    property_model   = model_json.get("property_model", {})
    ontology         = model_json.get("ontology", {})
    terminology_list = model_json.get("terminology", [])
    invariants_list  = model_json.get("invariants", [])

    # Entity nodes
    for group_key, ents in entities_section.items():
        if not isinstance(ents, list):
            continue
        for ent in ents:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name", "")
            desc = ent.get("description", "")
            props = property_model.get(name, {})
            prop_summary = "; ".join(
                f"{g}: {', '.join(ps)}" for g, ps in props.items() if isinstance(ps, list)
            )
            text = f"entity: {name} ({group_key}) — {desc}"
            if prop_summary:
                text += f". Properties: {prop_summary}"
            nodes.append(("entity", f"{domain}::{name}", text))

    # Ontology triplet nodes
    for i, triplet in enumerate(ontology.get("triplets", [])):
        if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
            s, p, o = triplet
            text = f"relationship: {s} {p} {o}"
            nodes.append(("triplet", f"{domain}::triplet::{i}", text))

    # Terminology nodes
    for term_entry in terminology_list:
        if not isinstance(term_entry, dict):
            continue
        term = term_entry.get("term", "")
        defn = term_entry.get("definition", "")
        role = term_entry.get("functional_role", "")
        related = ", ".join(term_entry.get("related_terms", []))
        text = f"term: {term} — {defn}. Functional role: {role}."
        if related:
            text += f" Related: {related}."
        nodes.append(("term", f"{domain}::term::{term}", text))

    # Invariant nodes
    for inv in invariants_list:
        if not isinstance(inv, dict):
            continue
        inv_id = inv.get("id", "")
        rule   = inv.get("rule", "")
        desc   = inv.get("description", "")
        text   = f"invariant {inv_id}: {rule} — {desc}"
        nodes.append(("invariant", f"{domain}::inv::{inv_id}", text))

    return nodes


def _embed_model_nodes(conn, model_id: int, model_json: dict) -> int:
    """Generate and store embeddings for all nodes in a domain model.

    Returns count of nodes embedded.
    """
    nodes = _build_node_texts(model_json)
    if not nodes:
        return 0

    texts  = [t for _, _, t in nodes]
    embeds = _embed_texts(texts)

    with conn.cursor() as cur:
        for (node_type, node_key, node_text), embedding in zip(nodes, embeds):
            cur.execute(
                """
                INSERT INTO domain_node_embeddings
                    (model_id, node_type, node_key, node_text, embedding)
                VALUES (%s,%s,%s,%s,%s)
                """,
                (model_id, node_type, node_key, node_text,
                 embedding if embedding else None),
            )
    conn.commit()
    return len(nodes)


# ── traversal ─────────────────────────────────────────────────────────────────

def _resolve_domain(domain: str) -> str:
    """Resolve a detector-emitted domain name to the canonical stored name.

    Resolution order:
    1. Exact dict lookup (fast path)
    2. Semantic alias matching via domain_alias_embeddings (if score >= 0.80)
    3. Passthrough (return as-is)
    """
    # Fast path: exact match
    if domain in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[domain]

    # Semantic alias resolution
    try:
        from hooks.embedding_registry import embed_texts, cosine_rank
        import psycopg2

        conn = _get_conn_safe()
        if conn is not None:
            try:
                q_vec = embed_texts([domain])[0]
                with conn.cursor() as cur:
                    cur.execute("SELECT alias_text, canonical, embedding FROM domain_alias_embeddings")
                    rows = cur.fetchall()
                if rows:
                    candidates = [(i, row[2]) for i, row in enumerate(rows)]
                    ranked = cosine_rank(q_vec, candidates)
                    if ranked and ranked[0][1] >= 0.80:
                        return rows[ranked[0][0]][1]
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        pass

    return domain


def _get_conn_safe():
    """Return DB connection or None (no crash if DB down)."""
    try:
        import psycopg2
        _load_env()
        url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
        return psycopg2.connect(url)
    except Exception:
        return None


def traverse_graph(
    conn,
    domain: str,
    query: str,
    top_k: int = 10,
    node_types: Optional[list[str]] = None,
    app_id: str = "base",
) -> list[dict]:
    """Semantic traversal of the domain graph.

    Embeds `query` with BGE-M3, computes cosine similarity against all stored
    node embeddings for `domain`, returns top_k results ranked by score.

    Parameters
    ----------
    conn        : psycopg2 connection to the domain DB
    domain      : domain name (matches domain_models.domain)
    query       : natural language query or keywords
    top_k       : number of results to return
    node_types  : optional filter, e.g. ['entity', 'term'] — None returns all
    app_id      : Application scope — sees own data + 'base' data (default 'base')

    Returns
    -------
    list of dicts: {node_type, node_key, node_text, score}
    """
    domain = _resolve_domain(domain)

    try:
        import numpy as np
    except ImportError:
        return []

    model = _get_embed_model()
    if model is None:
        return _keyword_traverse(conn, domain, query, top_k, node_types, app_id)

    # Embed query
    q_vec = model.encode([query], normalize_embeddings=True)[0]

    # Fetch all embeddings for this domain (app-scoped: own + base)
    type_filter = ""
    params: list = [domain, app_id]
    if node_types:
        placeholders = ",".join(["%s"] * len(node_types))
        type_filter = f"AND ne.node_type IN ({placeholders})"
        params.extend(node_types)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ne.node_type, ne.node_key, ne.node_text, ne.embedding
            FROM domain_node_embeddings ne
            JOIN domain_models dm ON ne.model_id = dm.id
            WHERE dm.domain = %s
            AND (dm.app_id = %s OR dm.app_id = 'base')
            {type_filter}
            AND ne.embedding IS NOT NULL
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return _keyword_traverse(conn, domain, query, top_k, node_types, app_id)

    results = []
    for node_type, node_key, node_text, embedding in rows:
        if not embedding:
            continue
        vec = np.array(embedding, dtype=np.float32)
        score = float(np.dot(q_vec, vec))
        results.append({
            "node_type": node_type,
            "node_key":  node_key,
            "node_text": node_text,
            "score":     score,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def _keyword_traverse(
    conn,
    domain: str,
    query: str,
    top_k: int,
    node_types: Optional[list[str]],
    app_id: str = "base",
) -> list[dict]:
    """Fallback: keyword search when BGE-M3 is unavailable, with token overlap scoring."""
    query_tokens = [t.lower() for t in re.split(r"[\s,._\-/]+", query) if len(t) > 1]
    if not query_tokens:
        return []

    # Use multiple LIKE clauses (OR'd) for better recall
    like_clauses = " OR ".join(["LOWER(ne.node_text) LIKE %s"] * len(query_tokens))
    type_filter = ""
    params: list = [domain, app_id] + [f"%{t}%" for t in query_tokens]
    if node_types:
        placeholders = ",".join(["%s"] * len(node_types))
        type_filter = f"AND ne.node_type IN ({placeholders})"
        params.extend(node_types)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT ne.node_type, ne.node_key, ne.node_text
            FROM domain_node_embeddings ne
            JOIN domain_models dm ON ne.model_id = dm.id
            WHERE dm.domain = %s
            AND (dm.app_id = %s OR dm.app_id = 'base')
            AND ({like_clauses})
            {type_filter}
            LIMIT %s
            """,
            params + [top_k * 3],  # fetch more, then rank
        )
        rows = cur.fetchall()

    # Score by token overlap ratio
    results = []
    for r in rows:
        text_lower = r[2].lower()
        matched = sum(1 for t in query_tokens if t in text_lower)
        score = matched / max(len(query_tokens), 1)
        results.append({"node_type": r[0], "node_key": r[1], "node_text": r[2], "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── structured context retrieval ──────────────────────────────────────────────

def get_domain_context(
    conn,
    domain: str,
    query: str = "",
    top_k: int = 12,
    app_id: str = "base",
) -> dict:
    """Return a structured domain context dict for grounding DESO reasoning.

    Combines:
    - Domain model overview (functional purpose, system boundaries)
    - Top-k semantically relevant nodes (entities, triplets, terms)
    - All invariants for the domain (always included — they are rules)

    Parameters
    ----------
    app_id : Application scope — sees own data + 'base' data (default 'base')

    Returns a dict suitable for passing to render_domain_context().
    Domain aliases are resolved automatically (e.g. 'Electrical Components' -> 'Industrial Component').
    """
    domain = _resolve_domain(domain)

    # Get model overview (app-scoped: own + base)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT domain, sub_domain, category, domain_summary,
                   sub_domain_summary, functional_purpose, system_boundaries
            FROM domain_models
            WHERE domain = %s
            AND (app_id = %s OR app_id = 'base')
            ORDER BY id DESC LIMIT 1
            """,
            (domain, app_id),
        )
        row = cur.fetchone()

    if not row:
        return {"domain": domain, "found": False, "nodes": []}

    overview = {
        "domain":               row[0],
        "sub_domain":           row[1],
        "category":             row[2],
        "domain_summary":       row[3],
        "sub_domain_summary":   row[4],
        "functional_purpose":   row[5],
        "system_boundaries":    row[6],
    }

    # Semantic traversal — entities, triplets, terms
    nodes = traverse_graph(conn, domain, query or domain, top_k=top_k,
                           node_types=["entity", "triplet", "term"],
                           app_id=app_id) if query or domain else []

    # Always load all invariants (app-scoped: own + base)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT di.invariant_id, di.rule, di.description
            FROM domain_invariants di
            JOIN domain_models dm ON di.model_id = dm.id
            WHERE dm.domain = %s
            AND (dm.app_id = %s OR dm.app_id = 'base')
            ORDER BY di.invariant_id
            """,
            (domain, app_id),
        )
        invariants = [
            {"id": r[0], "rule": r[1], "description": r[2]}
            for r in cur.fetchall()
        ]

    return {
        "domain":    domain,
        "found":     True,
        "overview":  overview,
        "nodes":     nodes,
        "invariants": invariants,
        "query":     query,
    }


# ── rendering ─────────────────────────────────────────────────────────────────

def render_domain_context(ctx: dict) -> str:
    """Render a domain context dict as an ASCII-safe string for LLM prompts."""
    if not ctx.get("found"):
        return f"[Domain graph: no model found for '{ctx.get('domain', '?')}']"

    ov = ctx.get("overview", {})
    lines = [
        "=== DOMAIN GRAPH CONTEXT ===",
        f"Domain          : {ov.get('domain', '')}",
        f"Sub-Domain      : {ov.get('sub_domain', '')}",
        f"Category        : {ov.get('category', '')}",
        f"Purpose         : {ov.get('functional_purpose', '')}",
        f"System Boundaries: {ov.get('system_boundaries', '')}",
        "",
    ]

    # Group retrieved nodes by type
    by_type: dict[str, list[dict]] = {}
    for node in ctx.get("nodes", []):
        by_type.setdefault(node["node_type"], []).append(node)

    if by_type.get("entity"):
        lines.append("-- Relevant Entities --")
        for n in by_type["entity"]:
            lines.append(f"  {n['node_text']}")
        lines.append("")

    if by_type.get("triplet"):
        lines.append("-- Relevant Relationships --")
        for n in by_type["triplet"]:
            lines.append(f"  {n['node_text']}")
        lines.append("")

    if by_type.get("term"):
        lines.append("-- Relevant Terminology --")
        for n in by_type["term"]:
            lines.append(f"  {n['node_text']}")
        lines.append("")

    if ctx.get("invariants"):
        lines.append("-- Domain Invariants (always true) --")
        for inv in ctx["invariants"]:
            lines.append(f"  [{inv['id']}] {inv['description']}")
        lines.append("")

    lines.append("=== END DOMAIN GRAPH ===")

    raw = "\n".join(lines)
    return raw.encode("ascii", errors="replace").decode("ascii")


# ── utility: load model from JSON file ────────────────────────────────────────

def load_model_from_file(
    conn,
    json_path: str,
    app_id: str = "base",
    solution_id: Optional[str] = None,
) -> int:
    """Load a domain model JSON file and persist it. Returns model_id."""
    with open(json_path, "r", encoding="utf-8") as f:
        model_json = json.load(f)
    init_domain_db(conn)
    return save_domain_model(conn, model_json, app_id=app_id, solution_id=solution_id)


# ── backfill: embed any nodes missing embeddings ──────────────────────────────

def backfill_node_embeddings(conn) -> int:
    """Embed domain_node_embeddings rows that have NULL embedding. Returns count."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, node_text FROM domain_node_embeddings WHERE embedding IS NULL"
        )
        rows = cur.fetchall()

    if not rows:
        return 0

    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    embeds = _embed_texts(texts)

    with conn.cursor() as cur:
        for row_id, embedding in zip(ids, embeds):
            cur.execute(
                "UPDATE domain_node_embeddings SET embedding = %s WHERE id = %s",
                (embedding if embedding else None, row_id),
            )
    conn.commit()
    return len(ids)
