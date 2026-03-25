"""
semantic_store.py -- BGE-M3 semantic embedding layer for the data_analysis memory store.

Extends db_store with vector embeddings stored as PostgreSQL float4[] arrays.
Cosine similarity is computed in Python (numpy) — no pgvector extension required.

Public API
----------
get_model() -> SentenceTransformer                      # cached singleton
build_run_text(run_row, steps, insights) -> str         # compose rich text for embedding
generate_metadata(run_row, steps, insights,
                  conn_string=None) -> dict             # LLM-enhanced metadata (tags, intent, etc.)
save_embedding(conn, run_id, text, metadata) -> None    # embed + store
backfill_embeddings(conn) -> int                        # embed all unembedded runs
recall_semantic(conn, query, limit=5,
                domain_filter=None) -> list[dict]       # cosine-similarity search
render_semantic_memory(results,
                       current_query="") -> str         # ASCII-safe console block

Schema additions (created by init_semantic_db, called automatically)
----------------------------------------------------------------------
  run_embeddings  — one row per run: embedding float4[], metadata JSONB
  embedding_tags  — denormalized tag rows for keyword search
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from typing import List, Optional

import numpy as np

_MODEL = None
_MODEL_NAME = "BAAI/bge-m3"
_EMBED_DIM = 1024


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

def get_model():
    """Return the BGE-M3 SentenceTransformer (loaded once, cached globally)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install with: pip install sentence-transformers"
            ) from exc
        import warnings, os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _embed(texts: List[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, 1024)."""
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SEMANTIC_DDL = """
CREATE TABLE IF NOT EXISTS run_embeddings (
    id          SERIAL PRIMARY KEY,
    run_id      TEXT UNIQUE NOT NULL REFERENCES analysis_runs(run_id),
    embed_text  TEXT NOT NULL,
    embedding   float4[] NOT NULL,
    domain      TEXT,
    intent      TEXT,
    data_description TEXT,
    tags        TEXT[],
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embedding_tags (
    id      SERIAL PRIMARY KEY,
    run_id  TEXT NOT NULL REFERENCES analysis_runs(run_id),
    tag     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_emb_run     ON run_embeddings(run_id);
CREATE INDEX IF NOT EXISTS idx_emb_domain  ON run_embeddings(domain);
CREATE INDEX IF NOT EXISTS idx_embtag_tag  ON embedding_tags(tag);
CREATE INDEX IF NOT EXISTS idx_embtag_run  ON embedding_tags(run_id);
"""


def init_semantic_db(conn) -> None:
    """Create semantic tables if they do not exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_SEMANTIC_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Rich text composition
# ---------------------------------------------------------------------------

def build_run_text(run_row: dict, steps: list, insights: list) -> str:
    """
    Compose a dense, semantically rich text document from a run's metadata
    for embedding. Includes domain, query, findings, step descriptions,
    and insight findings.
    """
    parts = []

    domain = run_row.get("domain") or "Unknown"
    analysis_type = run_row.get("analysis_type") or ""
    query = run_row.get("query") or ""
    findings = run_row.get("findings_summary") or ""
    workflows = ", ".join(run_row.get("workflows_run") or [])
    run_id = run_row.get("run_id") or ""
    dataset_name = run_row.get("dataset_name") or run_row.get("file_name") or ""

    parts.append(f"Domain: {domain}")
    parts.append(f"Analysis type: {analysis_type}")
    parts.append(f"Workflows: {workflows}")
    if dataset_name:
        parts.append(f"Dataset: {dataset_name}")
    parts.append(f"Query: {query}")
    parts.append(f"Key findings: {findings}")

    for s in steps:
        desc = s.get("description") or s.get("step_name") or ""
        result = s.get("result_summary") or ""
        if desc or result:
            parts.append(f"Step: {desc}. Result: {result}")

    for ins in insights:
        finding = ins.get("finding") or ""
        method = ins.get("method") or ""
        if finding:
            parts.append(f"Insight ({method}): {finding}")

    return " | ".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

_INTENT_PATTERNS = [
    (r"\b(predict|classif|random forest|logistic|model|ml|machine learning)\b", "predictive_modelling"),
    (r"\b(surviv|churn|time.to.event|kaplan|cox|hazard)\b", "survival_analysis"),
    (r"\b(anomaly|outlier|flag|wrong|bad.value|detect)\b", "anomaly_detection"),
    (r"\b(correlat|relationship|between)\b", "correlation_analysis"),
    (r"\b(sampl|comparison|diversit|scorecard)\b", "sampling_comparison"),
    (r"\b(profile|quality|overview|describe|summary)\b", "data_profiling"),
    (r"\b(aggregat|group.by|average|mean|total|sum|count.by)\b", "aggregation"),
    (r"\b(trend|time.series|over.time|forecast)\b", "time_series"),
]


def _infer_intent(query: str, analysis_type: str) -> str:
    text = (query + " " + analysis_type).lower()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, text):
            return intent
    return "general_analysis"


_DOMAIN_TAG_MAP = {
    "Healthcare": ["healthcare", "clinical", "patient", "medical", "ehr"],
    "Healthcare Supply Chain / GPO": ["gpo", "healthcare_supply_chain", "member_roster", "class_of_trade", "contract"],
    "Pharmaceutical": ["pharmaceutical", "drug", "clinical_trial", "regulatory"],
    "Financial Services / Lending": ["finance", "lending", "credit", "loan", "banking"],
    "Insurance": ["insurance", "claims", "underwriting", "policy"],
    "Retail & E-commerce": ["retail", "ecommerce", "product_catalogue", "sales"],
    "HR & People Analytics": ["hr", "people_analytics", "attrition", "workforce"],
    "Industrial Equipment": ["industrial", "equipment", "manufacturing", "maintenance"],
    "Electrical Components": ["electrical", "components", "product_data", "pim"],
    "Master Data Management": ["mdm", "master_data", "entity_resolution", "deduplication"],
    "Education": ["education", "student", "academic", "learning"],
    "General": ["general", "tabular_data"],
}


def _autogenerate_tags(
    domain: str,
    intent: str,
    query: str,
    findings: str,
    column_names: list,
    analysis_type: str,
) -> List[str]:
    """
    Generate keyword tags from structured metadata without requiring an LLM.
    Returns a deduplicated, lowercase list.
    """
    tags = set()

    # Domain tags
    for d, dtags in _DOMAIN_TAG_MAP.items():
        if domain and d.lower() in domain.lower():
            tags.update(dtags)

    # Intent tag
    if intent:
        tags.add(intent)

    # Analysis type
    if analysis_type:
        tags.add(analysis_type.lower().replace(" ", "_"))

    # Extract meaningful keywords from query (skip stopwords)
    _STOPWORDS = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "are", "was", "be", "by", "with", "this", "that", "can", "you",
        "me", "my", "from", "it", "as", "if", "do", "run", "give", "show",
        "which", "what", "how", "please", "i", "have", "has", "we", "based",
        "csv", "file", "dataset", "data", "column", "columns", "rows", "row",
    }
    words = re.findall(r"[a-z]{3,}", (query or "").lower())
    tags.update(w for w in words if w not in _STOPWORDS)

    # High-signal column name keywords
    _COL_STOPWORDS = {"id", "name", "date", "type", "code", "no", "num", "flag"}
    for col in column_names or []:
        parts = re.split(r"[_\s]+", col.lower())
        tags.update(p for p in parts if len(p) > 2 and p not in _COL_STOPWORDS)

    # Findings keywords
    finding_words = re.findall(r"[a-z]{4,}", (findings or "").lower())
    _FINDING_STOPWORDS = {"that", "with", "this", "from", "have", "been", "were", "more",
                          "than", "each", "some", "most", "also", "both", "only", "across"}
    tags.update(w for w in finding_words if w not in _FINDING_STOPWORDS)

    return sorted(tags)[:40]  # cap at 40 tags


def generate_metadata(
    run_row: dict,
    steps: list,
    insights: list,
    domain_ctx: Optional[dict] = None,
) -> dict:
    """
    Build structured metadata for a run: domain, intent, data_description, tags.
    Fully deterministic — no LLM required.
    """
    domain = run_row.get("domain") or "Unknown"
    query = run_row.get("query") or ""
    findings = run_row.get("findings_summary") or ""
    analysis_type = run_row.get("analysis_type") or ""
    workflows = run_row.get("workflows_run") or []
    dataset_name = run_row.get("dataset_name") or run_row.get("file_name") or ""
    row_count = run_row.get("row_count") or 0
    col_count = run_row.get("col_count") or 0

    # Column names from steps if available
    column_names = []
    for s in steps:
        params = s.get("params") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        if "column_names" in params:
            column_names = params["column_names"]
            break

    intent = _infer_intent(query, analysis_type)

    # Data description
    data_desc_parts = []
    if dataset_name:
        data_desc_parts.append(f"Dataset: {dataset_name}")
    if row_count:
        data_desc_parts.append(f"{row_count:,} rows")
    if col_count:
        data_desc_parts.append(f"{col_count} columns")
    if domain and domain != "Unknown":
        data_desc_parts.append(f"Domain: {domain}")
    if analysis_type:
        data_desc_parts.append(f"Analysis: {analysis_type}")
    if findings:
        data_desc_parts.append(f"Key findings: {findings}")
    data_description = ". ".join(data_desc_parts) if data_desc_parts else "Tabular data analysis run."

    tags = _autogenerate_tags(domain, intent, query, findings, column_names, analysis_type)

    # Domain context enrichment
    domain_summary = ""
    if domain_ctx and isinstance(domain_ctx, dict):
        domain_summary = domain_ctx.get("summary") or domain_ctx.get("analysis_lens") or ""

    return {
        "domain": domain,
        "intent": intent,
        "data_description": data_description,
        "tags": tags,
        "domain_summary": domain_summary,
        "analysis_type": analysis_type,
        "workflows": workflows,
        "dataset_name": dataset_name,
        "row_count": row_count,
        "col_count": col_count,
        "step_count": len(steps),
        "insight_count": len(insights),
    }


# ---------------------------------------------------------------------------
# Save embedding
# ---------------------------------------------------------------------------

def save_embedding(
    conn,
    run_id: str,
    embed_text: str,
    metadata: dict,
) -> None:
    """
    Embed *embed_text* using BGE-M3 and store in run_embeddings.
    Also inserts rows into embedding_tags for each tag.
    Idempotent — updates existing row if run_id already exists.
    """
    init_semantic_db(conn)

    vec = _embed([embed_text])[0]  # shape (1024,)
    vec_list = vec.tolist()

    tags = metadata.get("tags") or []

    with conn.cursor() as cur:
        # Upsert embedding row
        cur.execute(
            """
            INSERT INTO run_embeddings
                (run_id, embed_text, embedding, domain, intent,
                 data_description, tags, metadata)
            VALUES (%s, %s, %s::float4[], %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE
                SET embed_text       = EXCLUDED.embed_text,
                    embedding        = EXCLUDED.embedding,
                    domain           = EXCLUDED.domain,
                    intent           = EXCLUDED.intent,
                    data_description = EXCLUDED.data_description,
                    tags             = EXCLUDED.tags,
                    metadata         = EXCLUDED.metadata,
                    created_at       = NOW()
            """,
            (
                run_id,
                embed_text,
                vec_list,
                metadata.get("domain"),
                metadata.get("intent"),
                metadata.get("data_description"),
                tags,
                json.dumps(metadata, default=str),
            ),
        )
        # Refresh tag rows
        cur.execute("DELETE FROM embedding_tags WHERE run_id = %s", (run_id,))
        if tags:
            cur.executemany(
                "INSERT INTO embedding_tags (run_id, tag) VALUES (%s, %s)",
                [(run_id, t) for t in tags],
            )

    conn.commit()


# ---------------------------------------------------------------------------
# Backfill existing runs
# ---------------------------------------------------------------------------

def backfill_embeddings(conn) -> int:
    """
    Embed all analysis_runs rows that do not yet have a run_embeddings entry.
    Returns the number of runs newly embedded.
    """
    init_semantic_db(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.run_id, r.domain, r.analysis_type, r.query,
                   r.findings_summary, r.workflows_run,
                   d.file_name,
                   r.domain_ctx
            FROM analysis_runs r
            LEFT JOIN datasets d ON d.id = r.dataset_id
            WHERE r.run_id NOT IN (SELECT run_id FROM run_embeddings)
            ORDER BY r.created_at
            """
        )
        runs = cur.fetchall()

    count = 0
    for run_row_tuple in runs:
        run_id, domain, analysis_type, query, findings, workflows, file_name, domain_ctx_json = run_row_tuple

        # Fetch steps and insights for this run
        with conn.cursor() as cur:
            cur.execute(
                "SELECT step_name, description, params, result_summary FROM run_steps WHERE run_id = %s",
                (run_id,),
            )
            steps = [
                {"step_name": r[0], "description": r[1], "params": r[2], "result_summary": r[3]}
                for r in cur.fetchall()
            ]
            cur.execute(
                "SELECT finding, method FROM run_insights WHERE run_id = %s",
                (run_id,),
            )
            insights = [{"finding": r[0], "method": r[1]} for r in cur.fetchall()]

        domain_ctx = {}
        if domain_ctx_json:
            try:
                domain_ctx = json.loads(domain_ctx_json) if isinstance(domain_ctx_json, str) else domain_ctx_json
            except Exception:
                pass

        run_row = {
            "run_id": run_id,
            "domain": domain,
            "analysis_type": analysis_type,
            "query": query,
            "findings_summary": findings,
            "workflows_run": workflows or [],
            "dataset_name": file_name or "",
        }

        embed_text = build_run_text(run_row, steps, insights)
        metadata = generate_metadata(run_row, steps, insights, domain_ctx)

        save_embedding(conn, run_id, embed_text, metadata)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Semantic recall
# ---------------------------------------------------------------------------

def recall_semantic(
    conn,
    query: str,
    limit: int = 5,
    domain_filter: Optional[str] = None,
    tag_filter: Optional[List[str]] = None,
    min_score: float = 0.0,
) -> List[dict]:
    """
    Find the most semantically similar prior runs to *query* using
    BGE-M3 embeddings and cosine similarity (computed in Python).

    Parameters
    ----------
    conn          : open psycopg2 connection
    query         : natural-language question or description
    limit         : maximum results to return
    domain_filter : optional domain name to restrict results
    tag_filter    : optional list of tags; returns runs matching ANY tag
    min_score     : minimum cosine similarity threshold (0.0 = no filter)

    Returns list of dicts with keys:
        run_id, domain, analysis_type, query, findings_summary,
        created_at, workflows_run, artifact_count, insight_count,
        intent, tags, data_description, similarity_score, match_type
    """
    init_semantic_db(conn)

    # Embed the query
    q_vec = _embed([query])[0]  # (1024,)

    # Build SQL — fetch candidates (optionally filtered by domain/tag)
    conditions = []
    params = []

    if domain_filter:
        conditions.append("e.domain = %s")
        params.append(domain_filter)

    if tag_filter:
        placeholders = ",".join(["%s"] * len(tag_filter))
        conditions.append(
            f"e.run_id IN (SELECT run_id FROM embedding_tags WHERE tag IN ({placeholders}))"
        )
        params.extend(tag_filter)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT e.run_id, e.embedding, e.domain, e.intent, e.tags,
                   e.data_description,
                   r.analysis_type, r.query, r.findings_summary,
                   r.created_at, r.workflows_run,
                   (SELECT COUNT(*) FROM run_artifacts a WHERE a.run_id = e.run_id),
                   (SELECT COUNT(*) FROM run_insights  i WHERE i.run_id = e.run_id)
            FROM run_embeddings e
            JOIN analysis_runs r ON r.run_id = e.run_id
            {where_clause}
            """,
            params,
        )
        rows = cur.fetchall()

    if not rows:
        return []

    # Compute cosine similarity in numpy
    # Embeddings are already L2-normalised, so dot product == cosine similarity
    emb_matrix = np.array([r[1] for r in rows], dtype=np.float32)  # (N, 1024)
    scores = emb_matrix @ q_vec  # (N,)

    # Sort by score descending
    order = np.argsort(-scores)

    results = []
    for idx in order:
        score = float(scores[idx])
        if score < min_score:
            continue
        row = rows[idx]
        run_id, _, domain, intent, tags, data_desc, \
            analysis_type, run_query, findings, created_at, \
            workflows, artifact_count, insight_count = row

        results.append({
            "run_id": run_id,
            "domain": domain,
            "analysis_type": analysis_type,
            "query": run_query,
            "findings_summary": findings,
            "created_at": created_at,
            "workflows_run": workflows or [],
            "artifact_count": int(artifact_count),
            "insight_count": int(insight_count),
            "intent": intent,
            "tags": tags or [],
            "data_description": data_desc,
            "similarity_score": round(score, 4),
            "match_type": "semantic",
        })
        if len(results) >= limit:
            break

    return results


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_semantic_memory(results: List[dict], current_query: str = "") -> str:
    """Render semantic recall results as an ASCII-safe console block."""
    if not results:
        out = (
            "=== SEMANTIC MEMORY ===\n"
            "No semantically similar prior runs found.\n"
            "=======================\n"
        )
        return out.encode("ascii", errors="replace").decode("ascii")

    lines = [
        "=== SEMANTIC MEMORY ===",
        f"Found {len(results)} semantically similar prior run(s).",
        "",
    ]

    for r in results:
        score_bar = "#" * int(r["similarity_score"] * 20)
        lines.append(f"[SEMANTIC MATCH  score={r['similarity_score']:.3f}  {score_bar}]")
        lines.append(f"  Run     : {r['run_id']} | Domain : {r['domain'] or 'Unknown'}")
        lines.append(f"  Type    : {r['analysis_type'] or '?'} | Intent: {r['intent'] or '?'}")
        workflows_str = ", ".join(r.get("workflows_run") or [])
        lines.append(f"  Workflows: {workflows_str or '?'}")
        lines.append(f'  Query   : "{(r.get("query") or "")[:120]}"')
        lines.append(f"  Findings: {(r.get('findings_summary') or '')[:200]}")
        lines.append(f"  Tags    : {', '.join((r.get('tags') or [])[:12])}")
        lines.append(f"  Artifacts: {r['artifact_count']} | Insights: {r['insight_count']}")
        lines.append(f"  Date    : {str(r.get('created_at', ''))[:19]}")
        lines.append("")

    lines.append("Tip: High-score matches (>0.75) indicate very similar prior analyses — review findings before re-running.")
    lines.append("=======================")

    out = "\n".join(lines)
    return out.encode("ascii", errors="replace").decode("ascii")
