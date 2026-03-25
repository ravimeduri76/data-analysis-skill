"""
db_store.py -- PostgreSQL persistence layer for data_analysis skill.

Stores analysis runs, steps, artifacts, insights, and what-if suggestions
so future runs can retrieve and leverage learnings from past analyses.

Public API
----------
get_connection(conn_string=None) -> psycopg2.connection
init_db(conn) -> None
save_run(conn, trace, domain_ctx, findings_summary, analysis_type,
         workflows_run, row_count, col_count,
         what_if_suggestions=None) -> str
recall_similar_runs(conn, dataset_sha256=None, domain=None,
                    dataset_name=None, limit=5) -> list[dict]
render_memory(runs, current_query="") -> str

Environment
-----------
POSTGRES_URL -- libpq connection string, e.g.
    postgresql://user:password@localhost:5432/data_analysis
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import List, Optional


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(conn_string: Optional[str] = None):
    """Return a psycopg2 connection.

    Uses *conn_string* if provided, else falls back to the ``POSTGRES_URL``
    environment variable.  Raises ``ImportError`` (with pip hint) if
    psycopg2 is not installed.
    """
    try:
        import psycopg2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for DB persistence. "
            "Install with: pip install psycopg2-binary"
        ) from exc

    url = conn_string or os.getenv("POSTGRES_URL")
    if not url:
        raise ValueError(
            "No connection string provided and POSTGRES_URL is not set."
        )
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS datasets (
    id          SERIAL PRIMARY KEY,
    sha256      TEXT UNIQUE NOT NULL,
    file_path   TEXT,
    file_name   TEXT,
    row_count   INTEGER,
    col_count   INTEGER,
    first_seen  TIMESTAMPTZ DEFAULT NOW(),
    last_seen   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id               SERIAL PRIMARY KEY,
    run_id           TEXT UNIQUE NOT NULL,
    dataset_id       INTEGER REFERENCES datasets(id),
    domain           TEXT,
    analysis_type    TEXT,
    workflows_run    TEXT[],
    query            TEXT,
    findings_summary TEXT,
    domain_ctx       JSONB,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS run_steps (
    id             SERIAL PRIMARY KEY,
    run_id         TEXT REFERENCES analysis_runs(run_id),
    step_name      TEXT,
    description    TEXT,
    params         JSONB,
    result_summary TEXT,
    logged_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS run_artifacts (
    id          SERIAL PRIMARY KEY,
    run_id      TEXT REFERENCES analysis_runs(run_id),
    file_path   TEXT,
    description TEXT,
    row_count   INTEGER,
    col_count   INTEGER,
    logged_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS run_insights (
    id             SERIAL PRIMARY KEY,
    run_id         TEXT REFERENCES analysis_runs(run_id),
    finding        TEXT,
    metric         TEXT,
    source_hook    TEXT,
    columns_used   TEXT,
    rows_analyzed  INTEGER,
    method         TEXT,
    confidence     TEXT,
    caveats        TEXT,
    sql_equivalent TEXT
);

CREATE TABLE IF NOT EXISTS what_if_suggestions (
    id                 SERIAL PRIMARY KEY,
    run_id             TEXT REFERENCES analysis_runs(run_id),
    missing_data       TEXT,
    current_limitation TEXT,
    what_unlocks       TEXT,
    data_source        TEXT,
    priority           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_domain ON analysis_runs(domain);
CREATE INDEX IF NOT EXISTS idx_datasets_sha ON datasets(sha256);
"""


def init_db(conn) -> None:
    """Create all tables and indexes if they do not yet exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Save a complete run
# ---------------------------------------------------------------------------

def save_run(
    conn,
    trace,
    domain_ctx: dict,
    findings_summary: str,
    analysis_type: str,
    workflows_run: List[str],
    row_count: int,
    col_count: int,
    what_if_suggestions=None,
) -> str:
    """Persist a full analysis run to the database.

    Parameters
    ----------
    conn              : open psycopg2 connection
    trace             : AnalysisTrace instance (must have run_id, file_sha256,
                        dataset_path, query, _steps, _artifacts, _insights)
    domain_ctx        : dict returned by detect_domain
    findings_summary  : one-sentence summary of key findings
    analysis_type     : "profile" | "anomaly" | "sampling" | "survival"
    workflows_run     : list of workflow labels, e.g. ["A"]
    row_count         : number of rows in the analysed dataset
    col_count         : number of columns in the analysed dataset
    what_if_suggestions : list of suggestion dicts from suggest_what_if (optional)

    Returns
    -------
    run_id : str  (same as trace.run_id)
    """
    sha = getattr(trace, "file_sha256", "unavailable")
    dataset_path = getattr(trace, "dataset_path", "")
    file_name = os.path.basename(dataset_path) if dataset_path else ""

    with conn.cursor() as cur:
        # 1. Upsert dataset fingerprint
        cur.execute(
            """
            INSERT INTO datasets (sha256, file_path, file_name, row_count, col_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (sha256) DO UPDATE
                SET last_seen  = NOW(),
                    row_count  = EXCLUDED.row_count,
                    col_count  = EXCLUDED.col_count
            RETURNING id
            """,
            (sha, dataset_path, file_name, row_count, col_count),
        )
        dataset_id = cur.fetchone()[0]

        # 2. Insert analysis run
        domain_ctx_json = json.dumps(domain_ctx, default=str)
        cur.execute(
            """
            INSERT INTO analysis_runs
                (run_id, dataset_id, domain, analysis_type, workflows_run,
                 query, findings_summary, domain_ctx)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                trace.run_id,
                dataset_id,
                domain_ctx.get("domain", "Unknown"),
                analysis_type,
                workflows_run,
                getattr(trace, "query", ""),
                findings_summary,
                domain_ctx_json,
            ),
        )

        # 3. Bulk-insert steps
        steps = getattr(trace, "_steps", [])
        if steps:
            cur.executemany(
                """
                INSERT INTO run_steps
                    (run_id, step_name, description, params, result_summary, logged_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        trace.run_id,
                        s.get("step", ""),
                        s.get("description", ""),
                        json.dumps(s.get("params", {}), default=str),
                        s.get("result_summary", ""),
                        s.get("timestamp"),
                    )
                    for s in steps
                ],
            )

        # 4. Bulk-insert artifacts
        artifacts = getattr(trace, "_artifacts", [])
        if artifacts:
            cur.executemany(
                """
                INSERT INTO run_artifacts
                    (run_id, file_path, description, row_count, col_count, logged_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        trace.run_id,
                        a.get("path", a.get("file", "")),
                        a.get("description", ""),
                        a.get("row_count"),
                        a.get("col_count"),
                        a.get("timestamp"),
                    )
                    for a in artifacts
                ],
            )

        # 5. Bulk-insert insights
        insights = getattr(trace, "_insights", [])
        if insights:
            cur.executemany(
                """
                INSERT INTO run_insights
                    (run_id, finding, metric, source_hook, columns_used,
                     rows_analyzed, method, confidence, caveats, sql_equivalent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        trace.run_id,
                        ins.get("finding", ""),
                        ins.get("metric", ""),
                        ins.get("source_hook", ""),
                        ins.get("columns_used", ""),
                        ins.get("rows_analyzed"),
                        ins.get("method", ""),
                        ins.get("confidence", ""),
                        ins.get("caveats", ""),
                        ins.get("sql_equivalent", ""),
                    )
                    for ins in insights
                ],
            )

        # 6. Bulk-insert what-if suggestions
        if what_if_suggestions:
            rows = []
            for s in what_if_suggestions:
                if dataclasses.is_dataclass(s):
                    s = dataclasses.asdict(s)
                rows.append((
                    trace.run_id,
                    s.get("missing_data", ""),
                    s.get("current_limitation", ""),
                    s.get("what_unlocks", ""),
                    s.get("data_source", ""),
                    s.get("priority", ""),
                ))
            if rows:
                cur.executemany(
                    """
                    INSERT INTO what_if_suggestions
                        (run_id, missing_data, current_limitation,
                         what_unlocks, data_source, priority)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )

    conn.commit()

    # Auto-embed the run for semantic recall (best-effort — never breaks the workflow)
    try:
        from hooks.semantic_store import build_run_text, generate_metadata, save_embedding
        _run_row = {
            "run_id": trace.run_id,
            "domain": domain_ctx.get("domain", "Unknown"),
            "analysis_type": analysis_type,
            "query": getattr(trace, "query", ""),
            "findings_summary": findings_summary,
            "workflows_run": workflows_run,
            "dataset_name": file_name,
        }
        _steps_for_embed = [
            {
                "step_name": s.get("step", ""),
                "description": s.get("description", ""),
                "params": s.get("params", {}),
                "result_summary": s.get("result_summary", ""),
            }
            for s in steps
        ]
        _insights_for_embed = [
            {"finding": i.get("finding", ""), "method": i.get("method", "")}
            for i in insights
        ]
        _embed_text = build_run_text(_run_row, _steps_for_embed, _insights_for_embed)
        _metadata = generate_metadata(_run_row, _steps_for_embed, _insights_for_embed, domain_ctx)
        save_embedding(conn, trace.run_id, _embed_text, _metadata)
    except Exception:
        pass  # semantic embedding is optional; failure must never surface to the caller

    return trace.run_id


# ---------------------------------------------------------------------------
# Recall prior runs
# ---------------------------------------------------------------------------

def recall_similar_runs(
    conn,
    dataset_sha256: Optional[str] = None,
    domain: Optional[str] = None,
    dataset_name: Optional[str] = None,
    limit: int = 5,
) -> List[dict]:
    """Return a list of prior runs relevant to the current analysis.

    Matching priority (from highest to lowest):
    1. ``exact_file``   — same SHA-256 hash
    2. ``same_domain``  — same detected domain
    3. ``similar_file`` — file name ILIKE %dataset_name%

    Up to *limit* results are returned per matching tier.
    """
    results: List[dict] = []
    seen_run_ids: set = set()

    tiers = []
    if dataset_sha256 and dataset_sha256 != "unavailable":
        tiers.append(("exact_file", dataset_sha256, None))
    if domain:
        tiers.append(("same_domain", None, None))
    if dataset_name:
        tiers.append(("similar_file", None, dataset_name))

    with conn.cursor() as cur:
        for match_type, sha, name in tiers:
            if match_type == "exact_file":
                cur.execute(
                    """
                    SELECT
                        r.run_id, r.domain, r.analysis_type, r.query,
                        r.findings_summary, r.created_at, r.workflows_run,
                        (SELECT COUNT(*) FROM run_artifacts a WHERE a.run_id = r.run_id) AS artifact_count,
                        (SELECT COUNT(*) FROM run_insights  i WHERE i.run_id = r.run_id) AS insight_count
                    FROM analysis_runs r
                    JOIN datasets d ON d.id = r.dataset_id
                    WHERE d.sha256 = %s
                    ORDER BY r.created_at DESC
                    LIMIT %s
                    """,
                    (sha, limit),
                )
            elif match_type == "same_domain":
                placeholders = ",".join(["%s"] * len(seen_run_ids)) if seen_run_ids else "NULL"
                query_str = f"""
                    SELECT
                        r.run_id, r.domain, r.analysis_type, r.query,
                        r.findings_summary, r.created_at, r.workflows_run,
                        (SELECT COUNT(*) FROM run_artifacts a WHERE a.run_id = r.run_id) AS artifact_count,
                        (SELECT COUNT(*) FROM run_insights  i WHERE i.run_id = r.run_id) AS insight_count
                    FROM analysis_runs r
                    WHERE r.domain = %s
                    {"AND r.run_id NOT IN (" + placeholders + ")" if seen_run_ids else ""}
                    ORDER BY r.created_at DESC
                    LIMIT %s
                """
                params = [domain] + list(seen_run_ids) + [limit]
                cur.execute(query_str, params)
            else:  # similar_file
                placeholders = ",".join(["%s"] * len(seen_run_ids)) if seen_run_ids else "NULL"
                query_str = f"""
                    SELECT
                        r.run_id, r.domain, r.analysis_type, r.query,
                        r.findings_summary, r.created_at, r.workflows_run,
                        (SELECT COUNT(*) FROM run_artifacts a WHERE a.run_id = r.run_id) AS artifact_count,
                        (SELECT COUNT(*) FROM run_insights  i WHERE i.run_id = r.run_id) AS insight_count
                    FROM analysis_runs r
                    JOIN datasets d ON d.id = r.dataset_id
                    WHERE d.file_name ILIKE %s
                    {"AND r.run_id NOT IN (" + placeholders + ")" if seen_run_ids else ""}
                    ORDER BY r.created_at DESC
                    LIMIT %s
                """
                params = [f"%{name}%"] + list(seen_run_ids) + [limit]
                cur.execute(query_str, params)

            rows = cur.fetchall()
            for row in rows:
                run_id = row[0]
                if run_id in seen_run_ids:
                    continue
                seen_run_ids.add(run_id)
                workflows = row[6] or []
                results.append({
                    "run_id": run_id,
                    "domain": row[1],
                    "analysis_type": row[2],
                    "query": row[3],
                    "findings_summary": row[4],
                    "created_at": row[5],
                    "workflows_run": workflows,
                    "artifact_count": int(row[7]),
                    "insight_count": int(row[8]),
                    "match_type": match_type,
                })

    return results


# ---------------------------------------------------------------------------
# Render memory block
# ---------------------------------------------------------------------------

_MATCH_LABEL = {
    "exact_file": "[EXACT FILE MATCH]",
    "same_domain": "[SAME DOMAIN]",
    "similar_file": "[SIMILAR FILE NAME]",
}


def render_memory(runs: List[dict], current_query: str = "") -> str:
    """Render a human-readable ASCII console block of prior runs.

    Returns a string safe for ``print()`` in any terminal (ASCII-only).
    """
    if not runs:
        out = (
            "=== ANALYSIS MEMORY ===\n"
            "No prior runs found for this dataset.\n"
            "=======================\n"
        )
        return out.encode("ascii", errors="replace").decode("ascii")

    lines = [
        "=== ANALYSIS MEMORY ===",
        f"Found {len(runs)} relevant prior run(s) for this dataset/domain.",
        "",
    ]

    for run in runs:
        label = _MATCH_LABEL.get(run["match_type"], f"[{run['match_type'].upper()}]")
        lines.append(label)

        workflows_str = ", ".join(run.get("workflows_run") or [])
        created_str = str(run.get("created_at", ""))[:19]  # YYYY-MM-DD HH:MM:SS

        lines.append(
            f"  Run: {run['run_id']} | Domain: {run['domain'] or 'Unknown'}"
        )
        lines.append(
            f"  Type: {run['analysis_type'] or '?'} | Workflows: {workflows_str or '?'}"
        )
        query_text = (run.get("query") or "")[:120]
        lines.append(f'  Query: "{query_text}"')
        findings_text = (run.get("findings_summary") or "")[:200]
        lines.append(f"  Findings: {findings_text}")
        lines.append(
            f"  Artifacts: {run['artifact_count']} | Insights: {run['insight_count']}"
        )
        lines.append(f"  Date: {created_str}")
        lines.append("")

    lines.append("Tip: Review prior findings before running to avoid redundant analysis.")
    lines.append("=======================")

    out = "\n".join(lines)
    return out.encode("ascii", errors="replace").decode("ascii")
