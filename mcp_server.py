"""
mcp_server.py -- Analysis Memory MCP Server (SSE/stdio transport)

Wraps the PostgreSQL analysis memory store as an MCP server so Claude Code,
notebooks, agents, and any MCP-compatible client can query and write the DB
through a schema-free, tool/prompt/resource interface.

Capabilities
------------
  Tools (8)    : search_runs, get_run, list_runs, list_datasets,
                 get_domain_summary, save_analysis_run, log_insight,
                 backfill_embeddings
  Prompts (4)  : analyze_dataset, compare_runs, domain_briefing, leakage_audit
  Resources (4): analysis://runs/recent, analysis://domains,
                 analysis://datasets, analysis://run/{run_id}
  Sampling     : get_domain_summary + search_runs (graceful degradation)
  Elicitation  : save_analysis_run when domain confidence is low (graceful)
                 log_insight always — confirm, correct finding, add notes, sign off (graceful)

Start
-----
  python mcp_server.py          # starts stdio server; Ctrl-C to quit
  (Or let Claude Code launch it via .mcp.json)

Dependencies
------------
  pip install mcp python-dotenv psycopg2-binary
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Path + env setup — must happen before hook imports
# ---------------------------------------------------------------------------
_BASE_DIR = r"C:\Users\Ravi\research\data-analysis"
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
except ImportError:
    pass  # dotenv optional; POSTGRES_URL may be set directly in environment

# ---------------------------------------------------------------------------
# MCP imports
# ---------------------------------------------------------------------------
import argparse
import threading

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp import types
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions

# ---------------------------------------------------------------------------
# Connection pool — replaces single-connection cache for reliability
# ---------------------------------------------------------------------------
from psycopg2.pool import ThreadedConnectionPool

_pool: ThreadedConnectionPool | None = None
_pool_initialised: bool = False
_thread_conns: threading.local = threading.local()


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        dsn = os.environ.get(
            "POSTGRES_URL",
            "postgresql://postgres:memorystorepg@localhost:5432/data_analysis",
        )
        _pool = ThreadedConnectionPool(minconn=2, maxconn=10, dsn=dsn)
    return _pool


def _ensure_initialised() -> None:
    global _pool_initialised
    if _pool_initialised:
        return
    conn = _get_pool().getconn()
    try:
        from hooks.db_store import init_db
        from hooks.semantic_store import init_semantic_db
        init_db(conn)
        init_semantic_db(conn)
        _pool_initialised = True
    finally:
        _get_pool().putconn(conn)


def get_conn():
    """Get a connection from the pool (backward-compatible interface)."""
    _ensure_initialised()
    conn = _get_pool().getconn()
    if not hasattr(_thread_conns, "checked_out"):
        _thread_conns.checked_out = []
    _thread_conns.checked_out.append(conn)
    return conn


def _release_thread_conns() -> None:
    """Return all connections checked out by the current thread to the pool."""
    if not hasattr(_thread_conns, "checked_out"):
        return
    for conn in _thread_conns.checked_out[:]:
        try:
            conn.rollback()
            _get_pool().putconn(conn)
        except Exception:
            pass
    _thread_conns.checked_out.clear()


# ---------------------------------------------------------------------------
# Minimal trace object — db_store.save_run expects attribute access
# ---------------------------------------------------------------------------
class _RunTrace:
    """Lightweight stand-in for AnalysisTrace used by save_run."""

    def __init__(
        self,
        run_id: str,
        file_sha256: str,
        dataset_path: str,
        query: str,
        steps: list,
        artifacts: list,
        insights: list,
    ) -> None:
        self.run_id = run_id
        self.file_sha256 = file_sha256
        self.dataset_path = dataset_path
        self.query = query
        self._steps = steps
        self._artifacts = artifacts
        self._insights = insights


# ---------------------------------------------------------------------------
# JSON serialiser that handles datetime / psycopg2 types
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, indent=2)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
server = Server("analysis-memory")


# ============================================================
# TOOLS
# ============================================================

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_runs",
            description=(
                "Semantic similarity search over past analysis runs using BGE-M3 embeddings. "
                "Returns runs ranked by cosine similarity to the query. "
                "Set synthesize=true to also get an LLM-generated key-lessons summary (requires sampling support)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language description of what you are looking for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default 5)",
                        "default": 5,
                    },
                    "domain_filter": {
                        "type": "string",
                        "description": "Restrict to a specific domain (e.g. 'Healthcare')",
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum cosine similarity threshold 0.0–1.0 (default 0.0)",
                        "default": 0.0,
                    },
                    "synthesize": {
                        "type": "boolean",
                        "description": "Generate a key-lessons paragraph via LLM sampling (default false)",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_run",
            description=(
                "Retrieve full details for one analysis run: steps, artifacts, "
                "insights, what-if suggestions, and dataset metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "The run_id to retrieve",
                    },
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="list_runs",
            description="List recent analysis runs with optional filters by domain and analysis type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Filter by domain name",
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Filter by type: profile | anomaly | sampling | survival",
                        "enum": ["profile", "anomaly", "sampling", "survival"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 20)",
                        "default": 20,
                    },
                },
            },
        ),
        types.Tool(
            name="list_datasets",
            description="List all datasets ever analysed with row/col counts and last-seen date.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default 50)",
                        "default": 50,
                    },
                },
            },
        ),
        types.Tool(
            name="get_domain_summary",
            description=(
                "Aggregated statistics for a domain (run count, dataset count, analysis types, "
                "recent findings). Set synthesize=true to generate a plain-English narrative "
                "via LLM sampling."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain name (e.g. Healthcare, Retail, General)",
                    },
                    "synthesize": {
                        "type": "boolean",
                        "description": "Generate narrative via LLM sampling (default true)",
                        "default": True,
                    },
                },
                "required": ["domain"],
            },
        ),
        types.Tool(
            name="save_analysis_run",
            description=(
                "Persist a new analysis run to the database. "
                "Includes domain_ctx, steps, insights, artifacts, and what-if suggestions. "
                "If domain confidence is low or domain is 'General', triggers elicitation "
                "to confirm/correct the domain before saving."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Unique run identifier"},
                    "dataset_path": {"type": "string", "description": "Full path to the analysed file"},
                    "file_sha256": {
                        "type": "string",
                        "description": "SHA-256 of the dataset file (default 'unavailable')",
                        "default": "unavailable",
                    },
                    "query": {"type": "string", "description": "Original analysis question or objective"},
                    "domain_ctx": {
                        "type": "object",
                        "description": "Domain context dict from detect_domain (must include 'domain' key)",
                    },
                    "findings_summary": {
                        "type": "string",
                        "description": "One-sentence summary of key findings",
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "profile | anomaly | sampling | survival",
                        "enum": ["profile", "anomaly", "sampling", "survival"],
                    },
                    "workflows_run": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workflow labels executed (e.g. ['A'])",
                    },
                    "row_count": {"type": "integer", "description": "Number of rows in the dataset"},
                    "col_count": {"type": "integer", "description": "Number of columns in the dataset"},
                    "steps": {
                        "type": "array",
                        "description": "List of step dicts (step, description, params, result_summary, timestamp)",
                        "default": [],
                    },
                    "artifacts": {
                        "type": "array",
                        "description": "List of artifact dicts (path/file, description, row_count, col_count, timestamp)",
                        "default": [],
                    },
                    "insights": {
                        "type": "array",
                        "description": "List of InsightCard dicts (finding, metric, source_hook, ...)",
                        "default": [],
                    },
                    "what_if_suggestions": {
                        "type": "array",
                        "description": "List of what-if suggestion dicts from suggest_what_if",
                        "default": [],
                    },
                },
                "required": [
                    "run_id", "domain_ctx", "findings_summary",
                    "analysis_type", "workflows_run", "row_count", "col_count",
                ],
            },
        ),
        types.Tool(
            name="log_insight",
            description="Append an InsightCard to an existing analysis run. Triggers human review elicitation before logging — user can approve, correct the finding, add notes, and sign off.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "The run to append the insight to"},
                    "finding": {"type": "string", "description": "The insight finding text"},
                    "metric": {"type": "string", "default": ""},
                    "source_hook": {"type": "string", "default": ""},
                    "columns_used": {"type": "string", "default": ""},
                    "rows_analyzed": {"type": "integer"},
                    "method": {"type": "string", "default": ""},
                    "confidence": {"type": "string", "default": ""},
                    "caveats": {"type": "string", "default": ""},
                    "sql_equivalent": {"type": "string", "default": ""},
                },
                "required": ["run_id", "finding"],
            },
        ),
        types.Tool(
            name="backfill_embeddings",
            description=(
                "Embed all analysis runs that are missing BGE-M3 vector embeddings. "
                "Returns the count of newly-embedded runs. Safe to call repeatedly (idempotent)."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:  # type: ignore[override]
    try:
        return await _call_tool_impl(name, arguments)
    finally:
        _release_thread_conns()


async def _call_tool_impl(name: str, arguments: dict) -> list[types.TextContent]:
    conn = get_conn()
    ctx = server.request_context  # RequestContext with .session for sampling/elicitation

    # ------------------------------------------------------------------
    # search_runs
    # ------------------------------------------------------------------
    if name == "search_runs":
        query = arguments["query"]
        limit = int(arguments.get("limit", 5))
        domain_filter = arguments.get("domain_filter")
        min_score = float(arguments.get("min_score", 0.0))
        synthesize = bool(arguments.get("synthesize", False))

        from hooks.semantic_store import recall_semantic

        results = recall_semantic(
            conn, query, limit=limit,
            domain_filter=domain_filter,
            min_score=min_score,
        )

        if synthesize and results:
            synthesis = await _sample_synthesis(ctx, query, results)
            payload = _dumps({"results": results, "synthesis": synthesis})
        else:
            payload = _dumps(results)

        return [types.TextContent(type="text", text=payload)]

    # ------------------------------------------------------------------
    # get_run
    # ------------------------------------------------------------------
    elif name == "get_run":
        run_id = arguments["run_id"]

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.run_id, r.domain, r.analysis_type, r.workflows_run,
                       r.query, r.findings_summary, r.domain_ctx, r.created_at,
                       d.file_name, d.row_count, d.col_count, d.file_path
                FROM analysis_runs r
                LEFT JOIN datasets d ON d.id = r.dataset_id
                WHERE r.run_id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()

        if not row:
            return [types.TextContent(
                type="text",
                text=_dumps({"error": f"Run {run_id!r} not found"}),
            )]

        run: dict = {
            "run_id": row[0],
            "domain": row[1],
            "analysis_type": row[2],
            "workflows_run": row[3],
            "query": row[4],
            "findings_summary": row[5],
            "domain_ctx": row[6],
            "created_at": row[7],
            "dataset": {
                "file_name": row[8],
                "row_count": row[9],
                "col_count": row[10],
                "file_path": row[11],
            },
        }

        with conn.cursor() as cur:
            cur.execute(
                "SELECT step_name, description, params, result_summary, logged_at "
                "FROM run_steps WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            run["steps"] = [
                {
                    "step_name": r[0], "description": r[1],
                    "params": r[2], "result_summary": r[3], "logged_at": r[4],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT file_path, description, row_count, col_count, logged_at "
                "FROM run_artifacts WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            run["artifacts"] = [
                {
                    "file_path": r[0], "description": r[1],
                    "row_count": r[2], "col_count": r[3], "logged_at": r[4],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT finding, metric, source_hook, columns_used, rows_analyzed, "
                "method, confidence, caveats, sql_equivalent "
                "FROM run_insights WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            run["insights"] = [
                {
                    "finding": r[0], "metric": r[1], "source_hook": r[2],
                    "columns_used": r[3], "rows_analyzed": r[4], "method": r[5],
                    "confidence": r[6], "caveats": r[7], "sql_equivalent": r[8],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT missing_data, current_limitation, what_unlocks, data_source, priority "
                "FROM what_if_suggestions WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            run["what_if_suggestions"] = [
                {
                    "missing_data": r[0], "current_limitation": r[1],
                    "what_unlocks": r[2], "data_source": r[3], "priority": r[4],
                }
                for r in cur.fetchall()
            ]

        return [types.TextContent(type="text", text=_dumps(run))]

    # ------------------------------------------------------------------
    # list_runs
    # ------------------------------------------------------------------
    elif name == "list_runs":
        domain = arguments.get("domain")
        analysis_type = arguments.get("analysis_type")
        limit = int(arguments.get("limit", 20))

        conditions: list[str] = []
        params: list = []
        if domain:
            conditions.append("r.domain = %s")
            params.append(domain)
        if analysis_type:
            conditions.append("r.analysis_type = %s")
            params.append(analysis_type)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT r.run_id, r.domain, r.analysis_type, r.workflows_run,
                       r.query, r.findings_summary, r.created_at,
                       d.file_name, d.row_count, d.col_count
                FROM analysis_runs r
                LEFT JOIN datasets d ON d.id = r.dataset_id
                {where}
                ORDER BY r.created_at DESC
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        results = [
            {
                "run_id": r[0], "domain": r[1], "analysis_type": r[2],
                "workflows_run": r[3], "query": r[4], "findings_summary": r[5],
                "created_at": r[6], "dataset_name": r[7],
                "row_count": r[8], "col_count": r[9],
            }
            for r in rows
        ]
        return [types.TextContent(type="text", text=_dumps(results))]

    # ------------------------------------------------------------------
    # list_datasets
    # ------------------------------------------------------------------
    elif name == "list_datasets":
        limit = int(arguments.get("limit", 50))

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.file_name, d.file_path, d.row_count, d.col_count,
                       d.first_seen, d.last_seen,
                       (SELECT COUNT(*) FROM analysis_runs r WHERE r.dataset_id = d.id) AS run_count
                FROM datasets d
                ORDER BY d.last_seen DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

        results = [
            {
                "file_name": r[0], "file_path": r[1], "row_count": r[2],
                "col_count": r[3], "first_seen": r[4], "last_seen": r[5],
                "run_count": int(r[6]),
            }
            for r in rows
        ]
        return [types.TextContent(type="text", text=_dumps(results))]

    # ------------------------------------------------------------------
    # get_domain_summary
    # ------------------------------------------------------------------
    elif name == "get_domain_summary":
        domain = arguments["domain"]
        synthesize = bool(arguments.get("synthesize", True))

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS run_count,
                       COUNT(DISTINCT r.dataset_id) AS dataset_count,
                       MIN(r.created_at) AS first_run,
                       MAX(r.created_at) AS last_run,
                       array_agg(DISTINCT r.analysis_type)
                           FILTER (WHERE r.analysis_type IS NOT NULL) AS types,
                       array_agg(DISTINCT unnested_wf)
                           FILTER (WHERE unnested_wf IS NOT NULL) AS workflows
                FROM analysis_runs r
                LEFT JOIN LATERAL unnest(r.workflows_run) AS unnested_wf ON TRUE
                WHERE r.domain = %s
                """,
                (domain,),
            )
            agg = cur.fetchone()

            cur.execute(
                """
                SELECT findings_summary FROM analysis_runs
                WHERE domain = %s AND findings_summary IS NOT NULL
                ORDER BY created_at DESC LIMIT 10
                """,
                (domain,),
            )
            findings_rows = cur.fetchall()

        stats: dict = {
            "domain": domain,
            "run_count": int(agg[0]) if agg and agg[0] else 0,
            "dataset_count": int(agg[1]) if agg and agg[1] else 0,
            "first_run": agg[2] if agg else None,
            "last_run": agg[3] if agg else None,
            "analysis_types": agg[4] or [] if agg else [],
            "workflows": agg[5] or [] if agg else [],
            "recent_findings": [r[0] for r in findings_rows],
        }

        result: dict = {"stats": stats}

        if synthesize and stats["run_count"] > 0:
            narrative = await _sample_domain_narrative(ctx, stats)
            if narrative:
                result["narrative"] = narrative

        return [types.TextContent(type="text", text=_dumps(result))]

    # ------------------------------------------------------------------
    # save_analysis_run
    # ------------------------------------------------------------------
    elif name == "save_analysis_run":
        run_id: str = arguments["run_id"]
        domain_ctx: dict = arguments["domain_ctx"]
        findings_summary: str = arguments["findings_summary"]
        analysis_type: str = arguments["analysis_type"]
        workflows_run: list = arguments["workflows_run"]
        row_count: int = int(arguments["row_count"])
        col_count: int = int(arguments["col_count"])
        dataset_path: str = arguments.get("dataset_path", "")
        file_sha256: str = arguments.get("file_sha256", "unavailable")
        query: str = arguments.get("query", "")
        steps: list = arguments.get("steps", [])
        artifacts: list = arguments.get("artifacts", [])
        insights: list = arguments.get("insights", [])
        what_if_suggestions: list = arguments.get("what_if_suggestions", [])

        # Elicitation when domain confidence is low or domain is generic
        confidence = domain_ctx.get("confidence", "high")
        detected_domain = domain_ctx.get("domain", "General")
        if confidence in ("low", "medium") or detected_domain in ("General", "Unknown"):
            try:
                elicit_result = await ctx.session.elicit(
                    message=(
                        f"Domain detected as '{detected_domain}' with {confidence} confidence. "
                        "Please confirm or correct the domain before saving."
                    ),
                    requestedSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "title": "Domain",
                                "description": "Confirmed or corrected domain name",
                                "default": detected_domain,
                            },
                            "confirm": {
                                "type": "boolean",
                                "title": "Proceed with save",
                                "default": True,
                            },
                        },
                        "required": ["domain", "confirm"],
                    },
                )
                if elicit_result and hasattr(elicit_result, "action"):
                    if elicit_result.action == "submit":
                        data = elicit_result.data or {}
                        if not data.get("confirm", True):
                            return [types.TextContent(
                                type="text",
                                text=_dumps({"status": "cancelled", "reason": "User declined to save"}),
                            )]
                        if data.get("domain"):
                            domain_ctx = {**domain_ctx, "domain": data["domain"]}
                    # If action == "cancel" or "decline", proceed with detected domain
            except Exception:
                pass  # Elicitation not supported by client; save with detected domain

        trace = _RunTrace(
            run_id=run_id,
            file_sha256=file_sha256,
            dataset_path=dataset_path,
            query=query,
            steps=steps,
            artifacts=artifacts,
            insights=insights,
        )

        from hooks.db_store import save_run

        saved_id = save_run(
            conn, trace, domain_ctx, findings_summary, analysis_type,
            workflows_run, row_count, col_count,
            what_if_suggestions if what_if_suggestions else None,
        )
        return [types.TextContent(type="text", text=_dumps({"status": "saved", "run_id": saved_id}))]

    # ------------------------------------------------------------------
    # log_insight
    # ------------------------------------------------------------------
    elif name == "log_insight":
        run_id = arguments["run_id"]
        finding = arguments["finding"]
        caveats = arguments.get("caveats", "")

        # Elicitation: require human review before logging every insight
        try:
            elicit_result = await ctx.session.elicit(
                message=(
                    f"Review insight for run '{run_id}' before logging.\n\n"
                    f"Finding: {finding}"
                ),
                requestedSchema={
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "title": "Approve and log this insight",
                            "default": True,
                        },
                        "finding": {
                            "type": "string",
                            "title": "Finding (edit to correct)",
                            "default": finding,
                        },
                        "notes": {
                            "type": "string",
                            "title": "Additional notes / context (optional)",
                            "default": "",
                        },
                        "signed_off_by": {
                            "type": "string",
                            "title": "Sign-off name or initials",
                            "default": "",
                        },
                    },
                    "required": ["confirm", "finding"],
                },
            )
            if elicit_result and hasattr(elicit_result, "action"):
                if elicit_result.action != "submit":
                    return [types.TextContent(
                        type="text",
                        text=_dumps({"status": "cancelled", "reason": "User declined to log insight"}),
                    )]
                data = elicit_result.data or {}
                if not data.get("confirm", True):
                    return [types.TextContent(
                        type="text",
                        text=_dumps({"status": "cancelled", "reason": "User declined to log insight"}),
                    )]
                # Apply any corrections/additions from the human
                finding = data.get("finding", finding)
                parts = [p for p in [caveats, data.get("notes", "").strip()] if p]
                sign_off = data.get("signed_off_by", "").strip()
                if sign_off:
                    parts.append(f"Signed off by: {sign_off}")
                caveats = " | ".join(parts)
        except Exception:
            pass  # Elicitation not supported by client; log as-is

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO run_insights
                    (run_id, finding, metric, source_hook, columns_used,
                     rows_analyzed, method, confidence, caveats, sql_equivalent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    run_id,
                    finding,
                    arguments.get("metric", ""),
                    arguments.get("source_hook", ""),
                    arguments.get("columns_used", ""),
                    arguments.get("rows_analyzed"),
                    arguments.get("method", ""),
                    arguments.get("confidence", ""),
                    caveats,
                    arguments.get("sql_equivalent", ""),
                ),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return [types.TextContent(type="text", text=_dumps({"status": "logged", "insight_id": new_id, "run_id": run_id}))]

    # ------------------------------------------------------------------
    # backfill_embeddings
    # ------------------------------------------------------------------
    elif name == "backfill_embeddings":
        from hooks.semantic_store import backfill_embeddings

        count = backfill_embeddings(conn)
        return [types.TextContent(type="text", text=_dumps({"status": "done", "newly_embedded": count}))]

    # ------------------------------------------------------------------
    # Unknown
    # ------------------------------------------------------------------
    return [types.TextContent(type="text", text=_dumps({"error": f"Unknown tool: {name}"}))]


# ============================================================
# SAMPLING HELPERS
# ============================================================

async def _sample_synthesis(ctx, query: str, results: list) -> str:
    """
    Ask the LLM client to synthesize key lessons from semantic search results.
    Returns empty string if the client does not support sampling.
    """
    try:
        summaries = "\n".join(
            f"- [{r.get('domain', '?')}] {(r.get('findings_summary') or '')[:200]}"
            for r in results[:5]
        )
        prompt = (
            f"You are a data analysis assistant with access to analysis memory.\n"
            f"A user searched for: \"{query}\"\n\n"
            f"Most similar prior analysis runs:\n{summaries}\n\n"
            "In 2-3 sentences, synthesize the key lessons from these prior analyses "
            "that are most relevant to the current query."
        )
        response = await ctx.session.create_message(
            messages=[
                types.SamplingMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt),
                )
            ],
            maxTokens=300,
        )
        content = response.content
        return content.text if hasattr(content, "text") else str(content)
    except Exception:
        return ""


async def _sample_domain_narrative(ctx, stats: dict) -> str:
    """
    Ask the LLM client to narrate what we have learned in a domain.
    Returns empty string if the client does not support sampling.
    """
    try:
        domain = stats["domain"]
        findings_sample = "\n".join(
            f"- {f[:200]}" for f in stats["recent_findings"][:5]
        )
        prompt = (
            f"You are a data analysis assistant with memory of past analyses.\n\n"
            f"Domain: {domain}\n"
            f"Total analysis runs: {stats['run_count']} across {stats['dataset_count']} datasets\n"
            f"Analysis types used: {', '.join(stats['analysis_types'] or [])}\n"
            f"Recent findings:\n{findings_sample or '(none yet)'}\n\n"
            f"In 3-4 sentences, summarise: What have we learned from analysing {domain} datasets? "
            "What patterns recur? What are the recurring data quality issues or modelling challenges?"
        )
        response = await ctx.session.create_message(
            messages=[
                types.SamplingMessage(
                    role="user",
                    content=types.TextContent(type="text", text=prompt),
                )
            ],
            maxTokens=400,
        )
        content = response.content
        return content.text if hasattr(content, "text") else str(content)
    except Exception:
        return ""


# ============================================================
# PROMPTS
# ============================================================

@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="analyze_dataset",
            description="Structured prompt to start a data analysis workflow (A/B/C/D).",
            arguments=[
                types.PromptArgument(
                    name="file_path",
                    description="Absolute path to the dataset file (CSV, Excel, Parquet, JSON)",
                    required=True,
                ),
                types.PromptArgument(
                    name="question",
                    description="Analysis question or objective",
                    required=True,
                ),
            ],
        ),
        types.Prompt(
            name="compare_runs",
            description="Side-by-side comparison of two past analysis runs.",
            arguments=[
                types.PromptArgument(name="run_id_1", description="First run ID", required=True),
                types.PromptArgument(name="run_id_2", description="Second run ID", required=True),
            ],
        ),
        types.Prompt(
            name="domain_briefing",
            description="Brief on everything we have learned in a domain from past analyses.",
            arguments=[
                types.PromptArgument(
                    name="domain",
                    description="Domain name (e.g. Healthcare, Retail, General)",
                    required=True,
                ),
            ],
        ),
        types.Prompt(
            name="leakage_audit",
            description="Temporal leakage audit checklist for a new ML binary classification dataset.",
            arguments=[
                types.PromptArgument(
                    name="target_col",
                    description="Name of the target/label column",
                    required=True,
                ),
                types.PromptArgument(
                    name="feature_cols",
                    description="Comma-separated list of feature columns to audit",
                    required=True,
                ),
            ],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:

    if name == "analyze_dataset":
        file_path = arguments.get("file_path", "<file_path>")
        question = arguments.get("question", "<analysis question>")
        text = (
            f"Please analyse the dataset at: {file_path}\n\n"
            f"Analysis objective: {question}\n\n"
            "## Instructions\n\n"
            "1. Load the dataset and run `quality_report` to understand data shape, nulls, and types.\n"
            "2. Use `detect_domain` to classify the data domain (Healthcare, Retail, Finance, etc.).\n"
            "3. Check analysis memory using `search_runs` for prior analyses of similar data or questions.\n"
            "4. Choose the appropriate workflow based on the objective:\n"
            "   - **Workflow A** — Single-file profiling: quality, anomalies, correlations, outliers, missing values\n"
            "   - **Workflow B** — Multi-file sampling comparison: diversity, cardinality, null completeness\n"
            "   - **Workflow C** — Anomaly/outlier detection focus\n"
            "   - **Workflow D** — Survival/churn time-to-event analysis\n"
            "5. Execute the workflow and generate `InsightCard` objects for each key finding.\n"
            "6. Use `suggest_what_if` to produce data-improvement recommendations.\n"
            "7. Save the run via the `save_analysis_run` MCP tool.\n"
            "8. Write a findings summary and output an `AnalysisTrace` manifest.\n"
        )
        return types.GetPromptResult(
            description=f"Analysis prompt for {file_path}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )

    elif name == "compare_runs":
        run_id_1 = arguments.get("run_id_1", "<run_id_1>")
        run_id_2 = arguments.get("run_id_2", "<run_id_2>")
        text = (
            f"Please compare these two analysis runs side-by-side.\n\n"
            f"**Run 1:** `{run_id_1}`\n"
            f"**Run 2:** `{run_id_2}`\n\n"
            "Use the `get_run` tool to fetch full details for each, then compare:\n\n"
            "1. **Datasets** — same file? same domain? row/col counts\n"
            "2. **Analysis types and workflows** used\n"
            "3. **Key findings** — what did each run discover?\n"
            "4. **Insights** — list unique and common insights across both runs\n"
            "5. **What-if suggestions** — any recurring data gaps?\n"
            "6. **Conclusion** — which run was more thorough? What would you do differently?\n"
        )
        return types.GetPromptResult(
            description=f"Comparison of runs {run_id_1} and {run_id_2}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )

    elif name == "domain_briefing":
        domain = arguments.get("domain", "<domain>")
        text = (
            f"Please provide a briefing on the '{domain}' domain based on analysis memory.\n\n"
            f"Use the `get_domain_summary` tool with `domain='{domain}'` and `synthesize=true`.\n\n"
            "Structure your briefing as:\n\n"
            "1. **Domain Overview** — What kind of data have we analysed in this domain?\n"
            "2. **Recurring Patterns** — What patterns appear consistently across runs?\n"
            "3. **Common Data Quality Issues** — What quality problems recur?\n"
            "4. **Modelling Challenges** — What ML/statistical challenges are typical?\n"
            "5. **Recommended Starting Point** — Based on prior runs, what analysis approach works best?\n"
            "6. **Open Questions** — What what-if suggestions appear most frequently?\n"
        )
        return types.GetPromptResult(
            description=f"Domain briefing for {domain}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )

    elif name == "leakage_audit":
        target_col = arguments.get("target_col", "<target_col>")
        feature_cols = arguments.get("feature_cols", "<feature_cols>")
        features_list = [f.strip() for f in feature_cols.split(",") if f.strip()]
        checklist = "\n".join(f"- [ ] `{f}`" for f in features_list)
        text = (
            f"## Temporal Leakage Audit\n\n"
            f"**Target column:** `{target_col}`\n\n"
            f"**Features to audit:**\n{checklist}\n\n"
            "---\n\n"
            "### Step 1 — Cross-tab test\n"
            "For each feature, run:\n"
            f"```python\npd.crosstab(df[feature] > 0, df['{target_col}'])\n```\n"
            "**Red flag:** if no client exists with `feature > 0` AND `target = 0`, the feature is leaky.\n\n"
            "### Step 2 — Overlap test\n"
            f"```python\nlen(df[df[feature] > 0]) == len(df[df['{target_col}'] == 1])\n```\n"
            "**Red flag:** perfect overlap = perfect leakage.\n\n"
            "### Step 3 — PR-AUC sanity check\n"
            "If `PR-AUC = 1.0` and `MCC = 1.0` across ALL models, a leaky feature is almost certainly present.\n\n"
            "### Step 4 — Label-derived count correction\n"
            "When a feature counts events from the same table used to define the label:\n"
            "```python\ndf[feature] = (df[feature] - 1).clip(lower=0)\n```\n"
            "Verify: after correction, `feature > 0 AND target = 0` should have rows.\n\n"
            "### Step 5 — Prior-event recency features\n"
            "Use `grp.iloc[:-1]` to exclude the triggering event when computing prior-event features.\n"
            "Set sentinel `NEVER = 9999` for `days_since` on clients with ≤1 event.\n\n"
            "### Step 6 — Prospective temporal split (gold standard)\n"
            "Train on features from period T, target from period T+1 to eliminate all temporal leakage.\n"
        )
        return types.GetPromptResult(
            description=f"Leakage audit for target '{target_col}'",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )

    return types.GetPromptResult(
        description="Unknown prompt",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=f"Unknown prompt: {name}"),
            )
        ],
    )


# ============================================================
# RESOURCES
# ============================================================

@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="analysis://runs/recent",  # type: ignore[arg-type]
            name="Recent Analysis Runs",
            description="The 20 most recent analysis runs as JSON.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="analysis://domains",  # type: ignore[arg-type]
            name="Domains",
            description="All domains with run counts and last-run timestamps.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="analysis://datasets",  # type: ignore[arg-type]
            name="Datasets",
            description="All datasets ever analysed, with row/col counts and last-seen date.",
            mimeType="application/json",
        ),
        # analysis://run/{run_id} is handled dynamically in read_resource
        # (not listed here as a static resource since the ID is variable)
    ]


@server.read_resource()
async def read_resource(uri) -> str:  # type: ignore[override]
    uri_str = str(uri)
    conn = get_conn()

    # ------------------------------------------------------------------
    # analysis://runs/recent
    # ------------------------------------------------------------------
    if uri_str == "analysis://runs/recent":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.run_id, r.domain, r.analysis_type, r.workflows_run,
                       r.query, r.findings_summary, r.created_at, d.file_name
                FROM analysis_runs r
                LEFT JOIN datasets d ON d.id = r.dataset_id
                ORDER BY r.created_at DESC
                LIMIT 20
                """
            )
            rows = cur.fetchall()

        results = [
            {
                "run_id": r[0], "domain": r[1], "analysis_type": r[2],
                "workflows_run": r[3], "query": r[4], "findings_summary": r[5],
                "created_at": r[6], "dataset_name": r[7],
            }
            for r in rows
        ]
        return _dumps(results)

    # ------------------------------------------------------------------
    # analysis://domains
    # ------------------------------------------------------------------
    elif uri_str == "analysis://domains":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT domain, COUNT(*) AS run_count, MAX(created_at) AS last_run
                FROM analysis_runs
                WHERE domain IS NOT NULL
                GROUP BY domain
                ORDER BY run_count DESC
                """
            )
            rows = cur.fetchall()

        results = [
            {"domain": r[0], "run_count": int(r[1]), "last_run": r[2]}
            for r in rows
        ]
        return _dumps(results)

    # ------------------------------------------------------------------
    # analysis://datasets
    # ------------------------------------------------------------------
    elif uri_str == "analysis://datasets":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.file_name, d.file_path, d.row_count, d.col_count,
                       d.first_seen, d.last_seen,
                       (SELECT COUNT(*) FROM analysis_runs r WHERE r.dataset_id = d.id) AS run_count
                FROM datasets d
                ORDER BY d.last_seen DESC
                """
            )
            rows = cur.fetchall()

        results = [
            {
                "file_name": r[0], "file_path": r[1], "row_count": r[2],
                "col_count": r[3], "first_seen": r[4], "last_seen": r[5],
                "run_count": int(r[6]),
            }
            for r in rows
        ]
        return _dumps(results)

    # ------------------------------------------------------------------
    # analysis://run/{run_id}  — dynamic per-run detail
    # ------------------------------------------------------------------
    elif uri_str.startswith("analysis://run/"):
        run_id = uri_str[len("analysis://run/"):]

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.run_id, r.domain, r.analysis_type, r.workflows_run,
                       r.query, r.findings_summary, r.domain_ctx, r.created_at,
                       d.file_name, d.row_count, d.col_count
                FROM analysis_runs r
                LEFT JOIN datasets d ON d.id = r.dataset_id
                WHERE r.run_id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()

        if not row:
            return _dumps({"error": f"Run {run_id!r} not found"})

        run: dict = {
            "run_id": row[0], "domain": row[1], "analysis_type": row[2],
            "workflows_run": row[3], "query": row[4], "findings_summary": row[5],
            "domain_ctx": row[6], "created_at": row[7],
            "dataset": {"file_name": row[8], "row_count": row[9], "col_count": row[10]},
        }

        with conn.cursor() as cur:
            cur.execute(
                "SELECT step_name, description, params, result_summary, logged_at "
                "FROM run_steps WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            run["steps"] = [
                {
                    "step_name": r[0], "description": r[1],
                    "params": r[2], "result_summary": r[3], "logged_at": r[4],
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                "SELECT finding, metric, source_hook, method, confidence "
                "FROM run_insights WHERE run_id = %s ORDER BY id",
                (run_id,),
            )
            run["insights"] = [
                {
                    "finding": r[0], "metric": r[1], "source_hook": r[2],
                    "method": r[3], "confidence": r[4],
                }
                for r in cur.fetchall()
            ]

        return _dumps(run)

    return _dumps({"error": f"Unknown resource URI: {uri_str}"})


# ============================================================
# ENTRY POINT
# ============================================================

_INIT_OPTIONS = InitializationOptions(
    server_name="analysis-memory",
    server_version="2.0.0",
    capabilities=server.get_capabilities(
        notification_options=NotificationOptions(),
        experimental_capabilities={},
    ),
)


async def main_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, _INIT_OPTIONS)


def main_sse(host: str = "localhost", port: int = 8702) -> None:
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.responses import JSONResponse
    import uvicorn

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, _INIT_OPTIONS)

    async def handle_health(request):
        pool_ok = False
        try:
            pool = _get_pool()
            conn = pool.getconn()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            pool.putconn(conn)
            pool_ok = True
        except Exception:
            pass
        return JSONResponse({
            "status": "healthy" if pool_ok else "degraded",
            "server": "analysis-memory",
            "version": "2.0.0",
            "transport": "sse",
            "pool_ok": pool_ok,
        })

    app = Starlette(
        routes=[
            Route("/health", handle_health),
            Route("/sse", handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis Memory MCP Server")
    parser.add_argument(
        "--transport", choices=["sse", "stdio"], default="sse",
        help="Transport mode (default: sse)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8702)
    args = parser.parse_args()

    if args.transport == "stdio":
        asyncio.run(main_stdio())
    else:
        main_sse(host=args.host, port=args.port)
