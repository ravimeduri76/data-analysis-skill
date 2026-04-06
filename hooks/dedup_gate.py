"""
dedup_gate.py — Check for prior analysis of the same dataset before running SA1.

Uses the analysis_runs table (from db_store.py) to find prior runs on the
same file (by SHA-256 hash) within a configurable lookback window.

Public API
----------
check_prior_analysis(conn, dataset_sha256, days) -> dict | None
render_prior_analysis(prior) -> str
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def check_prior_analysis(
    conn,
    dataset_sha256: str,
    days: int = 7,
) -> Optional[dict]:
    """Check if this dataset was analyzed recently.

    Parameters
    ----------
    conn : psycopg2 connection
    dataset_sha256 : str
        SHA-256 hash of the dataset file.
    days : int
        Lookback window in days (default 7).

    Returns
    -------
    dict with prior run metadata if found, None otherwise.
    Keys: run_id, dataset_path, domain, findings_summary, created_at
    """
    if not dataset_sha256 or dataset_sha256 == "unavailable":
        return None

    sql = """
        SELECT run_id, dataset_path, domain, findings_summary, created_at
        FROM analysis_runs
        WHERE file_sha256 = %s
          AND created_at > NOW() - INTERVAL '%s days'
        ORDER BY created_at DESC
        LIMIT 1
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (dataset_sha256, days))
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "dataset_path": row[1],
            "domain": row[2],
            "findings_summary": row[3],
            "created_at": row[4],
        }
    except Exception as exc:
        log.warning("Dedup gate check failed: %s", exc)
        return None


def render_prior_analysis(prior: dict) -> str:
    """Format a prior analysis result for user display."""
    date_str = str(prior.get("created_at", "unknown"))[:19]
    lines = [
        f"Prior analysis found for this dataset (run {prior['run_id']}, {date_str}).",
        f"Domain: {prior.get('domain', 'unknown')}",
        f"Findings: {prior.get('findings_summary', 'N/A')[:200]}",
        "",
        "Options: Review prior findings | Rerun analysis",
    ]
    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
