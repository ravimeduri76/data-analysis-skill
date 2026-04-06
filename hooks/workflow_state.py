"""
workflow_state.py — Checkpoint persistence for DESO workflow durability.

Allows the orchestrator to save progress after each SA step and resume
an interrupted run from the last successful checkpoint.

Requires the workflow_checkpoints table (added in db_migrate.py).

Public API
----------
save_checkpoint(conn, run_id, step_name, step_index, status, context) -> None
get_last_checkpoint(conn, run_id) -> dict | None
can_resume(conn, run_id) -> tuple[bool, int]
list_checkpoints(conn, run_id) -> list[dict]
mark_run_complete(conn, run_id) -> None
"""

from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger(__name__)


def save_checkpoint(
    conn,
    run_id: str,
    step_name: str,
    step_index: int,
    status: str,
    context: dict,
) -> None:
    """Write or update a checkpoint for a workflow step.

    Parameters
    ----------
    conn : psycopg2 connection
    run_id : str
        Unique run identifier.
    step_name : str
        Step identifier (e.g. 'SA1', 'SA2', 'SA5_hitl_1').
    step_index : int
        Ordinal position in the orchestration sequence.
    status : str
        One of 'completed', 'failed', 'in_progress'.
    context : dict
        Context packet produced by this step (DCP, IS, AP, etc.).
    """
    sql = """
        INSERT INTO workflow_checkpoints (run_id, step_name, step_index, status, context_json)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id, step_name)
        DO UPDATE SET status = EXCLUDED.status,
                      context_json = EXCLUDED.context_json,
                      created_at = NOW()
    """
    try:
        context_json = json.dumps(context, default=str)
        with conn.cursor() as cur:
            cur.execute(sql, (run_id, step_name, step_index, status, context_json))
        conn.commit()
    except Exception as exc:
        log.warning("Failed to save checkpoint %s/%s: %s", run_id, step_name, exc)
        try:
            conn.rollback()
        except Exception:
            pass


def get_last_checkpoint(conn, run_id: str) -> Optional[dict]:
    """Return the most recent checkpoint for a run, or None.

    Returns
    -------
    dict with keys: step_name, step_index, status, context_json, created_at
    """
    sql = """
        SELECT step_name, step_index, status, context_json, created_at
        FROM workflow_checkpoints
        WHERE run_id = %s
        ORDER BY step_index DESC
        LIMIT 1
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "step_name": row[0],
            "step_index": row[1],
            "status": row[2],
            "context_json": row[3],
            "created_at": row[4],
        }
    except Exception as exc:
        log.warning("Failed to get checkpoint for %s: %s", run_id, exc)
        return None


def can_resume(conn, run_id: str) -> tuple[bool, int]:
    """Check if a run can be resumed from a prior checkpoint.

    Returns
    -------
    (resumable, resume_from_step)
        resumable is True if there is a completed checkpoint but the run
        is not fully complete. resume_from_step is the step_index to resume from.
    """
    last = get_last_checkpoint(conn, run_id)
    if last is None:
        return (False, 0)

    # If last step failed or is in_progress, resume from that step
    if last["status"] in ("failed", "in_progress"):
        return (True, last["step_index"])

    # If last step completed but it's not the final step (step 8 / SA6),
    # resume from the next step
    if last["status"] == "completed" and last["step_index"] < 8:
        return (True, last["step_index"] + 1)

    # Run is fully complete
    return (False, 0)


def list_checkpoints(conn, run_id: str) -> list[dict]:
    """Return all checkpoints for a run, ordered by step_index."""
    sql = """
        SELECT step_name, step_index, status, context_json, created_at
        FROM workflow_checkpoints
        WHERE run_id = %s
        ORDER BY step_index ASC
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (run_id,))
            rows = cur.fetchall()
        return [
            {
                "step_name": r[0],
                "step_index": r[1],
                "status": r[2],
                "context_json": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
    except Exception as exc:
        log.warning("Failed to list checkpoints for %s: %s", run_id, exc)
        return []


def mark_run_complete(conn, run_id: str) -> None:
    """Mark the final checkpoint as completed (convenience for SA6)."""
    save_checkpoint(conn, run_id, "SA6_complete", 8, "completed", {"status": "done"})
