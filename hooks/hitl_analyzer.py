"""
SA6.5 HITL Analyzer — pre-SA5 helper (Phase 6.5).

Three phases per agents/hitl_analyzer.md:
    analyze_all(requests)              -> list[AnalyzedRequest]
    aggregate_for_presentation(items)  -> HITLPresentation
    review_responses(presentation, responses) -> ReviewResult

Keep pure / functional. No I/O, no LLM calls — this is structural.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


COMPLEXITY_TIME_SEC = {"simple": 15, "moderate": 45, "complex": 120}


@dataclass
class HITLRequest:
    """Raw HITL request emitted by an executing worker agent."""
    request_id: str
    agent_id: str                 # e.g. "AG10"
    nature: str                   # "structural_approval" | "data_fix_review" | "merge_conflict" | "rule_confirmation"
    entity_name: Optional[str]
    entity_type: Optional[str]
    dataset: Optional[str]
    project: Optional[str]
    description: str
    priority: str = "non_blocking"  # "blocking" | "non_blocking"
    proposed_change: Optional[dict] = None


@dataclass
class AnalyzedRequest:
    request_id: str
    agent_id: str
    nature: str
    what_needed: str
    response_complexity: str       # "simple" | "moderate" | "complex"
    entity_context: dict
    grouping_key: str
    estimated_user_time_sec: int
    priority: str


@dataclass
class HITLPresentation:
    blocking_queue: list[AnalyzedRequest] = field(default_factory=list)
    batched_sections: dict[str, list[AnalyzedRequest]] = field(default_factory=dict)
    total_estimated_time_sec: int = 0
    request_count: int = 0


@dataclass
class ReviewResult:
    resolved: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    needs_reask: list[dict] = field(default_factory=list)


# ---------- phase 1: analyze ----------
def _classify_complexity(req: HITLRequest) -> str:
    if req.proposed_change and isinstance(req.proposed_change, dict) and len(req.proposed_change) > 3:
        return "complex"
    if req.nature in ("merge_conflict", "structural_approval"):
        return "moderate"
    return "simple"


def _what_needed(req: HITLRequest) -> str:
    return {
        "structural_approval": f"Approve schema change on {req.entity_name or 'entity'}",
        "data_fix_review": f"Confirm data fix on {req.entity_name or 'records'}",
        "merge_conflict": f"Choose canonical record for {req.entity_name or 'entity'}",
        "rule_confirmation": f"Confirm DQ rule for {req.entity_name or 'field'}",
    }.get(req.nature, req.description[:80])


def analyze_one(req: HITLRequest) -> AnalyzedRequest:
    complexity = _classify_complexity(req)
    return AnalyzedRequest(
        request_id=req.request_id,
        agent_id=req.agent_id,
        nature=req.nature,
        what_needed=_what_needed(req),
        response_complexity=complexity,
        entity_context={
            "entity_name": req.entity_name,
            "entity_type": req.entity_type,
            "dataset": req.dataset,
            "project": req.project,
        },
        grouping_key=f"{req.nature}_{req.entity_name or 'unspecified'}",
        estimated_user_time_sec=COMPLEXITY_TIME_SEC[complexity],
        priority=req.priority,
    )


def analyze_all(requests: list[HITLRequest]) -> list[AnalyzedRequest]:
    return [analyze_one(r) for r in requests]


# ---------- phase 2: aggregate ----------
_COMPLEXITY_RANK = {"simple": 0, "moderate": 1, "complex": 2}


def aggregate_for_presentation(analyzed: list[AnalyzedRequest]) -> HITLPresentation:
    pres = HITLPresentation()
    blocking = [a for a in analyzed if a.priority == "blocking"]
    non_blocking = [a for a in analyzed if a.priority != "blocking"]

    # blocking: keep insertion order (one-at-a-time per SA5 Rule 1)
    pres.blocking_queue = blocking

    # non-blocking: group by nature, sort within each group simple-first
    grouped: dict[str, list[AnalyzedRequest]] = {}
    for a in non_blocking:
        grouped.setdefault(a.nature, []).append(a)
    for k, lst in grouped.items():
        lst.sort(key=lambda a: (_COMPLEXITY_RANK[a.response_complexity], a.entity_context.get("entity_name") or ""))
    pres.batched_sections = grouped

    pres.request_count = len(analyzed)
    pres.total_estimated_time_sec = sum(a.estimated_user_time_sec for a in analyzed)
    return pres


# ---------- phase 3: review ----------
def review_responses(presentation: HITLPresentation, responses: dict[str, Any]) -> ReviewResult:
    """`responses`: {request_id: response_value}. Empty/None response -> unresolved."""
    result = ReviewResult()
    all_items = list(presentation.blocking_queue) + [a for items in presentation.batched_sections.values() for a in items]
    for item in all_items:
        resp = responses.get(item.request_id)
        if resp is None or (isinstance(resp, str) and not resp.strip()):
            result.unresolved.append({"request_id": item.request_id, "agent_id": item.agent_id, "reason": "no_response"})
        elif isinstance(resp, str) and resp.strip().lower() in ("unsure", "skip", "?", "i don't know"):
            result.needs_reask.append({"request_id": item.request_id, "agent_id": item.agent_id, "user_text": resp})
        else:
            result.resolved.append({"request_id": item.request_id, "agent_id": item.agent_id, "value": resp})
    return result


# ---------- helpers for downstream ----------
def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        return d
    return obj


# ---------- persistent HITL queue (WI-0B, OB1 pattern: schema-aware-routing) ----------
import json
import logging as _logging

_log = _logging.getLogger(__name__)

_PENDING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pending_confirmations (
    id              SERIAL PRIMARY KEY,
    request_id      TEXT UNIQUE NOT NULL,
    agent_id        TEXT,
    nature          TEXT,
    entity_name     TEXT,
    entity_type     TEXT,
    dataset         TEXT,
    project         TEXT,
    description     TEXT,
    priority        TEXT DEFAULT 'non_blocking',
    proposed_change JSONB,
    analysis_json   JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    resolved_value  JSONB,
    status          TEXT DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_pending_status_priority
    ON pending_confirmations (status, priority);
CREATE INDEX IF NOT EXISTS idx_pending_project_status
    ON pending_confirmations (project, status);
"""


def init_pending_confirmations(conn) -> None:
    """Create the pending_confirmations table (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(_PENDING_TABLE_SQL)
    conn.commit()


def persist_presentation(conn, presentation: HITLPresentation) -> int:
    """Persist all items from a HITLPresentation to pending_confirmations.

    Returns count of rows inserted (skips duplicates via ON CONFLICT).
    """
    all_items = list(presentation.blocking_queue)
    for items in presentation.batched_sections.values():
        all_items.extend(items)

    if not all_items:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for item in all_items:
            try:
                cur.execute("""
                    INSERT INTO pending_confirmations
                        (request_id, agent_id, nature, entity_name, entity_type,
                         dataset, project, description, priority, analysis_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id) DO NOTHING
                """, (
                    item.request_id, item.agent_id, item.nature,
                    item.entity_context.get("entity_name"),
                    item.entity_context.get("entity_type"),
                    item.entity_context.get("dataset"),
                    item.entity_context.get("project"),
                    item.what_needed, item.priority,
                    json.dumps(asdict(item)),
                ))
                inserted += cur.rowcount
            except Exception as exc:
                _log.warning("persist_presentation: skip %s — %s", item.request_id, exc)
    conn.commit()
    return inserted


def load_pending(conn, project: str | None = None, limit: int = 50) -> HITLPresentation:
    """Load pending HITL items from the database and reconstruct a HITLPresentation."""
    with conn.cursor() as cur:
        if project:
            cur.execute(
                "SELECT analysis_json FROM pending_confirmations "
                "WHERE status = 'pending' AND project = %s "
                "ORDER BY priority DESC, created_at LIMIT %s",
                (project, limit),
            )
        else:
            cur.execute(
                "SELECT analysis_json FROM pending_confirmations "
                "WHERE status = 'pending' "
                "ORDER BY priority DESC, created_at LIMIT %s",
                (limit,),
            )
        rows = cur.fetchall()

    analyzed: list[AnalyzedRequest] = []
    for (ajson,) in rows:
        d = ajson if isinstance(ajson, dict) else json.loads(ajson)
        analyzed.append(AnalyzedRequest(**{
            k: d[k] for k in AnalyzedRequest.__dataclass_fields__
            if k in d
        }))

    return aggregate_for_presentation(analyzed)


def fuzzy_entity_check(
    conn,
    entity_name: str,
    entity_type: str = "",
    threshold: float = 0.75,
    agent_id: str = "AG03",
    project: str = "",
) -> HITLRequest | None:
    """Check if an entity name has ambiguous near-matches in the knowledge graph.

    If multiple candidates are found above threshold, creates and returns a
    HITLRequest with nature='entity_confirmation' for human disambiguation.
    Returns None if the entity is unambiguous (0 or 1 match).

    WI-1D: OB1 pattern from schema-aware-routing — fuzzy entity resolution
    with human confirmation queue.
    """
    try:
        from hooks.intent_graph import find_similar_entities
    except ImportError:
        return None

    candidates = find_similar_entities(conn, entity_name, entity_type,
                                        top_k=5, min_score=threshold)

    # Filter out exact self-match
    candidates = [c for c in candidates if c["name"].lower() != entity_name.lower()]

    if len(candidates) == 0:
        return None  # Unambiguous — no similar entities

    # Multiple candidates found — create HITL request
    import uuid
    return HITLRequest(
        request_id=f"entity_{uuid.uuid4().hex[:8]}",
        agent_id=agent_id,
        nature="entity_confirmation",
        entity_name=entity_name,
        entity_type=entity_type,
        dataset=None,
        project=project,
        description=(
            f"Entity '{entity_name}' has {len(candidates)} similar match(es) in the knowledge graph: "
            + ", ".join(f"'{c['name']}' (score={c['score']:.3f})" for c in candidates[:3])
            + ". Please confirm which entity is intended, or if this is a new entity."
        ),
        priority="non_blocking",
        proposed_change={"candidates": candidates, "original_name": entity_name},
    )


def resolve_pending(conn, request_id: str, value: Any) -> None:
    """Mark a pending confirmation as resolved."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE pending_confirmations "
            "SET status = 'resolved', resolved_value = %s, resolved_at = NOW() "
            "WHERE request_id = %s AND status = 'pending'",
            (json.dumps(value) if not isinstance(value, str) else json.dumps(value), request_id),
        )
    conn.commit()
