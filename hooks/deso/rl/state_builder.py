"""Build StateSnapshot from DESORunner context packets.

Reads from runner._packets (DCP, IS, AP), runner._completed_steps,
and runner._adaptive_state to construct the RL state representation.
"""

from __future__ import annotations

import logging
from typing import Any

from hooks.deso.rl.rl_state import StateSnapshot

log = logging.getLogger(__name__)


def build_state_snapshot(
    runner: Any,
    step_label: str,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
) -> StateSnapshot:
    """Construct a StateSnapshot from the current DESORunner state.

    Args:
        runner: DESORunner instance.
        step_label: Current step (e.g. "SA3", "SA4").
        latency_ms: Cumulative latency so far.
        cost_usd: Cumulative API cost so far.

    Returns:
        Frozen StateSnapshot capturing current orchestration state.
    """
    packets = getattr(runner, "_packets", {})
    dcp = packets.get("DCP", {})
    is_spec = packets.get("IS", {})
    ap = packets.get("AP", {})

    # Extract domain info from DCP
    domain = dcp.get("domain", "")
    lifecycle_stage = dcp.get("lifecycle_stage", "")

    # Dataset dimensions from DCP extras or AP
    dataset_rows = dcp.get("extras", {}).get("dataset_rows", 0)
    dataset_cols = dcp.get("extras", {}).get("dataset_cols", 0)
    avg_null_pct = dcp.get("extras", {}).get("avg_null_pct", 0.0)

    # Agents already called from AP
    agents_called = []
    if ap.get("agents"):
        agents_called = [
            a.get("agent_id", "") for a in ap["agents"] if a.get("agent_id")
        ]

    # LLM used from IS or AP extras
    llm_used = is_spec.get("llm_used", "") or ap.get("llm_used", "")

    # Steps completed
    completed_steps = getattr(runner, "_completed_steps", set())
    steps_completed = len(completed_steps)

    # Prior quality scores from adaptive state
    prior_quality_scores: list[float] = []
    adaptive_state = getattr(runner, "_adaptive_state", None)
    if adaptive_state is not None:
        step_states = {}
        if hasattr(adaptive_state, "step_states"):
            step_states = adaptive_state.step_states
        elif hasattr(adaptive_state, "to_dict"):
            step_states = adaptive_state.to_dict().get("step_states", {})
        for _so, ss in step_states.items():
            quality = ss if isinstance(ss, dict) else (ss.to_dict() if hasattr(ss, "to_dict") else {})
            overall = quality.get("quality", {}).get("overall") if isinstance(quality.get("quality"), dict) else None
            if overall is not None:
                prior_quality_scores.append(float(overall))

    # Run ID from session
    session = getattr(runner, "_session", None)
    run_id = ""
    if session:
        run_id = getattr(session, "run_id", "") or getattr(session, "session_id", "")
    if not run_id:
        run_id = getattr(runner, "user_query", "")[:50]

    return StateSnapshot(
        run_id=run_id,
        step_label=step_label,
        domain=domain,
        lifecycle_stage=lifecycle_stage,
        dataset_rows=dataset_rows,
        dataset_cols=dataset_cols,
        avg_null_pct=avg_null_pct,
        agents_called=agents_called,
        llm_used=llm_used,
        steps_completed=steps_completed,
        prior_quality_scores=prior_quality_scores,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def embed_state(state: StateSnapshot) -> StateSnapshot:
    """Best-effort: add domain and query embeddings via embedding_registry.

    Returns the same state with embeddings populated, or unchanged if
    embedding_registry is unavailable.
    """
    try:
        from hooks.embedding_registry import embed_texts

        if state.domain and not state.domain_embedding:
            vecs = embed_texts([state.domain])
            state.domain_embedding = vecs[0].tolist()

        query_text = f"{state.step_label} {state.domain} {state.lifecycle_stage}"
        if query_text.strip() and not state.query_embedding:
            vecs = embed_texts([query_text])
            state.query_embedding = vecs[0].tolist()
    except Exception as exc:
        log.debug("embed_state skipped (non-fatal): %s", exc)

    return state
