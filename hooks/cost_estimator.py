"""
cost_estimator.py — Pre-execution cost and runtime estimation for DESO plans.

Public API
----------
estimate_cost(ap, dataset_size) -> dict
render_cost_estimate(estimate)  -> str
"""

from __future__ import annotations

from typing import Optional

# Per-agent cost heuristics (USD per invocation, approximate)
_AGENT_COST: dict[str, float] = {
    # LLM-heavy agents
    "AG06": 0.02,   # InitialProfiler (LLM routing)
    "AG07": 0.03,   # FullProfiler (multi-tool)
    "AG08": 0.02,   # SchemaRecommender
    "AG14": 0.01,   # AnomalyDetector
    "AG20": 0.02,   # FitForUse
    "AG24": 0.02,   # CoreOntologyBuilder
    "AG25": 0.05,   # DataEnricher.External (API calls)
    "AG29": 0.03,   # NBARecommender
    "AG30": 0.02,   # ContentGenerator
    "AG36": 0.02,   # DataAnalysis
    "AG37": 0.04,   # RAGAgent (embedding + index)
}
_DEFAULT_AGENT_COST = 0.01  # agents not listed above

# Per-agent runtime heuristics (seconds, base for small datasets)
_AGENT_RUNTIME: dict[str, int] = {
    "AG05": 5,    # Sampler
    "AG07": 20,   # FullProfiler
    "AG14": 15,   # AnomalyDetector
    "AG16": 30,   # DQFixer
    "AG25": 60,   # External enrichment
    "AG27": 10,   # X360
    "AG28": 20,   # GoldenDataManager
    "AG36": 30,   # DataAnalysis
    "AG37": 45,   # RAGAgent
}
_DEFAULT_AGENT_RUNTIME = 10


def estimate_cost(
    ap: dict,
    dataset_size: tuple[int, int],
    llm_calls_expected: int = 0,
) -> dict:
    """Estimate cost, runtime, and agent count from an Agent Plan.

    Parameters
    ----------
    ap : dict
        Agent Plan context packet (from SA3). Expected keys:
        - agents: list of dicts with at least 'agent_id'
    dataset_size : tuple[int, int]
        (row_count, col_count) of the primary dataset.
    llm_calls_expected : int
        Extra LLM calls beyond agent defaults (domain disambiguation, etc.)

    Returns
    -------
    dict with keys:
        estimated_api_cost   : float  (USD)
        estimated_runtime_sec: int
        agents               : int
        breakdown            : list[dict]  (per-agent detail)
    """
    agents = ap.get("agents", [])
    rows, cols = dataset_size

    # Scale factor: datasets > 50K rows increase runtime linearly
    scale = max(1.0, rows / 50_000) if rows > 0 else 1.0

    breakdown = []
    total_cost = 0.0
    total_runtime = 0

    for agent in agents:
        agent_id = agent.get("agent_id", "unknown")
        cost = _AGENT_COST.get(agent_id, _DEFAULT_AGENT_COST)
        runtime = int(_AGENT_RUNTIME.get(agent_id, _DEFAULT_AGENT_RUNTIME) * scale)
        total_cost += cost
        total_runtime += runtime
        breakdown.append({
            "agent_id": agent_id,
            "estimated_cost": round(cost, 4),
            "estimated_runtime_sec": runtime,
        })

    # Add LLM overhead costs
    llm_overhead = llm_calls_expected * 0.005  # ~$0.005 per call
    total_cost += llm_overhead

    return {
        "estimated_api_cost": round(total_cost, 4),
        "estimated_runtime_sec": total_runtime,
        "agents": len(agents),
        "breakdown": breakdown,
    }


def render_cost_estimate(est: dict) -> str:
    """Return an ASCII-safe summary string for user display."""
    lines = [
        f"Plan: {est['agents']} agents, "
        f"~${est['estimated_api_cost']:.2f} API cost, "
        f"~{_fmt_time(est['estimated_runtime_sec'])} runtime.",
    ]
    return lines[0].encode("ascii", errors="replace").decode("ascii")


def _fmt_time(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    remaining = seconds % 60
    if remaining == 0:
        return f"{minutes} min"
    return f"{minutes} min {remaining}s"
