"""
degradation_tracker.py — Collects degradation events during a DESO run.

SA6 includes the tracker's output in the Final Response under
"Limitations & Degradations".

Public API
----------
DegradationTracker()
    .record(subsystem, message, impact)
    .has_events -> bool
    .render() -> str
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List


class DegradationTracker:
    """Accumulates degradation events across a single DESO run."""

    def __init__(self) -> None:
        self.events: List[dict] = []

    def record(self, subsystem: str, message: str, impact: str) -> None:
        """Record a degradation event.

        Parameters
        ----------
        subsystem : str
            Component that degraded (e.g. 'llm_client', 'domain_graph', 'semantic_store').
        message : str
            What happened (e.g. 'Groq unavailable, fell back to Anthropic').
        impact : str
            User-facing impact (e.g. 'Domain detection used heuristic — lower accuracy').
        """
        self.events.append({
            "subsystem": subsystem,
            "message": message,
            "impact": impact,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def has_events(self) -> bool:
        return len(self.events) > 0

    def render(self) -> str:
        """Return an ASCII-safe summary of all degradation events."""
        if not self.events:
            return "No degradations encountered."
        lines = ["Limitations encountered during this run:"]
        for i, evt in enumerate(self.events, 1):
            lines.append(
                f"  {i}. [{evt['subsystem']}] {evt['message']}"
            )
            lines.append(f"     Impact: {evt['impact']}")
        result = "\n".join(lines)
        return result.encode("ascii", errors="replace").decode("ascii")

    def to_list(self) -> List[dict]:
        """Return raw events for JSON serialization."""
        return list(self.events)
