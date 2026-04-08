"""
seed_agents.py — Seed Base application agent cards (AG01-AG39).

Delegates to the existing seed_execution_graph.py for agent card data,
but passes app_id='base' to all upsert calls.
"""

from __future__ import annotations

import os
import sys

# Ensure hooks directory is importable
_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def seed_agents(conn, app_id: str = "base") -> int:
    """Seed agent cards for the given app_id. Returns count of agents seeded.

    Imports the agent data from the existing seed_execution_graph.py module
    and re-inserts with the app_id parameter.
    """
    from execution_graph import init_execution_db, upsert_agent_card

    init_execution_db(conn)

    # Import agent definitions from the original seed script
    from seed_execution_graph import AGENTS

    count = 0
    for agent in AGENTS:
        upsert_agent_card(
            conn,
            agent_id=agent["agent_id"],
            name=agent["name"],
            type_=agent.get("type", "worker"),
            lifecycle_stage=agent.get("lifecycle_stage"),
            domain=agent.get("domain", "Any"),
            capability=agent.get("capability"),
            skill_workflow=agent.get("skill_workflow"),
            ingress_schema=agent.get("ingress_schema"),
            egress_schema=agent.get("egress_schema"),
            runtime_hint=agent.get("runtime_hint", "sync"),
            hitl_required=agent.get("hitl_required", False),
            notes=agent.get("notes"),
            app_id=app_id,
        )
        count += 1
    conn.commit()
    return count


if __name__ == "__main__":
    from execution_graph import get_execution_conn
    conn = get_execution_conn()
    n = seed_agents(conn)
    conn.close()
    print(f"Seeded {n} agent cards for base.")
