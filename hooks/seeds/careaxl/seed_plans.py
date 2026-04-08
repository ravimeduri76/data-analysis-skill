"""
seed_plans.py — Seed CareAXL-specific execution plans.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


PLANS = [
    {
        "name": "probus_analytics_pipeline",
        "label": "Probus Analytics Pipeline",
        "description": "Clinical operations analytics — ingest, profile, analyse readmission and LOS patterns",
        "intent_patterns": ["probus_pipeline", "probus_analytics", "clinical_profiling", "readmission_analysis"],
        "lifecycle_stages": ["preparation", "analytics"],
        "domains": ["Healthcare"],
        "requires_dataset": True,
        "solution_id": "probus",
    },
    {
        "name": "reema_analytics_pipeline",
        "label": "Reema Analytics Pipeline",
        "description": "Community health analytics — SDOH screening, community needs assessment, care gap analysis",
        "intent_patterns": ["reema_pipeline", "reema_analytics", "sdoh_screening", "community_needs_assessment"],
        "lifecycle_stages": ["preparation", "analytics"],
        "domains": ["Healthcare"],
        "requires_dataset": True,
        "solution_id": "reema",
    },
]

STEPS = [
    # Probus
    {"plan_name": "probus_analytics_pipeline", "step_order": 1, "agent_id": "AG04", "purpose": "Ingest clinical data"},
    {"plan_name": "probus_analytics_pipeline", "step_order": 2, "agent_id": "AG06", "purpose": "Initial clinical data profile"},
    {"plan_name": "probus_analytics_pipeline", "step_order": 3, "agent_id": "AG07", "purpose": "Full clinical data profile"},
    {"plan_name": "probus_analytics_pipeline", "step_order": 4, "agent_id": "AG14", "purpose": "Detect clinical anomalies"},
    {"plan_name": "probus_analytics_pipeline", "step_order": 5, "agent_id": "AG36", "purpose": "Readmission and LOS analytics"},
    # Reema
    {"plan_name": "reema_analytics_pipeline", "step_order": 1, "agent_id": "AG04", "purpose": "Ingest community health data"},
    {"plan_name": "reema_analytics_pipeline", "step_order": 2, "agent_id": "AG06", "purpose": "Initial SDOH data profile"},
    {"plan_name": "reema_analytics_pipeline", "step_order": 3, "agent_id": "AG07", "purpose": "Full SDOH data profile"},
    {"plan_name": "reema_analytics_pipeline", "step_order": 4, "agent_id": "AG14", "purpose": "Detect community health anomalies"},
    {"plan_name": "reema_analytics_pipeline", "step_order": 5, "agent_id": "AG36", "purpose": "Community needs analytics"},
]

DEPENDENCIES = [
    {"plan_name": "probus_analytics_pipeline", "from_agent_id": "AG04", "to_agent_id": "AG06", "dep_type": "sequence"},
    {"plan_name": "probus_analytics_pipeline", "from_agent_id": "AG06", "to_agent_id": "AG07", "dep_type": "sequence"},
    {"plan_name": "probus_analytics_pipeline", "from_agent_id": "AG07", "to_agent_id": "AG14", "dep_type": "sequence"},
    {"plan_name": "probus_analytics_pipeline", "from_agent_id": "AG14", "to_agent_id": "AG36", "dep_type": "sequence"},
    {"plan_name": "reema_analytics_pipeline", "from_agent_id": "AG04", "to_agent_id": "AG06", "dep_type": "sequence"},
    {"plan_name": "reema_analytics_pipeline", "from_agent_id": "AG06", "to_agent_id": "AG07", "dep_type": "sequence"},
    {"plan_name": "reema_analytics_pipeline", "from_agent_id": "AG07", "to_agent_id": "AG14", "dep_type": "sequence"},
    {"plan_name": "reema_analytics_pipeline", "from_agent_id": "AG14", "to_agent_id": "AG36", "dep_type": "sequence"},
]


def seed_plans(conn, app_id: str = "careaxl") -> int:
    """Seed CareAXL execution plans. Returns plan count."""
    from execution_graph import (
        init_execution_db, upsert_execution_plan, add_execution_step,
        add_execution_dependency,
    )

    init_execution_db(conn)

    count = 0
    for plan in PLANS:
        upsert_execution_plan(
            conn,
            name=plan["name"],
            label=plan["label"],
            description=plan["description"],
            intent_patterns=plan.get("intent_patterns"),
            lifecycle_stages=plan.get("lifecycle_stages"),
            domains=plan.get("domains"),
            requires_dataset=plan.get("requires_dataset", True),
            app_id=app_id,
            solution_id=plan.get("solution_id"),
        )
        count += 1
    conn.commit()

    for step in STEPS:
        try:
            add_execution_step(
                conn,
                plan_name=step["plan_name"],
                step_order=step["step_order"],
                agent_id=step["agent_id"],
                purpose=step.get("purpose"),
                hitl_gate=step.get("hitl_gate"),
                hitl_reason=step.get("hitl_reason"),
            )
        except Exception:
            pass
    conn.commit()

    for dep in DEPENDENCIES:
        try:
            add_execution_dependency(
                conn,
                plan_name=dep["plan_name"],
                from_agent_id=dep["from_agent_id"],
                to_agent_id=dep["to_agent_id"],
                dep_type=dep.get("dep_type", "sequence"),
            )
        except Exception:
            pass
    conn.commit()

    return count


if __name__ == "__main__":
    from execution_graph import get_execution_conn
    conn = get_execution_conn()
    n = seed_plans(conn)
    conn.close()
    print(f"Seeded {n} CareAXL execution plans.")
