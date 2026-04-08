"""
seed_intents.py — Seed CareAXL-specific intent nodes.

Probus and Reema module intents for healthcare operations and community health.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


INTENTS = [
    # Abstract
    {
        "name": "healthcare_operations",
        "label": "Healthcare Operations",
        "description": "Manage healthcare operational data pipelines, analytics, and workflows",
        "level": "abstract",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
    # Mid — Probus
    {
        "name": "probus_pipeline",
        "label": "Probus Data Pipeline",
        "description": "Clinical operations data pipeline for Probus module",
        "level": "mid",
        "domain": "Healthcare",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "probus_analytics",
        "label": "Probus Analytics",
        "description": "Clinical operations analytics — readmission, LOS, quality metrics",
        "level": "mid",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
    # Mid — Reema
    {
        "name": "reema_pipeline",
        "label": "Reema Data Pipeline",
        "description": "Community health and SDOH data pipeline for Reema module",
        "level": "mid",
        "domain": "Healthcare",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "reema_analytics",
        "label": "Reema Analytics",
        "description": "Social determinants of health analytics — community needs, care gaps",
        "level": "mid",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
    # Specific — Probus
    {
        "name": "clinical_profiling",
        "label": "Clinical Data Profiling",
        "description": "Profile clinical datasets for data quality, completeness, and patterns",
        "level": "specific",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
    {
        "name": "readmission_analysis",
        "label": "Readmission Analysis",
        "description": "Analyze and predict 30-day readmission patterns",
        "level": "specific",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
    # Specific — Reema
    {
        "name": "sdoh_screening",
        "label": "SDOH Screening",
        "description": "Screen patients for social determinants of health risk factors",
        "level": "specific",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
    {
        "name": "community_needs_assessment",
        "label": "Community Needs Assessment",
        "description": "Assess community health needs using population data and SDOH indicators",
        "level": "specific",
        "domain": "Healthcare",
        "lifecycle_stage": "analytics",
    },
]

TRIPLETS = [
    ("probus_pipeline", "IS_A", "healthcare_operations"),
    ("probus_analytics", "IS_A", "healthcare_operations"),
    ("reema_pipeline", "IS_A", "healthcare_operations"),
    ("reema_analytics", "IS_A", "healthcare_operations"),
    ("clinical_profiling", "IS_A", "probus_analytics"),
    ("readmission_analysis", "IS_A", "probus_analytics"),
    ("sdoh_screening", "IS_A", "reema_analytics"),
    ("community_needs_assessment", "IS_A", "reema_analytics"),
    ("probus_pipeline", "PRECEDES", "probus_analytics"),
    ("reema_pipeline", "PRECEDES", "reema_analytics"),
    ("clinical_profiling", "ENABLES", "readmission_analysis"),
]

ALIASES = [
    ("probus_pipeline", "probus"),
    ("probus_analytics", "clinical analytics"),
    ("reema_pipeline", "reema"),
    ("reema_analytics", "community health analytics"),
    ("sdoh_screening", "sdoh"),
    ("sdoh_screening", "social determinants"),
    ("community_needs_assessment", "community needs"),
    ("readmission_analysis", "readmission"),
    ("clinical_profiling", "clinical profile"),
]


def seed_intents(conn, app_id: str = "careaxl") -> int:
    """Seed CareAXL intent nodes. Returns node count."""
    from intent_graph import (
        init_intent_db, upsert_intent_node, add_intent_triplet, add_intent_alias,
    )

    init_intent_db(conn)

    count = 0
    for intent in INTENTS:
        upsert_intent_node(
            conn,
            name=intent["name"],
            label=intent["label"],
            description=intent["description"],
            level=intent["level"],
            domain=intent["domain"],
            lifecycle_stage=intent.get("lifecycle_stage"),
            app_id=app_id,
        )
        count += 1
    conn.commit()

    for subj, pred, obj in TRIPLETS:
        try:
            add_intent_triplet(conn, subj, pred, obj)
        except Exception:
            pass
    conn.commit()

    for intent_name, alias in ALIASES:
        try:
            add_intent_alias(conn, intent_name, alias)
        except Exception:
            pass
    conn.commit()

    return count


if __name__ == "__main__":
    from intent_graph import get_intent_conn
    conn = get_intent_conn()
    n = seed_intents(conn)
    conn.close()
    print(f"Seeded {n} CareAXL intent nodes.")
