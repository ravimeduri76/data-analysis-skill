"""
seed_plans.py — Seed DataAXL-specific execution plans.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


PLANS = [
    {
        "name": "product_data_prep_pipeline",
        "label": "Product Data Prep Pipeline",
        "description": "Full product data prep: ingest from Syndigo/Stibo, harmonize schemas, recommend structure, enrich attributes",
        "intent_patterns": ["product_data_prep", "syndigo_ingest", "stibo_ingest", "product_harmonize", "product_enrich"],
        "lifecycle_stages": ["preparation", "enrichment"],
        "domains": ["Commercial / Industrial"],
        "requires_dataset": True,
        "solution_id": "product_data_prep",
    },
    {
        "name": "product_syndication_pipeline",
        "label": "Product Syndication Pipeline",
        "description": "Distribute enriched product data to Amazon, GDSN, and Walmart endpoints",
        "intent_patterns": ["product_data_syndication", "amazon_syndication", "gdsn_syndication", "walmart_syndication"],
        "lifecycle_stages": ["publishing"],
        "domains": ["Commercial / Industrial"],
        "requires_dataset": True,
        "solution_id": "product_data_syndication",
    },
    {
        "name": "customer_data_prep_pipeline",
        "label": "Customer Data Prep Pipeline",
        "description": "Full customer data prep: ingest from SAP/JDE, search & match, golden record, address enrichment",
        "intent_patterns": ["customer_data_prep", "sap_customer_ingest", "jde_customer_ingest", "customer_search_match", "customer_golden_record", "address_enrichment"],
        "lifecycle_stages": ["preparation", "enrichment"],
        "domains": ["Master Data Management"],
        "requires_dataset": True,
        "solution_id": "customer_data_prep",
    },
]

STEPS = [
    # Product Data Prep
    {"plan_name": "product_data_prep_pipeline", "step_order": 1, "agent_id": "AG04", "purpose": "Ingest product data from source system (Syndigo/Stibo)"},
    {"plan_name": "product_data_prep_pipeline", "step_order": 2, "agent_id": "AG07", "purpose": "Full profile of ingested product data"},
    {"plan_name": "product_data_prep_pipeline", "step_order": 3, "agent_id": "AG08", "purpose": "Recommend unified product schema"},
    {"plan_name": "product_data_prep_pipeline", "step_order": 4, "agent_id": "AG09", "purpose": "Harmonize source schemas to recommended structure"},
    {"plan_name": "product_data_prep_pipeline", "step_order": 5, "agent_id": "AG10", "purpose": "Enforce schema constraints", "hitl_gate": "before", "hitl_reason": "Structural DDL changes require confirmation"},
    {"plan_name": "product_data_prep_pipeline", "step_order": 6, "agent_id": "AG25", "purpose": "Enrich product attributes from external sources"},
    # Product Syndication
    {"plan_name": "product_syndication_pipeline", "step_order": 1, "agent_id": "AG04", "purpose": "Load enriched product data"},
    {"plan_name": "product_syndication_pipeline", "step_order": 2, "agent_id": "AG34", "purpose": "Manage product catalog for target endpoints"},
    {"plan_name": "product_syndication_pipeline", "step_order": 3, "agent_id": "AG35", "purpose": "Syndicate to target endpoints", "hitl_gate": "before", "hitl_reason": "External data push requires confirmation"},
    # Customer Data Prep
    {"plan_name": "customer_data_prep_pipeline", "step_order": 1, "agent_id": "AG04", "purpose": "Ingest customer data from SAP/JDE"},
    {"plan_name": "customer_data_prep_pipeline", "step_order": 2, "agent_id": "AG07", "purpose": "Full profile of customer data"},
    {"plan_name": "customer_data_prep_pipeline", "step_order": 3, "agent_id": "AG23", "purpose": "Build customer ontology for matching"},
    {"plan_name": "customer_data_prep_pipeline", "step_order": 4, "agent_id": "AG27", "purpose": "Customer 360 — search and match across sources"},
    {"plan_name": "customer_data_prep_pipeline", "step_order": 5, "agent_id": "AG28", "purpose": "Golden record management", "hitl_gate": "before", "hitl_reason": "Writing golden records requires confirmation"},
    {"plan_name": "customer_data_prep_pipeline", "step_order": 6, "agent_id": "AG25", "purpose": "Address enrichment from external databases"},
]

DEPENDENCIES = [
    # Product Data Prep
    {"plan_name": "product_data_prep_pipeline", "from_agent_id": "AG04", "to_agent_id": "AG07", "dep_type": "sequence"},
    {"plan_name": "product_data_prep_pipeline", "from_agent_id": "AG07", "to_agent_id": "AG08", "dep_type": "sequence"},
    {"plan_name": "product_data_prep_pipeline", "from_agent_id": "AG08", "to_agent_id": "AG09", "dep_type": "sequence"},
    {"plan_name": "product_data_prep_pipeline", "from_agent_id": "AG09", "to_agent_id": "AG10", "dep_type": "sequence"},
    {"plan_name": "product_data_prep_pipeline", "from_agent_id": "AG10", "to_agent_id": "AG25", "dep_type": "sequence"},
    # Product Syndication
    {"plan_name": "product_syndication_pipeline", "from_agent_id": "AG04", "to_agent_id": "AG34", "dep_type": "sequence"},
    {"plan_name": "product_syndication_pipeline", "from_agent_id": "AG34", "to_agent_id": "AG35", "dep_type": "sequence"},
    # Customer Data Prep
    {"plan_name": "customer_data_prep_pipeline", "from_agent_id": "AG04", "to_agent_id": "AG07", "dep_type": "sequence"},
    {"plan_name": "customer_data_prep_pipeline", "from_agent_id": "AG07", "to_agent_id": "AG23", "dep_type": "sequence"},
    {"plan_name": "customer_data_prep_pipeline", "from_agent_id": "AG23", "to_agent_id": "AG27", "dep_type": "sequence"},
    {"plan_name": "customer_data_prep_pipeline", "from_agent_id": "AG27", "to_agent_id": "AG28", "dep_type": "sequence"},
    {"plan_name": "customer_data_prep_pipeline", "from_agent_id": "AG28", "to_agent_id": "AG25", "dep_type": "sequence"},
]


def seed_plans(conn, app_id: str = "dataaxl") -> int:
    """Seed DataAXL execution plans. Returns plan count."""
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
    print(f"Seeded {n} DataAXL execution plans.")
