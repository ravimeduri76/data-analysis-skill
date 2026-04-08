"""
seed_careaxl_domain.py — Seed 4 CareAXL healthcare domain models.

Run (idempotent):
    python hooks/seed_careaxl_domain.py

Models seeded
-------------
- Healthcare / Revenue Cycle / RCM
- Healthcare / Clinical Ops / Care Risk
- Healthcare / Master Data / Patient 360
- Healthcare / Preventative Care / Wellness
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.domain_graph import get_domain_conn, init_domain_db, save_domain_model

_DOCS = os.path.expanduser("~/Documents")

_MODELS: list[tuple[str, str, str]] = [
    # (file, app_id, solution_id)
    (os.path.join(_DOCS, "Healthcare_RCM_domain_model.json"), "careaxl", "revenue_cycle_management"),
    (os.path.join(_DOCS, "Healthcare_CareRisk_domain_model.json"), "careaxl", "care_risk_management"),
    (os.path.join(_DOCS, "Healthcare_Patient360_domain_model.json"), "careaxl", "patient_360"),
    (os.path.join(_DOCS, "Healthcare_Wellness_domain_model.json"), "careaxl", "wellness_preventative_health"),
]


def _already_exists(conn, domain: str, category: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM domain_models WHERE domain=%s AND category=%s LIMIT 1",
            (domain, category),
        )
        return cur.fetchone() is not None


def seed() -> None:
    conn = get_domain_conn()
    init_domain_db(conn)
    count = 0

    for path, app_id, solution_id in _MODELS:
        if not os.path.exists(path):
            print(f"  SKIP  {path} (file not found)")
            continue

        with open(path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        domain = model_json.get("domain_overview", {}).get("domain", "")
        category = model_json.get("domain_overview", {}).get("category", "")

        if _already_exists(conn, domain, category):
            print(f"  SKIP  {domain}/{category} already in DB")
            continue

        model_id = save_domain_model(conn, model_json, app_id=app_id, solution_id=solution_id)
        print(f"  OK    {domain}/{category} -> model_id={model_id}  (app={app_id}, sol={solution_id})")
        count += 1

    conn.close()
    print(f"Seed complete. {count} models seeded.")


if __name__ == "__main__":
    seed()
