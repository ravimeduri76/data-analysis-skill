"""
seed_domain.py — Seed the domain knowledge graph with example domain models.

Run once (or re-run safely — checks for existing domain before inserting):

    python hooks/seed_domain.py

Models seeded
-------------
- Industrial Component / Electrical Equipment / Relays  (from Relays_domain_model.json)
"""

from __future__ import annotations

import json
import os
import sys

# Allow running from the project root or from hooks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.domain_graph import get_domain_conn, init_domain_db, save_domain_model

_MODELS = [
    os.path.join(os.path.expanduser("~"), "Documents", "Relays_domain_model.json"),
    os.path.join(os.path.expanduser("~"), "Documents", "Healthcare_MedicationManagement_domain_model.json"),
    os.path.join(os.path.expanduser("~"), "Documents", "Ecommerce_domain_model.json"),
    os.path.join(os.path.expanduser("~"), "Documents", "TravelHospitality_domain_model.json"),
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

    for path in _MODELS:
        if not os.path.exists(path):
            print(f"  SKIP  {path} (file not found)")
            continue

        with open(path, "r", encoding="utf-8") as f:
            model_json = json.load(f)

        domain   = model_json.get("domain_overview", {}).get("domain", "")
        category = model_json.get("domain_overview", {}).get("category", "")

        if _already_exists(conn, domain, category):
            print(f"  SKIP  {domain}/{category} already in DB")
            continue

        model_id = save_domain_model(conn, model_json)
        print(f"  OK    {domain}/{category} -> model_id={model_id}")

    conn.close()
    print("Seed complete.")


if __name__ == "__main__":
    seed()
