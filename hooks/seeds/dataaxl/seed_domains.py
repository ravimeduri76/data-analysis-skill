"""
seed_domains.py — Seed DataAXL-specific domain models.

Loads the Ecommerce domain model and registers it to app_id='dataaxl'.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def seed_domains(conn, app_id: str = "dataaxl") -> int:
    """Seed domain models for DataAXL. Returns count of models seeded."""
    from domain_graph import init_domain_db, load_model_from_file

    init_domain_db(conn)

    models_dir = os.path.join(os.path.expanduser("~"), "Documents")
    seeded = 0

    # Ecommerce domain model — primary model for DataAXL product solutions
    ecommerce_path = os.path.join(models_dir, "Ecommerce_domain_model.json")
    if os.path.exists(ecommerce_path):
        model_id = load_model_from_file(conn, ecommerce_path, app_id=app_id)
        if model_id:
            seeded += 1
            print(f"  Seeded: Ecommerce domain model (model_id={model_id}, app_id={app_id})")

    return seeded


if __name__ == "__main__":
    from domain_graph import get_domain_conn
    conn = get_domain_conn()
    n = seed_domains(conn)
    conn.close()
    print(f"Seeded {n} domain models for dataaxl.")
