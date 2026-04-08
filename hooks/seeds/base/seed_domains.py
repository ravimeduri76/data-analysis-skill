"""
seed_domains.py — Seed Base application domain models.

Delegates to existing seed_domain.py, passing app_id='base'.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def seed_domains(conn, app_id: str = "base") -> int:
    """Seed domain models for the given app_id. Returns count of models seeded."""
    from domain_graph import init_domain_db, load_model_from_file

    init_domain_db(conn)

    # Load from the standard Relays domain model
    model_path = os.path.join(
        os.path.expanduser("~"), "Documents", "Relays_domain_model.json"
    )

    if not os.path.exists(model_path):
        return 0

    model_id = load_model_from_file(conn, model_path, app_id=app_id)
    return 1 if model_id else 0


if __name__ == "__main__":
    from domain_graph import get_domain_conn
    conn = get_domain_conn()
    n = seed_domains(conn)
    conn.close()
    print(f"Seeded {n} domain models for base.")
