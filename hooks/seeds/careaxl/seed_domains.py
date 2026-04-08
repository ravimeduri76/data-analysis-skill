"""
seed_domains.py — Seed CareAXL-specific domain models.

CareAXL currently inherits domain models from Base. Placeholder for future
CareAXL-specific domain models (e.g., SDOH taxonomy, clinical ontology).
"""

from __future__ import annotations


def seed_domains(conn, app_id: str = "careaxl") -> int:
    """Seed domain models for CareAXL. Returns count of models seeded.

    Currently returns 0 — CareAXL inherits from Base via the OR clause.
    """
    return 0
