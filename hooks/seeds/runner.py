"""
runner.py — Seed orchestrator for DESO multi-application architecture.

Public API:
    seed_app(app_id, conn=None)  — seed a single application
    seed_all(conn=None)          — seed all registered applications (base first)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Ensure hooks directory is importable
_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def _get_conn():
    """Get a DB connection from domain_graph."""
    from domain_graph import get_domain_conn
    return get_domain_conn()


def seed_app(app_id: str, conn=None) -> dict:
    """Seed all data for a single application.

    Always seeds 'base' first if app_id is not 'base' (dependency resolution).

    Returns summary dict: {app_id, agents, intents, plans, domains}
    """
    close_conn = False
    if conn is None:
        conn = _get_conn()
        close_conn = True

    # Run migration first
    from db_migrate import run_migration
    run_migration(conn)

    summary = {"app_id": app_id, "agents": 0, "intents": {}, "plans": 0, "domains": 0}

    try:
        if app_id == "base":
            summary.update(_seed_base(conn))
        elif app_id == "dataaxl":
            summary.update(_seed_dataaxl(conn))
        elif app_id == "careaxl":
            summary.update(_seed_careaxl(conn))
        else:
            summary["error"] = f"Unknown app_id: {app_id}"
    finally:
        if close_conn:
            conn.close()

    return summary


def seed_all(conn=None) -> list[dict]:
    """Seed all registered applications in dependency order.

    Always seeds 'base' first, then other apps.
    Returns list of summary dicts.
    """
    close_conn = False
    if conn is None:
        conn = _get_conn()
        close_conn = True

    from db_migrate import run_migration
    run_migration(conn)

    results = []
    try:
        # Base always first
        results.append(_seed_base(conn))
        results.append(_seed_dataaxl(conn))
        results.append(_seed_careaxl(conn))
    finally:
        if close_conn:
            conn.close()

    return results


def _seed_base(conn) -> dict:
    """Seed base application."""
    summary = {"app_id": "base", "agents": 0, "intents": {}, "plans": 0, "domains": 0}

    try:
        from seeds.base.seed_agents import seed_agents
        summary["agents"] = seed_agents(conn, app_id="base")
    except Exception as e:
        summary["agents_error"] = str(e)

    try:
        from seeds.base.seed_intents import seed_intents
        summary["intents"] = seed_intents(conn, app_id="base")
    except Exception as e:
        summary["intents_error"] = str(e)

    try:
        from seeds.base.seed_plans import seed_plans
        summary["plans"] = seed_plans(conn, app_id="base")
    except Exception as e:
        summary["plans_error"] = str(e)

    try:
        from seeds.base.seed_domains import seed_domains
        summary["domains"] = seed_domains(conn, app_id="base")
    except Exception as e:
        summary["domains_error"] = str(e)

    return summary


def _seed_dataaxl(conn) -> dict:
    """Seed DataAXL application."""
    summary = {"app_id": "dataaxl", "intents": 0, "plans": 0, "domains": 0}

    try:
        from seeds.dataaxl.seed_intents import seed_intents
        summary["intents"] = seed_intents(conn, app_id="dataaxl")
    except Exception as e:
        summary["intents_error"] = str(e)

    try:
        from seeds.dataaxl.seed_plans import seed_plans
        summary["plans"] = seed_plans(conn, app_id="dataaxl")
    except Exception as e:
        summary["plans_error"] = str(e)

    try:
        from seeds.dataaxl.seed_domains import seed_domains
        summary["domains"] = seed_domains(conn, app_id="dataaxl")
    except Exception as e:
        summary["domains_error"] = str(e)

    return summary


def _seed_careaxl(conn) -> dict:
    """Seed CareAXL application."""
    summary = {"app_id": "careaxl", "intents": 0, "plans": 0, "domains": 0}

    try:
        from seeds.careaxl.seed_intents import seed_intents
        summary["intents"] = seed_intents(conn, app_id="careaxl")
    except Exception as e:
        summary["intents_error"] = str(e)

    try:
        from seeds.careaxl.seed_plans import seed_plans
        summary["plans"] = seed_plans(conn, app_id="careaxl")
    except Exception as e:
        summary["plans_error"] = str(e)

    try:
        from seeds.careaxl.seed_domains import seed_domains
        summary["domains"] = seed_domains(conn, app_id="careaxl")
    except Exception as e:
        summary["domains_error"] = str(e)

    return summary


if __name__ == "__main__":
    import json
    results = seed_all()
    print(json.dumps(results, indent=2, default=str))
