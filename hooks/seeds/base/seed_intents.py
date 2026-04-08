"""
seed_intents.py — Seed Base application intent nodes.

Delegates to existing seed_intent_graph.py, seed_mdm_intents.py, and
seed_pipeline_gaps.py, passing app_id='base'.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def seed_intents(conn, app_id: str = "base") -> dict:
    """Seed intent nodes, triplets, and aliases for the given app_id.

    Runs all three existing seed scripts in order. Returns summary dict.
    """
    from intent_graph import init_intent_db

    init_intent_db(conn)

    results = {"nodes": 0, "triplets": 0, "aliases": 0}

    # 1. Base intent graph
    try:
        from seed_intent_graph import seed_base_intents
        r = seed_base_intents(conn, app_id=app_id)
        if isinstance(r, dict):
            results["nodes"] += r.get("nodes", 0)
            results["triplets"] += r.get("triplets", 0)
            results["aliases"] += r.get("aliases", 0)
    except (ImportError, TypeError):
        # Original script may not accept app_id — run it as-is
        try:
            from seed_intent_graph import main as seed_intent_main
            seed_intent_main()
        except Exception:
            pass

    # 2. MDM intents
    try:
        from seed_mdm_intents import seed_mdm_intents as _seed_mdm
        _seed_mdm(conn, app_id=app_id)
    except (ImportError, TypeError):
        try:
            from seed_mdm_intents import main as seed_mdm_main
            seed_mdm_main()
        except Exception:
            pass

    # 3. Pipeline gap intents
    try:
        from seed_pipeline_gaps import seed_intent_gaps
        seed_intent_gaps(conn, app_id=app_id)
    except (ImportError, TypeError):
        try:
            from seed_pipeline_gaps import main as seed_gaps_main
            seed_gaps_main()
        except Exception:
            pass

    conn.commit()
    return results


if __name__ == "__main__":
    from intent_graph import get_intent_conn
    conn = get_intent_conn()
    r = seed_intents(conn)
    conn.close()
    print(f"Seeded intents for base: {r}")
