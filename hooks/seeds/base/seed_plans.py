"""
seed_plans.py — Seed Base application execution plans.

Delegates to existing seed_execution_graph.py for plan data,
passing app_id='base'.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


def seed_plans(conn, app_id: str = "base") -> int:
    """Seed execution plans, steps, dependencies, and examples for the given app_id.

    Returns count of plans seeded.
    """
    from execution_graph import (
        init_execution_db, upsert_execution_plan, add_execution_step,
        add_execution_dependency, add_execution_example,
    )

    init_execution_db(conn)

    # Import plan definitions from the original seed script
    try:
        from seed_execution_graph import PLANS, STEPS, DEPENDENCIES, EXAMPLES
    except ImportError:
        return 0

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
            conditions=plan.get("conditions"),
            app_id=app_id,
            solution_id=plan.get("solution_id"),
        )
        count += 1

    # Steps
    for step in STEPS:
        try:
            add_execution_step(
                conn,
                plan_name=step["plan_name"],
                step_order=step["step_order"],
                agent_id=step["agent_id"],
                purpose=step.get("purpose"),
                ingress_map=step.get("ingress_map"),
                egress_keys=step.get("egress_keys"),
                is_optional=step.get("is_optional", False),
                condition=step.get("condition"),
                hitl_gate=step.get("hitl_gate"),
                hitl_reason=step.get("hitl_reason"),
            )
        except Exception:
            pass  # May already exist (UNIQUE constraint)

    # Dependencies
    for dep in DEPENDENCIES:
        try:
            add_execution_dependency(
                conn,
                plan_name=dep["plan_name"],
                from_agent_id=dep["from_agent_id"],
                to_agent_id=dep["to_agent_id"],
                dep_type=dep.get("dep_type", "sequence"),
                description=dep.get("description"),
            )
        except Exception:
            pass

    # Examples
    for ex in EXAMPLES:
        try:
            add_execution_example(
                conn,
                example_name=ex["example_name"],
                intent_context=ex["intent_context"],
                domain_context=ex["domain_context"],
                selected_plan=ex["selected_plan"],
                reasoning=ex["reasoning"],
                plan_overrides=ex.get("plan_overrides"),
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
    print(f"Seeded {n} execution plans for base.")
