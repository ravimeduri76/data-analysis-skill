"""
test_deso_pipeline.py — End-to-end test of the DESO cognitive pipeline:
  Domain Detection → Intent Discovery → Execution Plan Resolution

Tests 10 real-world scenarios across Industrial, Healthcare, MDM, and Consumer domains.

Usage:
    python hooks/test_deso_pipeline.py
    python hooks/test_deso_pipeline.py --scenario 5       # run a single scenario
    python hooks/test_deso_pipeline.py --no-llm           # skip LLM tier, test fallback only
"""

from __future__ import annotations

import json
import os
import sys
import time
import textwrap
from typing import Optional

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": 1,
        "query": "I want to improve my product catalog on TRUCK PARTS by referring OEM / Manufacturer's data sheets",
        "expected_domain": "Commercial / Industrial",
        "expected_lifecycle": "enrichment",
        "has_dataset": True,
        "notes": "Product enrichment from external OEM/manufacturer data sheets",
    },
    {
        "id": 2,
        "query": "I want to extract fields from fastener descriptions",
        "expected_domain": "Commercial / Industrial",
        "expected_lifecycle": "preparation",
        "has_dataset": True,
        "notes": "Schema extraction / attribute parsing from unstructured product descriptions",
    },
    {
        "id": 3,
        "query": "I want to improve the Schema and ensure important technical attributes that customers look for are available",
        "expected_domain": "Commercial / Industrial",
        "expected_lifecycle": "preparation",
        "has_dataset": True,
        "notes": "Schema enhancement — recommending + enforcing technical attributes",
    },
    {
        "id": 4,
        "query": "Want to assess the quality of my customer data and ensure no duplicates",
        "expected_domain": "Master Data Management",
        "expected_lifecycle": "analytics",
        "has_dataset": True,
        "notes": "DQ assessment + dedup — quality + MDM crossover",
    },
    {
        "id": 5,
        "query": "I want to unify customer data across 7 ERPs and create Golden Records and set up a process to ensure the quality of customer master",
        "expected_domain": "Master Data Management",
        "expected_lifecycle": "preparation",
        "has_dataset": True,
        "notes": "Full MDM lifecycle: entity resolution, golden record, ongoing DQ governance",
    },
    {
        "id": 6,
        "query": "I want to understand Payer behavior and see if there are deliberate defer, delay, deny tendencies",
        "expected_domain": "Healthcare",
        "expected_lifecycle": "analytics",
        "has_dataset": True,
        "notes": "Payer analytics — behavioral pattern detection, potential anomaly/fraud detection",
    },
    {
        "id": 7,
        "query": "I want to plug revenue leakage on Home Care particularly on Visit Syncs and F2F",
        "expected_domain": "Healthcare",
        "expected_lifecycle": "analytics",
        "has_dataset": True,
        "notes": "Revenue cycle analytics — leakage detection, operational data quality",
    },
    {
        "id": 8,
        "query": "How can I identify PCS patients most likely to be readmitted in next 30 days",
        "expected_domain": "Healthcare",
        "expected_lifecycle": "analytics",
        "has_dataset": True,
        "notes": "Predictive analytics — readmission risk scoring, survival/ML modeling",
    },
    {
        "id": 9,
        "query": "How can I prevent or mitigate Mortality in Hospice care",
        "expected_domain": "Healthcare",
        "expected_lifecycle": "analytics",
        "has_dataset": True,
        "notes": "Clinical outcome analytics — mortality risk factors, survival analysis",
    },
    {
        "id": 10,
        "query": "I want to drive customer engagement and give daily suggestions on fitness and diet",
        "expected_domain": "Wellness",
        "expected_lifecycle": "recommendation",
        "has_dataset": True,
        "notes": "Consumer wellness — recommendation engine, content generation",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WIDTH = 100
_SEP = "=" * _WIDTH


def _ascii(text: str) -> str:
    """Ensure ASCII-safe output."""
    return text.encode("ascii", errors="replace").decode("ascii")


def _indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(text, prefix)


def _load_env() -> None:
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Data_Analysis", ".env",
    )
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _build_dcp(scenario: dict, domain_ctx: dict) -> dict:
    """Build a minimal Domain Context Pack for resolve_execution_plan."""
    return {
        "domain": domain_ctx.get("domain", scenario.get("expected_domain", "General")),
        "confidence": domain_ctx.get("confidence", "medium"),
        "query_has_dataset": scenario.get("has_dataset", True),
        "graph_available": domain_ctx.get("found", False),
        "user_query": scenario["query"],
        "lifecycle_stage": scenario.get("expected_lifecycle", "analytics"),
        "summary": domain_ctx.get("overview", {}).get("domain_summary", ""),
    }


def _build_is_context(scenario: dict, intent_ctx: dict) -> dict:
    """Build a minimal Intent Specification for render_execution_plan_as_ap."""
    matched = intent_ctx.get("matched_node") or {}
    parents = intent_ctx.get("abstract_parents", [])
    children = intent_ctx.get("specific_children", [])

    return {
        "explicit_goal": scenario["query"],
        "latent_goal": parents[0]["label"] if parents else "",
        "matched_intent": matched.get("name", "unknown"),
        "intent_confidence": intent_ctx.get("confidence", 0.0),
        "lifecycle_stage": matched.get("lifecycle_stage", scenario.get("expected_lifecycle", "")),
        "requires_hitl": False,
        "risk_signals": [],
        "missing_inputs": [],
        "specific_children": [c.get("name", "") for c in children],
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: dict,
    domain_conn,
    intent_conn,
    exec_conn,
    api_key: Optional[str] = None,
    use_llm: bool = True,
) -> dict:
    """Run one scenario through the full DESO cognitive pipeline."""

    from hooks.domain_graph import get_domain_context
    from hooks.intent_graph import resolve_intent
    from hooks.execution_graph import (
        resolve_execution_plan,
        render_execution_plan_as_ap,
    )

    sid = scenario["id"]
    query = scenario["query"]
    result = {"scenario_id": sid, "query": query}

    # ── Step 1: Domain Detection ──────────────────────────────────────────
    # Use domain graph context (semantic search over seeded domain models)
    t0 = time.time()
    domain_ctx = get_domain_context(domain_conn, scenario.get("expected_domain", ""), query)
    result["domain"] = {
        "detected": domain_ctx.get("domain", "unknown"),
        "found_in_graph": domain_ctx.get("found", False),
        "overview": domain_ctx.get("overview", {}),
        "top_nodes": [
            {"type": n["node_type"], "key": n["node_key"], "score": round(n["score"], 3)}
            for n in domain_ctx.get("nodes", [])[:5]
        ],
        "invariants": domain_ctx.get("invariants", []),
        "elapsed_ms": round((time.time() - t0) * 1000),
    }

    # ── Step 2: Intent Discovery ──────────────────────────────────────────
    t0 = time.time()
    intent_ctx = resolve_intent(intent_conn, query, domain=scenario.get("expected_domain"))
    matched = intent_ctx.get("matched_node") or {}
    result["intent"] = {
        "matched_node": matched.get("name", "NONE"),
        "matched_label": matched.get("label", ""),
        "confidence": round(intent_ctx.get("confidence", 0.0), 3),
        "level": matched.get("level", ""),
        "lifecycle_stage": matched.get("lifecycle_stage", ""),
        "abstract_parents": [
            p.get("name", "") for p in intent_ctx.get("abstract_parents", [])
        ],
        "specific_children": [
            c.get("name", "") for c in intent_ctx.get("specific_children", [])
        ],
        "requires_clarification": intent_ctx.get("requires_clarification", False),
        "ambiguity_candidates": [
            {"name": c.get("name", ""), "score": round(c.get("score", 0), 3)}
            for c in intent_ctx.get("ambiguity_candidates", [])
        ],
        "elapsed_ms": round((time.time() - t0) * 1000),
    }

    # ── Step 3: Execution Plan Resolution ─────────────────────────────────
    dcp = _build_dcp(scenario, domain_ctx)
    is_ctx = _build_is_context(scenario, intent_ctx)

    t0 = time.time()
    plan_result = resolve_execution_plan(
        exec_conn,
        intent_context=intent_ctx,
        dcp=dcp,
        api_key=api_key if use_llm else None,
    )
    plan_elapsed = round((time.time() - t0) * 1000)

    plan_obj = plan_result.get("plan")
    result["execution_plan"] = {
        "plan_name": plan_obj.get("name", "NONE") if plan_obj else "NONE",
        "plan_label": plan_obj.get("label", "") if plan_obj else "",
        "confidence": round(plan_result.get("confidence", 0.0), 3),
        "method": plan_result.get("method", "fallback"),
        "reasoning": plan_result.get("reasoning", ""),
        "warnings": plan_result.get("warnings", []),
        "overrides": plan_result.get("overrides"),
        "agents": [],
        "hitl_gates": [],
        "elapsed_ms": plan_elapsed,
    }

    if plan_obj:
        steps = plan_obj.get("steps", [])
        for s in steps:
            result["execution_plan"]["agents"].append({
                "step_order": s.get("step_order"),
                "agent_id": s.get("agent_card_id", s.get("agent_id", "")),
                "agent_name": s.get("agent_name", ""),
                "purpose": s.get("purpose", ""),
                "hitl_gate": s.get("hitl_gate"),
                "is_optional": s.get("is_optional", False),
            })
            if s.get("hitl_gate"):
                result["execution_plan"]["hitl_gates"].append({
                    "agent": s.get("agent_card_id", s.get("agent_id", "")),
                    "gate": s.get("hitl_gate"),
                    "reason": s.get("hitl_reason", ""),
                })

    # ── Step 4: Render as Agent Plan (AP) ─────────────────────────────────
    if plan_obj:
        ap_text = render_execution_plan_as_ap(plan_result, is_ctx)
        result["agent_plan_text"] = ap_text
    else:
        result["agent_plan_text"] = "(No plan resolved — SA3 would use hardcoded rules)"

    return result


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(results: list[dict]) -> str:
    """Render all scenario results as a readable text report."""
    lines = []
    lines.append(_SEP)
    lines.append("  DESO COGNITIVE PIPELINE — END-TO-END TEST REPORT")
    lines.append(f"  {len(results)} scenarios tested")
    lines.append(_SEP)
    lines.append("")

    for r in results:
        sid = r["scenario_id"]
        lines.append(f"{'─' * _WIDTH}")
        lines.append(f"  SCENARIO {sid}")
        lines.append(f"{'─' * _WIDTH}")
        lines.append(f"  Query: {_ascii(r['query'])}")
        lines.append("")

        # Domain
        d = r.get("domain", {})
        lines.append(f"  [1] DOMAIN DETECTION  ({d.get('elapsed_ms', '?')}ms)")
        lines.append(f"      Domain       : {d.get('detected', '?')}")
        lines.append(f"      In graph     : {d.get('found_in_graph', False)}")
        if d.get("top_nodes"):
            lines.append(f"      Top entities :")
            for n in d["top_nodes"][:3]:
                lines.append(f"        - [{n['type']}] {n['key']} (sim={n['score']})")
        if d.get("invariants"):
            lines.append(f"      Invariants   : {len(d['invariants'])} domain rules loaded")
        lines.append("")

        # Intent
        i = r.get("intent", {})
        lines.append(f"  [2] INTENT DISCOVERY  ({i.get('elapsed_ms', '?')}ms)")
        lines.append(f"      Matched      : {i.get('matched_node', 'NONE')}")
        lines.append(f"      Label        : {i.get('matched_label', '')}")
        lines.append(f"      Confidence   : {i.get('confidence', 0)}")
        lines.append(f"      Level        : {i.get('level', '?')}")
        lines.append(f"      Lifecycle    : {i.get('lifecycle_stage', '?')}")
        if i.get("abstract_parents"):
            lines.append(f"      Parents      : {', '.join(i['abstract_parents'])}")
        if i.get("specific_children"):
            lines.append(f"      Children     : {', '.join(i['specific_children'])}")
        if i.get("requires_clarification"):
            lines.append(f"      ** AMBIGUOUS — requires clarification **")
            for c in i.get("ambiguity_candidates", []):
                lines.append(f"        - {c['name']} (sim={c['score']})")
        lines.append("")

        # Execution Plan
        e = r.get("execution_plan", {})
        lines.append(f"  [3] EXECUTION PLAN  ({e.get('elapsed_ms', '?')}ms)")
        lines.append(f"      Plan         : {e.get('plan_name', 'NONE')}")
        lines.append(f"      Label        : {e.get('plan_label', '')}")
        lines.append(f"      Confidence   : {e.get('confidence', 0)}")
        lines.append(f"      Method       : {e.get('method', 'fallback')}")
        if e.get("reasoning"):
            reasoning = e["reasoning"]
            if len(reasoning) > 300:
                reasoning = reasoning[:300] + "..."
            lines.append(f"      Reasoning    : {_ascii(reasoning)}")
        if e.get("warnings"):
            lines.append(f"      Warnings     : {'; '.join(e['warnings'])}")
        if e.get("overrides"):
            ov = e["overrides"]
            parts = []
            if ov.get("add_steps"):
                parts.append(f"+{len(ov['add_steps'])} steps")
            if ov.get("remove_agents"):
                parts.append(f"-{len(ov['remove_agents'])} agents")
            if ov.get("modify_steps"):
                parts.append(f"~{len(ov['modify_steps'])} modified")
            lines.append(f"      Overrides    : {', '.join(parts)}")
        lines.append("")

        # Agent sequence
        agents = e.get("agents", [])
        if agents:
            lines.append(f"      Agent Sequence ({len(agents)} agents):")
            prev_order = None
            for a in agents:
                order = a["step_order"]
                parallel = " (parallel)" if order == prev_order else ""
                opt = " [optional]" if a.get("is_optional") else ""
                hitl = f" [HITL:{a['hitl_gate']}]" if a.get("hitl_gate") else ""
                lines.append(
                    f"        Step {order}: {a['agent_id']} {a['agent_name']}"
                    f"{parallel}{opt}{hitl}"
                )
                if a.get("purpose"):
                    purpose = a["purpose"]
                    if len(purpose) > 80:
                        purpose = purpose[:80] + "..."
                    lines.append(f"               Purpose: {_ascii(purpose)}")
                prev_order = order
            lines.append("")

        # HITL Gates
        hitl = e.get("hitl_gates", [])
        if hitl:
            lines.append(f"      HITL Gates ({len(hitl)}):")
            for h in hitl:
                lines.append(f"        - {h['gate'].upper()} {h['agent']}: {_ascii(h.get('reason', ''))}")
            lines.append("")

        # Full AP text
        lines.append(f"  [4] RENDERED AGENT PLAN (AP)")
        lines.append("")
        ap = r.get("agent_plan_text", "")
        for ap_line in ap.split("\n"):
            lines.append(f"      {_ascii(ap_line)}")
        lines.append("")
        lines.append("")

    # Summary table
    lines.append(_SEP)
    lines.append("  SUMMARY")
    lines.append(_SEP)
    lines.append("")
    lines.append(f"  {'#':<4} {'Method':<12} {'Conf':<7} {'Plan':<35} {'Intent':<30} {'IntConf':<8}")
    lines.append(f"  {'─'*4} {'─'*12} {'─'*7} {'─'*35} {'─'*30} {'─'*8}")
    for r in results:
        e = r.get("execution_plan", {})
        i = r.get("intent", {})
        lines.append(
            f"  {r['scenario_id']:<4} "
            f"{e.get('method', '?'):<12} "
            f"{e.get('confidence', 0):<7.3f} "
            f"{e.get('plan_name', 'NONE'):<35} "
            f"{i.get('matched_node', 'NONE'):<30} "
            f"{i.get('confidence', 0):<8.3f}"
        )
    lines.append("")
    lines.append(_SEP)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WI-1C: Evaluation Harness (OB1 pattern: n-agentic-harnesses)
# ---------------------------------------------------------------------------

# Acceptance criteria per scenario. Each criterion is a (name, check_fn) pair
# where check_fn takes the scenario result dict and returns (passed: bool, detail: str).

def _check_domain(expected: str):
    def _fn(r: dict) -> tuple[bool, str]:
        detected = r.get("domain", {}).get("detected", "unknown")
        ok = expected.lower() in detected.lower() or detected.lower() in expected.lower()
        return ok, f"expected={expected}, got={detected}"
    return _fn


def _check_intent_confidence(min_conf: float):
    def _fn(r: dict) -> tuple[bool, str]:
        conf = r.get("intent", {}).get("confidence", 0)
        return conf >= min_conf, f"confidence={conf:.3f} (min={min_conf})"
    return _fn


def _check_plan_resolved(r: dict) -> tuple[bool, str]:
    name = r.get("execution_plan", {}).get("plan_name", "NONE")
    return name != "NONE", f"plan={name}"


def _check_no_hitl(r: dict) -> tuple[bool, str]:
    clarif = r.get("intent", {}).get("requires_clarification", False)
    hitl_gates = r.get("execution_plan", {}).get("hitl_gates", [])
    return not clarif and len(hitl_gates) == 0, f"clarification={clarif}, hitl_gates={len(hitl_gates)}"


def _check_has_hitl(r: dict) -> tuple[bool, str]:
    clarif = r.get("intent", {}).get("requires_clarification", False)
    hitl_gates = r.get("execution_plan", {}).get("hitl_gates", [])
    return clarif or len(hitl_gates) > 0, f"clarification={clarif}, hitl_gates={len(hitl_gates)}"


def _check_agent_present(agent_id: str):
    def _fn(r: dict) -> tuple[bool, str]:
        agents = r.get("execution_plan", {}).get("agents", [])
        ids = [a.get("agent_id", "") for a in agents]
        return agent_id in ids, f"agents={ids}"
    return _fn


def _check_plan_method(methods: list[str]):
    def _fn(r: dict) -> tuple[bool, str]:
        method = r.get("execution_plan", {}).get("method", "fallback")
        return method in methods, f"method={method}, expected one of {methods}"
    return _fn


EVALUATION_HARNESS: dict[int, list[tuple[str, callable]]] = {
    1: [
        ("domain is Commercial/Industrial", _check_domain("Commercial / Industrial")),
        ("intent confidence > 0.4", _check_intent_confidence(0.4)),
        ("execution plan resolved", _check_plan_resolved),
        ("plan method is llm or semantic", _check_plan_method(["llm", "semantic", "pattern_match"])),
    ],
    2: [
        ("domain is Commercial/Industrial", _check_domain("Commercial / Industrial")),
        ("intent confidence > 0.4", _check_intent_confidence(0.4)),
        ("execution plan resolved", _check_plan_resolved),
    ],
    3: [
        ("domain is Commercial/Industrial", _check_domain("Commercial / Industrial")),
        ("execution plan resolved", _check_plan_resolved),
    ],
    4: [
        ("domain is MDM", _check_domain("Master Data Management")),
        ("intent confidence > 0.4", _check_intent_confidence(0.4)),
        ("execution plan resolved", _check_plan_resolved),
    ],
    5: [
        ("domain is MDM", _check_domain("Master Data Management")),
        ("intent confidence > 0.4", _check_intent_confidence(0.4)),
        ("execution plan resolved", _check_plan_resolved),
    ],
    6: [
        ("domain is Healthcare", _check_domain("Healthcare")),
        ("intent confidence > 0.4", _check_intent_confidence(0.4)),
        ("execution plan resolved", _check_plan_resolved),
    ],
    7: [
        ("domain is Healthcare", _check_domain("Healthcare")),
        ("execution plan resolved", _check_plan_resolved),
    ],
    8: [
        ("domain is Healthcare", _check_domain("Healthcare")),
        ("intent confidence > 0.4", _check_intent_confidence(0.4)),
        ("execution plan resolved", _check_plan_resolved),
    ],
    9: [
        ("domain is Healthcare", _check_domain("Healthcare")),
        ("execution plan resolved", _check_plan_resolved),
    ],
    10: [
        ("domain is Wellness", _check_domain("Wellness")),
        ("execution plan resolved", _check_plan_resolved),
    ],
}


def run_harness(results: list[dict], scenario_id: int | None = None) -> dict:
    """Run the evaluation harness against pipeline results.

    Returns {scenario_id: {"passed": int, "failed": int, "total": int, "details": [...]}}
    """
    harness_results: dict[int, dict] = {}

    for r in results:
        sid = r.get("scenario_id")
        if scenario_id is not None and sid != scenario_id:
            continue
        if "error" in r:
            harness_results[sid] = {"passed": 0, "failed": 1, "total": 1,
                                     "details": [{"criterion": "no error", "passed": False, "detail": r["error"]}]}
            continue

        criteria = EVALUATION_HARNESS.get(sid, [])
        details = []
        passed = 0
        failed = 0
        for name, check_fn in criteria:
            try:
                ok, detail = check_fn(r)
            except Exception as exc:
                ok, detail = False, f"check error: {exc}"
            details.append({"criterion": name, "passed": ok, "detail": detail})
            if ok:
                passed += 1
            else:
                failed += 1
        harness_results[sid] = {"passed": passed, "failed": failed, "total": passed + failed, "details": details}

    return harness_results


def render_harness_report(harness_results: dict) -> str:
    """Render harness results as a readable report."""
    lines = ["", "=" * 80, "  EVALUATION HARNESS RESULTS", "=" * 80, ""]

    total_passed = 0
    total_failed = 0

    for sid in sorted(harness_results.keys()):
        hr = harness_results[sid]
        total_passed += hr["passed"]
        total_failed += hr["failed"]
        status = "PASS" if hr["failed"] == 0 else "FAIL"
        lines.append(f"  Scenario {sid}: {status} ({hr['passed']}/{hr['total']} criteria)")
        for d in hr["details"]:
            mark = "  [+]" if d["passed"] else "  [-]"
            lines.append(f"    {mark} {d['criterion']}: {d['detail']}")
        lines.append("")

    lines.append("-" * 80)
    lines.append(f"  TOTAL: {total_passed} passed, {total_failed} failed, "
                 f"{total_passed + total_failed} criteria across {len(harness_results)} scenarios")
    pct = (total_passed / max(1, total_passed + total_failed)) * 100
    lines.append(f"  PASS RATE: {pct:.0f}%")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test DESO cognitive pipeline")
    parser.add_argument("--scenario", type=int, default=0, help="Run single scenario (1-10)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM tier, test fallback only")
    parser.add_argument("--harness", action="store_true", help="Run evaluation harness with acceptance criteria")
    args = parser.parse_args()

    _load_env()
    # Force-read from .env file directly if not in environ
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for env_candidate in [
            os.path.join(os.path.expanduser("~"), "Data_Analysis", ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        ]:
            if os.path.exists(env_candidate):
                with open(env_candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("ANTHROPIC_API_KEY="):
                            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
                            break
                if os.environ.get("ANTHROPIC_API_KEY"):
                    break
    api_key = os.environ.get("ANTHROPIC_API_KEY") if not args.no_llm else None

    # Import hooks
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hooks.domain_graph import get_domain_conn
    from hooks.intent_graph import get_intent_conn
    from hooks.execution_graph import get_execution_conn

    # Get connections
    domain_conn = get_domain_conn()
    intent_conn = get_intent_conn()
    exec_conn = get_execution_conn()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["id"] == args.scenario]
        if not scenarios:
            print(f"Scenario {args.scenario} not found (valid: 1-10)")
            sys.exit(1)

    print(f"\nRunning {len(scenarios)} scenario(s)...")
    print(f"LLM tier: {'enabled' if api_key else 'disabled (fallback only)'}")
    print()

    results = []
    for sc in scenarios:
        print(f"  [{sc['id']}/10] {sc['query'][:70]}...")
        try:
            r = run_scenario(
                sc, domain_conn, intent_conn, exec_conn,
                api_key=api_key, use_llm=bool(api_key),
            )
            results.append(r)
            e = r.get("execution_plan", {})
            print(f"         -> {e.get('method', '?')} | conf={e.get('confidence', 0):.2f} | plan={e.get('plan_name', 'NONE')}")
        except Exception as ex:
            print(f"         -> ERROR: {ex}")
            results.append({"scenario_id": sc["id"], "query": sc["query"], "error": str(ex)})

    # Render and save report
    report = render_report(results)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "deso_pipeline_test_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also save raw JSON
    json_path = os.path.join(out_dir, "deso_pipeline_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")
    print(f"JSON results:    {json_path}")

    # WI-1C: Run evaluation harness if requested
    if args.harness:
        harness_results = run_harness(results, scenario_id=args.scenario or None)
        harness_text = render_harness_report(harness_results)
        print(harness_text)

        harness_path = os.path.join(out_dir, "deso_harness_results.json")
        with open(harness_path, "w", encoding="utf-8") as f:
            json.dump(harness_results, f, indent=2, default=str)
        print(f"Harness results: {harness_path}")
    else:
        print()
        # Windows console may not support all chars — force ASCII
        sys.stdout.buffer.write(_ascii(report).encode("ascii", errors="replace"))
        sys.stdout.buffer.write(b"\n")

    # Close connections
    domain_conn.close()
    intent_conn.close()
    exec_conn.close()


if __name__ == "__main__":
    main()
