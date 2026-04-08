"""
mcp_server_deso.py — DESO MCP Server with app-scoped tools.

Exposes DESO knowledge graph tools via the Model Context Protocol,
with app_id scoping for multi-application support.

Transport: stdio (started on demand by MCP clients)

Tools:
    resolve_domain(query, app_id?)          — Scoped domain graph lookup
    resolve_intent(query, app_id?, domain?) — Scoped intent graph lookup
    resolve_plan(intent_ctx, dcp, app_id?, solution_id?) — Scoped execution plan
    list_agents(app_id?)                    — Base + app agents
    list_apps()                             — All registered applications
    list_solutions(app_id)                  — Solutions for an app

Resources:
    deso://apps                              — All registered applications
    deso://apps/{app_id}                     — Application manifest
    deso://apps/{app_id}/solutions           — Solutions for an app
    deso://apps/{app_id}/agents              — Visible agents for an app
"""

from __future__ import annotations

import json
import os
import sys

# Ensure hooks directory is importable
_HOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hooks")
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)

from mcp.server import Server
from mcp.server.stdio import run_stdio
from mcp.types import (
    TextContent,
    Tool,
    Resource,
)

server = Server("deso-server")


# ---------------------------------------------------------------------------
# Lazy DB connection
# ---------------------------------------------------------------------------

_conn_cache: dict = {}


def _get_conn():
    """Get or create a shared DB connection."""
    if "conn" not in _conn_cache or _conn_cache["conn"].closed:
        from domain_graph import get_domain_conn
        _conn_cache["conn"] = get_domain_conn()
        # Run migration on first connect
        from db_migrate import run_migration
        run_migration(_conn_cache["conn"])
    return _conn_cache["conn"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="resolve_domain",
            description="Resolve domain context from the domain knowledge graph, scoped by application",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query or domain keywords"},
                    "app_id": {"type": "string", "description": "Application ID (default: base)", "default": "base"},
                    "domain": {"type": "string", "description": "Known domain name (optional)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="resolve_intent",
            description="Resolve user intent from the intent knowledge graph, scoped by application",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query or explicit goal"},
                    "app_id": {"type": "string", "description": "Application ID (default: base)", "default": "base"},
                    "domain": {"type": "string", "description": "Domain filter (optional)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="resolve_plan",
            description="Resolve an execution plan from the execution graph, scoped by application and solution",
            inputSchema={
                "type": "object",
                "properties": {
                    "intent_context": {"type": "object", "description": "Intent context dict (from resolve_intent)"},
                    "dcp": {"type": "object", "description": "Domain context pack (from resolve_domain)"},
                    "app_id": {"type": "string", "description": "Application ID (default: base)", "default": "base"},
                    "solution_id": {"type": "string", "description": "Solution ID (optional)"},
                },
                "required": ["intent_context", "dcp"],
            },
        ),
        Tool(
            name="list_agents",
            description="List agent cards visible to an application (base + app-specific)",
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "description": "Application ID (default: base)", "default": "base"},
                    "type_filter": {"type": "string", "description": "Filter by agent type (optional)"},
                    "domain_filter": {"type": "string", "description": "Filter by domain (optional)"},
                },
            },
        ),
        Tool(
            name="list_apps",
            description="List all registered DESO applications",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_solutions",
            description="List solutions for a given application",
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "description": "Application ID"},
                },
                "required": ["app_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "resolve_domain":
            return await _resolve_domain(arguments)
        elif name == "resolve_intent":
            return await _resolve_intent(arguments)
        elif name == "resolve_plan":
            return await _resolve_plan(arguments)
        elif name == "list_agents":
            return await _list_agents(arguments)
        elif name == "list_apps":
            return await _list_apps(arguments)
        elif name == "list_solutions":
            return await _list_solutions(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def _resolve_domain(args: dict) -> list[TextContent]:
    query = args["query"]
    app_id = args.get("app_id", "base")
    domain = args.get("domain")

    conn = _get_conn()

    if domain:
        from domain_graph import get_domain_context, render_domain_context
        ctx = get_domain_context(conn, domain, query, app_id=app_id)
        text = render_domain_context(ctx)
    else:
        # Use domain_detector for detection, then graph for context
        from domain_detector import detect_domain
        domain_ctx = detect_domain(None, user_query=query)
        detected_domain = domain_ctx.get("domain", "General")

        from domain_graph import get_domain_context, render_domain_context
        ctx = get_domain_context(conn, detected_domain, query, app_id=app_id)
        header = f"Detected domain: {detected_domain} (confidence: {domain_ctx.get('confidence', 'unknown')})\n\n"
        text = header + render_domain_context(ctx)

    return [TextContent(type="text", text=text)]


async def _resolve_intent(args: dict) -> list[TextContent]:
    query = args["query"]
    app_id = args.get("app_id", "base")
    domain = args.get("domain")

    conn = _get_conn()
    from intent_graph import resolve_intent as _resolve, render_intent_context
    ctx = _resolve(conn, query, domain=domain, app_id=app_id)
    text = render_intent_context(ctx)
    return [TextContent(type="text", text=text)]


async def _resolve_plan(args: dict) -> list[TextContent]:
    intent_context = args["intent_context"]
    dcp = args["dcp"]
    app_id = args.get("app_id", "base")
    solution_id = args.get("solution_id")

    conn = _get_conn()
    from execution_graph import resolve_execution_plan, render_execution_plan_as_ap
    result = resolve_execution_plan(
        conn, intent_context, dcp,
        app_id=app_id, solution_id=solution_id,
    )
    if result.get("plan"):
        text = render_execution_plan_as_ap(result, intent_context)
    else:
        text = json.dumps(result, indent=2, default=str)
    return [TextContent(type="text", text=text)]


async def _list_agents(args: dict) -> list[TextContent]:
    app_id = args.get("app_id", "base")
    type_filter = args.get("type_filter")
    domain_filter = args.get("domain_filter")

    conn = _get_conn()
    from execution_graph import list_agent_cards
    cards = list_agent_cards(conn, type_filter=type_filter,
                             domain_filter=domain_filter, app_id=app_id)
    lines = [f"Agents visible to '{app_id}' ({len(cards)} total):", ""]
    for c in cards:
        hitl = " [HITL]" if c.get("hitl_required") else ""
        lines.append(f"  {c['agent_id']}: {c['name']} ({c['type']}){hitl}")
    return [TextContent(type="text", text="\n".join(lines))]


async def _list_apps(args: dict) -> list[TextContent]:
    from app_registry import list_apps, load_app_manifest
    apps = list_apps()
    lines = [f"Registered applications ({len(apps)}):", ""]
    for app_id in apps:
        try:
            manifest = load_app_manifest(app_id)
            lines.append(f"  {app_id}: {manifest['name']} — {manifest['description']}")
            lines.append(f"    Solutions: {', '.join(manifest.get('solutions', []))}")
            lines.append(f"    Depends on: {', '.join(manifest.get('depends_on', [])) or 'none'}")
            lines.append("")
        except Exception:
            lines.append(f"  {app_id}: (manifest error)")
    return [TextContent(type="text", text="\n".join(lines))]


async def _list_solutions(args: dict) -> list[TextContent]:
    app_id = args["app_id"]
    from app_registry import list_solutions, load_solution_manifest
    sols = list_solutions(app_id)
    lines = [f"Solutions for '{app_id}' ({len(sols)}):", ""]
    for sol_id in sols:
        try:
            manifest = load_solution_manifest(app_id, sol_id)
            agents = ", ".join(manifest.get("primary_agents", []))
            lines.append(f"  {sol_id}: {manifest['name']}")
            lines.append(f"    {manifest['description']}")
            lines.append(f"    Agents: {agents}")
            lines.append(f"    Patterns: {', '.join(manifest.get('intent_patterns', []))}")
            lines.append("")
        except Exception:
            lines.append(f"  {sol_id}: (manifest error)")
    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@server.list_resources()
async def list_resources() -> list[Resource]:
    resources = [
        Resource(
            uri="deso://apps",
            name="DESO Applications",
            description="All registered DESO applications",
            mimeType="application/json",
        ),
    ]
    from app_registry import list_apps
    for app_id in list_apps():
        resources.extend([
            Resource(
                uri=f"deso://apps/{app_id}",
                name=f"{app_id} manifest",
                description=f"Application manifest for {app_id}",
                mimeType="application/json",
            ),
            Resource(
                uri=f"deso://apps/{app_id}/solutions",
                name=f"{app_id} solutions",
                description=f"Solutions for {app_id}",
                mimeType="application/json",
            ),
            Resource(
                uri=f"deso://apps/{app_id}/agents",
                name=f"{app_id} agents",
                description=f"Visible agents for {app_id}",
                mimeType="application/json",
            ),
        ])
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    from app_registry import (
        list_apps, load_app_manifest, list_solutions,
        load_solution_manifest, get_visible_agents,
    )

    if uri == "deso://apps":
        apps = []
        for app_id in list_apps():
            try:
                apps.append(load_app_manifest(app_id))
            except Exception:
                apps.append({"app_id": app_id, "error": "manifest not found"})
        return json.dumps(apps, indent=2)

    parts = uri.replace("deso://apps/", "").split("/")
    app_id = parts[0] if parts else None

    if not app_id:
        return json.dumps({"error": f"Invalid URI: {uri}"})

    if len(parts) == 1:
        # deso://apps/{app_id}
        try:
            return json.dumps(load_app_manifest(app_id), indent=2)
        except FileNotFoundError:
            return json.dumps({"error": f"App not found: {app_id}"})

    sub = parts[1]
    if sub == "solutions":
        sols = []
        for sol_id in list_solutions(app_id):
            try:
                sols.append(load_solution_manifest(app_id, sol_id))
            except Exception:
                sols.append({"solution_id": sol_id, "error": "manifest not found"})
        return json.dumps(sols, indent=2)

    if sub == "agents":
        agents = get_visible_agents(app_id)
        return json.dumps({"app_id": app_id, "agents": agents})

    return json.dumps({"error": f"Unknown resource path: {uri}"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    await run_stdio(server)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
