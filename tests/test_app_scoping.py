"""
test_app_scoping.py — Verify application isolation in graph queries and registry.

Tests cover:
- App manifest loading
- Solution manifest loading
- Query-to-app resolution
- Agent visibility per app
- App_id scoping in hook APIs (smoke tests — require DB)
"""

from __future__ import annotations

import os
import sys
import pytest

# Add hooks to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))


# ---------------------------------------------------------------------------
# App Registry tests (no DB required)
# ---------------------------------------------------------------------------

class TestAppRegistry:
    """Tests for hooks/app_registry.py — manifest loading and resolution."""

    def test_list_apps(self):
        from app_registry import list_apps
        apps = list_apps()
        assert "base" in apps
        assert "dataaxl" in apps
        assert "careaxl" in apps

    def test_load_app_manifest(self):
        from app_registry import load_app_manifest
        base = load_app_manifest("base")
        assert base["app_id"] == "base"
        assert base["name"] == "Base"
        assert isinstance(base["shared_agents"], list)

    def test_load_app_manifest_dataaxl(self):
        from app_registry import load_app_manifest
        app = load_app_manifest("dataaxl")
        assert app["app_id"] == "dataaxl"
        assert "base" in app["depends_on"]

    def test_load_app_manifest_not_found(self):
        from app_registry import load_app_manifest
        with pytest.raises(FileNotFoundError):
            load_app_manifest("nonexistent_app")

    def test_list_solutions(self):
        from app_registry import list_solutions
        sols = list_solutions("dataaxl")
        assert "product_data_prep" in sols
        assert "product_data_syndication" in sols
        assert "customer_data_prep" in sols

    def test_list_solutions_base(self):
        from app_registry import list_solutions
        sols = list_solutions("base")
        assert "ontology_builder" in sols
        assert "system_admin" in sols

    def test_load_solution_manifest(self):
        from app_registry import load_solution_manifest
        sol = load_solution_manifest("dataaxl", "product_data_prep")
        assert sol["solution_id"] == "product_data_prep"
        assert sol["app_id"] == "dataaxl"
        assert "AG04" in sol["primary_agents"]

    def test_load_solution_manifest_not_found(self):
        from app_registry import load_solution_manifest
        with pytest.raises(FileNotFoundError):
            load_solution_manifest("dataaxl", "nonexistent_solution")

    def test_get_visible_agents_base(self):
        from app_registry import get_visible_agents
        agents = get_visible_agents("base")
        assert "AG01" in agents
        assert "AG24" in agents

    def test_get_visible_agents_dataaxl(self):
        from app_registry import get_visible_agents
        agents = get_visible_agents("dataaxl")
        # Should include base shared agents
        assert "AG01" in agents
        # Should include dataaxl solution primary agents
        assert "AG09" in agents  # from product_data_prep
        assert "AG34" in agents  # from product_data_syndication

    def test_resolve_app_product_data_prep(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("I need to run product data prep from Syndigo")
        assert app_id == "dataaxl"
        assert sol_id == "product_data_prep"

    def test_resolve_app_syndigo_ingest(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("syndigo ingest pipeline")
        assert app_id == "dataaxl"
        assert sol_id == "product_data_prep"

    def test_resolve_app_customer_match(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("customer dedup and golden record build")
        assert app_id == "dataaxl"
        assert sol_id == "customer_data_prep"

    def test_resolve_app_probus(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("run probus analytics")
        assert app_id == "careaxl"
        assert sol_id == "probus"

    def test_resolve_app_reema(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("reema community health pipeline")
        assert app_id == "careaxl"
        assert sol_id == "reema"

    def test_resolve_app_fallback_to_base(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("some random unrelated query")
        assert app_id == "base"
        assert sol_id is None

    def test_resolve_app_ontology_builder(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("build ontology for healthcare domain")
        assert app_id == "base"
        assert sol_id == "ontology_builder"

    def test_resolve_app_product_syndication(self):
        from app_registry import resolve_app_from_query
        app_id, sol_id = resolve_app_from_query("syndicate product data to amazon feed")
        assert app_id == "dataaxl"
        assert sol_id == "product_data_syndication"


# ---------------------------------------------------------------------------
# DB-dependent tests (skip if no DB connection available)
# ---------------------------------------------------------------------------

def _get_test_conn():
    """Try to get a DB connection; return None if unavailable."""
    try:
        from domain_graph import get_domain_conn
        conn = get_domain_conn()
        return conn
    except Exception:
        return None


@pytest.fixture
def db_conn():
    """Fixture that provides a DB connection or skips."""
    conn = _get_test_conn()
    if conn is None:
        pytest.skip("No database connection available")
    yield conn
    conn.close()


class TestAppScopingDB:
    """DB-dependent tests for app_id scoping in hook APIs."""

    def test_migration_idempotent(self, db_conn):
        from db_migrate import run_migration
        # Should not raise on repeated calls
        run_migration(db_conn)
        run_migration(db_conn)

    def test_domain_graph_app_scoping(self, db_conn):
        from domain_graph import get_domain_context
        # Default app_id='base' should work
        ctx = get_domain_context(db_conn, "Industrial Component", "relay types")
        assert isinstance(ctx, dict)

    def test_intent_graph_app_scoping(self, db_conn):
        from intent_graph import find_candidate_intents
        # Default app_id='base' should work
        results = find_candidate_intents(db_conn, "survival analysis", limit=5)
        assert isinstance(results, list)

    def test_execution_graph_app_scoping(self, db_conn):
        from execution_graph import list_agent_cards
        # Default app_id='base' should work
        cards = list_agent_cards(db_conn)
        assert isinstance(cards, list)

    def test_base_data_visible_from_dataaxl(self, db_conn):
        from execution_graph import list_agent_cards
        # DataAXL should see base agents (app_id='base' OR app_id='dataaxl')
        base_cards = list_agent_cards(db_conn, app_id="base")
        dataaxl_cards = list_agent_cards(db_conn, app_id="dataaxl")
        # DataAXL sees at least as many as base (base data visible via OR clause)
        assert len(dataaxl_cards) >= len(base_cards)
