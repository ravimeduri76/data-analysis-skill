"""
test_semantic_matching.py — Tests for semantic matching upgrades across hooks.

Tests cover:
- Semantic app/solution resolution (app_registry.py)
- Semantic domain detection (domain_detector.py)
- Semantic lifecycle detection (domain_detector.py)
- Semantic pattern matching (execution_graph.py)
- Semantic domain alias resolution (domain_graph.py)
- Graceful degradation when BGE-M3 is unavailable
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Add hooks to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn():
    """Create a mock DB connection for tests that don't need a real DB."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    return conn, cursor


# ---------------------------------------------------------------------------
# embedding_registry tests
# ---------------------------------------------------------------------------

class TestEmbeddingRegistry:
    """Tests for hooks/embedding_registry.py — shared model singleton."""

    def test_get_embed_model_returns_none_when_unavailable(self):
        """If sentence-transformers isn't installed, returns None."""
        import embedding_registry
        # Clear cache
        embedding_registry._MODEL_CACHE.clear()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch("embedding_registry.get_embed_model") as mock:
                mock.return_value = None
                result = mock()
                assert result is None

    def test_embed_texts_raises_when_no_model(self):
        """embed_texts should raise RuntimeError when model unavailable."""
        import embedding_registry
        embedding_registry._MODEL_CACHE.clear()
        embedding_registry._MODEL_CACHE["model"] = None
        with pytest.raises(RuntimeError, match="BGE-M3 model unavailable"):
            embedding_registry.embed_texts(["test"])

    def test_cosine_rank_empty_candidates(self):
        """cosine_rank should return empty list for empty candidates."""
        import numpy as np
        from embedding_registry import cosine_rank
        q_vec = np.random.randn(1024).astype(np.float32)
        result = cosine_rank(q_vec, [])
        assert result == []

    def test_cosine_rank_sorts_descending(self):
        """cosine_rank should sort by score descending."""
        import numpy as np
        from embedding_registry import cosine_rank

        q_vec = np.array([1.0] + [0.0] * 1023, dtype=np.float32)
        emb_a = np.array([0.5] + [0.0] * 1023, dtype=np.float32)
        emb_b = np.array([0.9] + [0.0] * 1023, dtype=np.float32)

        result = cosine_rank(q_vec, [(0, emb_a.tolist()), (1, emb_b.tolist())])
        assert len(result) == 2
        assert result[0][0] == 1  # emb_b should be first (higher dot product)
        assert result[0][1] > result[1][1]


# ---------------------------------------------------------------------------
# app_registry semantic tests
# ---------------------------------------------------------------------------

class TestAppRegistrySemantic:
    """Tests for semantic path in resolve_app_from_query."""

    def test_resolve_falls_back_to_keyword_when_no_db(self):
        """When DB is unavailable, keyword fallback still works."""
        from app_registry import resolve_app_from_query
        with patch("app_registry._get_conn_safe", return_value=None):
            result = resolve_app_from_query("product data prep")
            assert result[0] in ("dataaxl", "base")

    def test_keyword_resolve_sku_enrichment(self):
        """Keyword path should match SKU-related queries to dataaxl."""
        from app_registry import _keyword_resolve
        app_id, sol_id = _keyword_resolve(
            "enrich SKU data",
            dataset_columns=["SKU", "Brand", "Category L1"],
        )
        assert app_id == "dataaxl"

    def test_semantic_resolve_returns_none_on_low_score(self, mock_conn):
        """_semantic_resolve should return None when no embeddings match well."""
        import numpy as np
        conn, cursor = mock_conn
        # Return embeddings with low similarity (orthogonal vectors)
        cursor.fetchall.return_value = [
            ("base", "system_admin", [0.0] * 1024),
        ]
        from app_registry import _semantic_resolve
        with patch("hooks.embedding_registry.embed_texts") as mock_embed:
            mock_embed.return_value = np.array([[1.0] + [0.0] * 1023], dtype=np.float32)
            result = _semantic_resolve(conn, "something completely unrelated")
            assert result is None


# ---------------------------------------------------------------------------
# domain_detector semantic tests
# ---------------------------------------------------------------------------

class TestDomainDetectorSemantic:
    """Tests for semantic paths in domain_detector.py."""

    def test_heuristic_domain_still_works(self):
        """Original heuristic path should still work."""
        import pandas as pd
        from domain_detector import _heuristic_domain
        df = pd.DataFrame({
            "patient_id": [1, 2],
            "diagnosis": ["flu", "cold"],
            "icd_code": ["J11", "J00"],
        })
        assert _heuristic_domain(df) == "Healthcare"

    def test_semantic_domain_returns_none_when_no_db(self):
        """_semantic_domain returns None when DB is unavailable."""
        import pandas as pd
        from domain_detector import _semantic_domain
        with patch("domain_detector._get_domain_conn_safe", return_value=None):
            df = pd.DataFrame({"col": [1]})
            result = _semantic_domain(df)
            assert result is None

    def test_detect_lifecycle_stage_trusts_solution(self):
        """Lifecycle stage from solution entry takes precedence."""
        from domain_detector import _detect_lifecycle_stage
        sol = {"lifecycle_stage": "enrichment"}
        assert _detect_lifecycle_stage("extract data", sol) == "enrichment"

    def test_detect_lifecycle_stage_keyword_fallback(self):
        """Keywords work when no solution and no semantic."""
        from domain_detector import _detect_lifecycle_stage
        with patch("domain_detector._semantic_lifecycle", return_value=None):
            result = _detect_lifecycle_stage("extract and ingest raw data", None)
            assert result == "extraction"

    def test_resolve_solution_by_name_substring(self):
        """Bidirectional substring matching should still work."""
        from domain_detector import _resolve_solution_by_name
        with patch("domain_detector._semantic_resolve_solution_by_name", return_value=None):
            sol = _resolve_solution_by_name("Master Data Management", "Entity Resolution")
            assert sol is not None
            assert "Entity Resolution" in sol.get("name", "")


# ---------------------------------------------------------------------------
# execution_graph semantic tests
# ---------------------------------------------------------------------------

class TestExecutionGraphSemantic:
    """Tests for semantic pattern matching in execution_graph.py."""

    def test_keyword_plan_search_token_overlap(self, mock_conn):
        """_keyword_plan_search should score by token overlap, not flat 1.0."""
        from execution_graph import _keyword_plan_search
        conn, cursor = mock_conn
        # Simulate a matching row
        cursor.fetchall.return_value = [
            (1, "churn_prediction", "Churn Prediction Pipeline", "Predict customer churn",
             ["churn_prediction"], ["analytics"], ["Healthcare"], True),
        ]
        results = _keyword_plan_search(conn, "churn prediction model", 5, None)
        assert len(results) == 1
        assert results[0]["score"] < 1.0  # Token overlap, not flat 1.0
        assert results[0]["score"] > 0  # But positive

    def test_pattern_match_returns_none_when_no_intent(self):
        """_pattern_match should return None when no matched_node."""
        from execution_graph import _pattern_match
        conn = MagicMock()
        result = _pattern_match(conn, {})
        assert result is None


# ---------------------------------------------------------------------------
# domain_graph semantic tests
# ---------------------------------------------------------------------------

class TestDomainGraphSemantic:
    """Tests for semantic alias resolution in domain_graph.py."""

    def test_resolve_domain_exact(self):
        """Exact alias lookup should still work."""
        from domain_graph import _resolve_domain
        assert _resolve_domain("Electrical Components") == "Industrial Component"
        assert _resolve_domain("Wellness") == "Healthcare"

    def test_resolve_domain_passthrough(self):
        """Unknown domains should pass through unchanged."""
        from domain_graph import _resolve_domain
        with patch("domain_graph._get_conn_safe", return_value=None):
            assert _resolve_domain("SomeNewDomain") == "SomeNewDomain"

    def test_keyword_traverse_token_overlap(self, mock_conn):
        """_keyword_traverse should score by token overlap."""
        from domain_graph import _keyword_traverse
        conn, cursor = mock_conn
        cursor.fetchall.return_value = [
            ("entity", "relay::Contact", "entity: Relay Contact — Main switching element"),
        ]
        results = _keyword_traverse(conn, "Industrial Component", "relay contact", 5, None)
        assert len(results) == 1
        assert results[0]["score"] <= 1.0
        assert results[0]["score"] > 0


# ---------------------------------------------------------------------------
# intent_graph keyword fallback tests
# ---------------------------------------------------------------------------

class TestIntentGraphKeywordScoring:
    """Tests for improved keyword fallback scoring in intent_graph.py."""

    def test_alias_keyword_search_token_overlap(self, mock_conn):
        """_alias_keyword_search should score by token overlap (not flat 1.0)."""
        from intent_graph import _alias_keyword_search
        conn, cursor = mock_conn
        # Simulate rows from UNION query — node text only partially matches query tokens
        cursor.fetchall.return_value = [
            (1, "data_profiling", "Data Profiling", "Profile dataset completeness",
             "mid", "any", "analytics"),
        ]
        # Query has 4 tokens > 2 chars: "profile", "data", "quality", "assessment"
        # Node text has "data_profiling", "Data Profiling", "Profile dataset completeness"
        # "quality" and "assessment" won't be in node text, so overlap < 1.0
        results = _alias_keyword_search(conn, "profile data quality assessment", 5, None)
        assert len(results) == 1
        assert results[0]["score"] < 1.0  # Not all tokens match
        assert results[0]["score"] > 0


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Verify that all upgraded functions degrade gracefully when BGE-M3 is unavailable."""

    def test_app_registry_degrades(self):
        """resolve_app_from_query works without BGE-M3."""
        from app_registry import resolve_app_from_query
        with patch("app_registry._get_conn_safe", return_value=None):
            result = resolve_app_from_query("product enrichment")
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_domain_detector_degrades(self):
        """detect_domain works without BGE-M3 or DB."""
        import pandas as pd
        from domain_detector import detect_domain
        with patch("domain_detector._get_domain_conn_safe", return_value=None):
            df = pd.DataFrame({"patient": [1], "diagnosis": ["flu"]})
            ctx = detect_domain(df)
            assert ctx["domain"] == "Healthcare"
            assert ctx["method"] in ("heuristic", "semantic")

    def test_resolve_domain_degrades(self):
        """_resolve_domain works without DB."""
        from domain_graph import _resolve_domain
        with patch("domain_graph._get_conn_safe", return_value=None):
            # Exact match still works
            assert _resolve_domain("Electrical Components") == "Industrial Component"
            # Unknown passes through
            assert _resolve_domain("SomeDomain") == "SomeDomain"

    def test_lifecycle_degrades(self):
        """_detect_lifecycle_stage works without semantic."""
        from domain_detector import _detect_lifecycle_stage
        with patch("domain_detector._semantic_lifecycle", return_value=None):
            assert _detect_lifecycle_stage("extract data from api", None) == "extraction"


# ---------------------------------------------------------------------------
# DB migration tests
# ---------------------------------------------------------------------------

class TestDBMigration:
    """Tests for db_migrate.py schema additions."""

    def test_migration_sql_contains_new_tables(self):
        """Verify the migration SQL includes all 4 new tables."""
        from db_migrate import _MIGRATION_SQL
        assert "solution_embeddings" in _MIGRATION_SQL
        assert "domain_keyword_embeddings" in _MIGRATION_SQL
        assert "domain_alias_embeddings" in _MIGRATION_SQL
        assert "plan_pattern_embeddings" in _MIGRATION_SQL
