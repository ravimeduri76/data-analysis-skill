"""
test_harmonizer.py — TC12: AG09 Harmonizer reference implementation tests.

Run: pytest hooks/tests/test_harmonizer.py -v
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

_HOOKS_DIR = os.path.join(os.path.expanduser("~"), "research", "data-analysis")
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)

from hooks.agents.harmonizer import (
    apply_controlled_vocabulary,
    apply_entity_resolution,
    check_invariants,
    completeness_report,
    harmonize,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

GRAPH_ENTITIES = [
    {"name": "Hewlett-Packard", "related_terms": ["HP", "hp inc", "hewlett packard"]},
    {"name": "Dell Technologies", "related_terms": ["Dell", "dell inc"]},
    {"name": "IEEE", "related_terms": ["ieee", "I.E.E.E."]},
]

TERMINOLOGY = [
    {"term": "Desktop Computer", "related_terms": ["desktop", "desk top", "pc"]},
    {"term": "Laptop", "related_terms": ["notebook", "portable computer"]},
    {"term": "Server", "related_terms": ["rack server", "tower server"]},
]

PROPERTY_MODEL = [
    {
        "entity": "Product",
        "properties": [
            {"name": "brand", "enumerated_values": ["Hewlett-Packard", "Dell Technologies", "Lenovo"]},
            {"name": "category", "enumerated_values": ["Desktop Computer", "Laptop", "Server"]},
            {"name": "sku"},
            {"name": "price"},
        ],
    },
]

INVARIANTS = [
    {"rule": "sku NOT NULL", "description": "Every product must have a SKU"},
    {"rule": "category IN ('DESKTOP COMPUTER', 'LAPTOP', 'SERVER')", "description": "Category must be valid"},
    {"rule": "sku UNIQUE", "description": "SKUs must be unique"},
]


# ── TC12 Tests ─────────────────────────────────────────────────────────────

class TestTC12Harmonizer:

    def test_harmonize_with_terminology(self):
        df = pd.DataFrame({"category": ["desktop", "notebook", "rack server"]})
        dcp = {"terminology": TERMINOLOGY}
        result_df, report = harmonize(df, dcp)
        assert result_df["category"].tolist() == ["Desktop Computer", "Laptop", "Server"]

    def test_entity_resolution_maps_aliases(self):
        series = pd.Series(["HP", "Dell", "IEEE"])
        resolved, report = apply_entity_resolution(series, GRAPH_ENTITIES)
        assert resolved.iloc[0] == "Hewlett-Packard"
        assert resolved.iloc[1] == "Dell Technologies"
        assert resolved.iloc[2] == "IEEE"
        assert report["resolved"] == 3

    def test_controlled_vocabulary_normalization(self):
        series = pd.Series(["desktop", "notebook", "tower server"])
        normalized, report = apply_controlled_vocabulary(series, TERMINOLOGY)
        assert normalized.iloc[0] == "Desktop Computer"
        assert normalized.iloc[1] == "Laptop"
        assert normalized.iloc[2] == "Server"
        assert report["normalized"] == 3

    def test_check_invariants_catches_violations(self):
        df = pd.DataFrame({
            "sku": ["A001", None, "A001"],  # null + duplicate
            "category": ["DESKTOP COMPUTER", "INVALID", "LAPTOP"],
        })
        violations = check_invariants(df, INVARIANTS)
        # Should catch: sku null, sku duplicate, category invalid
        assert len(violations) >= 2
        violation_types = [v["invariant"] for v in violations]
        assert any("NOT NULL" in v for v in violation_types)
        assert any("UNIQUE" in v for v in violation_types)

    def test_completeness_report_per_property_group(self):
        df = pd.DataFrame({
            "brand": ["HP", "Dell", None],
            "category": ["Desktop", "Laptop", "Server"],
            "sku": ["A1", None, "A3"],
            "price": [100, 200, 300],
        })
        report = completeness_report(df, PROPERTY_MODEL)
        assert "Product" in report
        assert report["Product"]["brand"] == pytest.approx(2 / 3, abs=0.01)
        assert report["Product"]["category"] == 1.0
        assert report["Product"]["sku"] == pytest.approx(2 / 3, abs=0.01)
        assert report["Product"]["price"] == 1.0

    def test_harmonize_empty_dcp_is_noop(self):
        df = pd.DataFrame({"col": ["abc", "def"]})
        result_df, report = harmonize(df, {})
        assert result_df["col"].tolist() == ["Abc", "Def"]  # only grammar cleanup
        assert report["entity_resolution"] == {}
        assert report["vocabulary"] == {}

    def test_entity_resolution_fuzzy_match(self):
        series = pd.Series(["Hewlet Packard"])  # typo
        resolved, report = apply_entity_resolution(series, GRAPH_ENTITIES)
        assert resolved.iloc[0] == "Hewlett-Packard"
        assert report["resolved"] == 1
