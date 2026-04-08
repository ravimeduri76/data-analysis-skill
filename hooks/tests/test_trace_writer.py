"""
test_trace_writer.py — TC13: Context Packet Save tests.

Validates that log_packet stores data, save_manifest includes packets
in both JSON and Markdown, and numpy types are handled safely.

Run: pytest hooks/tests/test_trace_writer.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

_HOOKS_DIR = os.path.join(os.path.expanduser("~"), "research", "data-analysis")
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)

from hooks.trace_writer import AnalysisTrace


def _make_trace() -> AnalysisTrace:
    return AnalysisTrace(
        dataset_path="test_data.csv",
        query="test query",
        domain_ctx={"domain": "Test", "confidence": "high"},
    )


class TestTC13ContextPacketSave:

    def test_log_packet_stores_data(self):
        trace = _make_trace()
        trace.log_packet("DCP", {"domain": "Healthcare", "confidence": "high"})
        assert len(trace._packets) == 1
        assert trace._packets[0]["name"] == "DCP"
        assert trace._packets[0]["data"]["domain"] == "Healthcare"

    def test_save_manifest_includes_context_packets_json(self):
        trace = _make_trace()
        trace.log_packet("DCP", {"domain": "Healthcare"})
        trace.log_packet("IS", {"explicit_goal": "Analyse data"})

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, _ = trace.save_manifest(tmpdir)
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        assert "context_packets" in manifest
        assert "DCP" in manifest["context_packets"]
        assert "IS" in manifest["context_packets"]
        assert manifest["context_packets"]["DCP"]["domain"] == "Healthcare"

    def test_save_manifest_includes_packets_section_md(self):
        trace = _make_trace()
        trace.log_packet("AP", {"agents": ["AG07", "AG09"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            _, md_path = trace.save_manifest(tmpdir)
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()

        assert "## Context Packets" in md_content
        assert "AP" in md_content

    def test_log_packet_handles_numpy_types(self):
        """numpy int64 and similar types must serialize without error."""
        try:
            import numpy as np
            has_numpy = True
        except ImportError:
            has_numpy = False

        trace = _make_trace()

        if has_numpy:
            data = {"count": np.int64(42), "score": np.float64(0.95)}
        else:
            # Fallback: test with regular types (still validates the path)
            data = {"count": 42, "score": 0.95}

        trace.log_packet("TEST", data)
        # numpy types get serialized via default=str, so int64→str "42"
        # The key point is that it doesn't raise — data is safely stored
        assert "count" in trace._packets[0]["data"]

        # Also verify it saves to JSON without error
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, _ = trace.save_manifest(tmpdir)
            with open(json_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            assert "count" in manifest["context_packets"]["TEST"]
