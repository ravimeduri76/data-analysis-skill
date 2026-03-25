"""
trace_writer.py -- Audit breadcrumbs linking steps, artifacts, and InsightCards
into a verifiable manifest.

Public API
----------
AnalysisTrace(dataset_path, query, domain_ctx)
    .log_step(step_name, description, params, result_summary)
    .log_artifact(file_path, description, row_count, col_count)
    .log_insight(card)
    .save_manifest(output_dir) -> (json_path, md_path)
    .render_breadcrumbs()      -> str
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from hooks.explainability import InsightCard


def _sha256_file(path: str) -> str:
    """Compute SHA-256 of a file in 64 KB chunks. Returns 'unavailable' on any error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


def _safe_cell(value: object) -> str:
    """Sanitize a value for use in a Markdown table cell."""
    return str(value).replace("|", "/").replace("\n", " ")


class AnalysisTrace:
    """Collects steps, artifacts, and insights for a single analysis run."""

    def __init__(self, dataset_path: str, query: str, domain_ctx: dict) -> None:
        self.dataset_path = dataset_path
        self.query = query
        self.domain_ctx = domain_ctx
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.file_sha256 = _sha256_file(dataset_path)
        self._steps: list = []
        self._artifacts: list = []
        self._insights: list = []
        self._manifest_files: list = []

    # ── logging ───────────────────────────────────────────────────────────────

    def log_step(
        self,
        step_name: str,
        description: str,
        params: dict,
        result_summary: str,
    ) -> "AnalysisTrace":
        """Append a named step to the trace."""
        self._steps.append({
            "step": step_name,
            "description": description,
            "params": json.loads(json.dumps(params, default=str)),
            "result_summary": result_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return self

    def log_artifact(
        self,
        file_path: str,
        description: str,
        row_count: int,
        col_count: int,
    ) -> "AnalysisTrace":
        """Record a saved output file."""
        self._artifacts.append({
            "file": os.path.basename(file_path),
            "path": file_path,
            "description": description,
            "row_count": row_count,
            "col_count": col_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return self

    def log_insight(self, card: object) -> "AnalysisTrace":
        """Append an InsightCard (dataclass or plain object) to the trace."""
        try:
            if dataclasses.is_dataclass(card):
                self._insights.append(dataclasses.asdict(card))  # type: ignore[arg-type]
            else:
                self._insights.append(card.__dict__.copy())
        except Exception:
            self._insights.append({"error": "Could not serialize insight card"})
        return self

    # ── manifest output ───────────────────────────────────────────────────────

    def save_manifest(self, output_dir: str) -> Tuple[str, str]:
        """Write trace_{run_id}.json and trace_{run_id}.md.

        Returns (json_path, md_path).
        """
        os.makedirs(output_dir, exist_ok=True)
        base = f"trace_{self.run_id}"
        json_path = os.path.join(output_dir, f"{base}.json")
        md_path = os.path.join(output_dir, f"{base}.md")

        # ── machine-readable JSON ────────────────────────────────────────────
        manifest = {
            "run_id": self.run_id,
            "dataset_path": self.dataset_path,
            "file_sha256": self.file_sha256,
            "query": self.query,
            "domain": self.domain_ctx.get("domain", "Unknown"),
            "steps": self._steps,
            "artifacts": self._artifacts,
            "insights": self._insights,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        # ── SME-friendly Markdown ────────────────────────────────────────────
        sha_short = (
            self.file_sha256[:12]
            if self.file_sha256 != "unavailable"
            else "unavailable"
        )
        domain = self.domain_ctx.get("domain", "Unknown")

        md_lines = [
            "# Analysis Trace Report",
            "",
            (
                f"**Run ID:** {self.run_id}"
                f"   **SHA-256:** {sha_short}..."
                f"   **Domain:** {domain}"
            ),
            "",
            "## Steps",
            "",
            "| # | Step | Description | Result | Timestamp |",
            "|---|------|-------------|--------|-----------|",
        ]
        for i, step in enumerate(self._steps, 1):
            md_lines.append(
                f"| {i} | {_safe_cell(step['step'])} | {_safe_cell(step['description'])} "
                f"| {_safe_cell(step['result_summary'])} | {_safe_cell(step['timestamp'])} |"
            )

        md_lines += [
            "",
            "## Artifacts  (check boxes for SME sign-off)",
            "",
            "| # | File | Description | Rows | Cols | Sign-off |",
            "|---|------|-------------|------|------|----------|",
        ]
        for i, art in enumerate(self._artifacts, 1):
            md_lines.append(
                f"| {i} | {_safe_cell(art['file'])} | {_safe_cell(art['description'])} "
                f"| {art['row_count']} | {art['col_count']} | [ ] |"
            )

        md_lines += [
            "",
            "## Key Insights",
            "",
            "| # | Finding | Method | Confidence |",
            "|---|---------|--------|------------|",
        ]
        for i, ins in enumerate(self._insights, 1):
            finding = _safe_cell(ins.get("finding", ""))
            method = _safe_cell(ins.get("method", ""))
            confidence = _safe_cell(ins.get("confidence", ""))
            md_lines.append(f"| {i} | {finding} | {method} | {confidence} |")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        self._manifest_files = [json_path, md_path]
        return json_path, md_path

    # ── console summary ───────────────────────────────────────────────────────

    def render_breadcrumbs(self) -> str:
        """Return a short ASCII console summary of this trace."""
        sha_short = (
            self.file_sha256[:12]
            if self.file_sha256 != "unavailable"
            else "unavailable"
        )
        dataset_name = os.path.basename(self.dataset_path)
        if self._manifest_files:
            manifest_str = f"trace_{self.run_id}.json + .md"
        else:
            manifest_str = "not saved yet"

        lines = [
            "=== ANALYSIS TRACE ===",
            f"Run ID   : {self.run_id}",
            f"Dataset  : {dataset_name}",
            f"SHA-256  : {sha_short}",
            f"Domain   : {self.domain_ctx.get('domain', 'Unknown')}",
            f"Steps    : {len(self._steps)} logged",
            f"Artifacts: {len(self._artifacts)} saved",
            f"Insights : {len(self._insights)} captured",
            f"Manifests: {manifest_str}",
            "======================",
        ]
        result = "\n".join(lines)
        return result.encode("ascii", errors="replace").decode("ascii")
