"""
app_registry.py — Application and solution manifest loading + resolution.

Loads static JSON manifests from apps/{app_id}/app.json and
apps/{app_id}/solutions/{solution_id}.json. Resolves user queries to
(app_id, solution_id) by matching against solution intent_patterns.

Public API
----------
load_app_manifest(app_id)                        -> dict
load_solution_manifest(app_id, solution_id)      -> dict
list_apps()                                      -> list[str]
list_solutions(app_id)                           -> list[str]
get_visible_agents(app_id)                       -> list[str]
resolve_app_from_query(query)                    -> tuple[str, str | None]
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_APPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "apps",
)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_app_manifest(app_id: str) -> dict:
    """Load an application manifest from apps/{app_id}/app.json.

    Returns the parsed JSON dict. Raises FileNotFoundError if not found.
    """
    path = os.path.join(_APPS_DIR, app_id, "app.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_solution_manifest(app_id: str, solution_id: str) -> dict:
    """Load a solution manifest from apps/{app_id}/solutions/{solution_id}.json.

    Returns the parsed JSON dict. Raises FileNotFoundError if not found.
    """
    path = os.path.join(_APPS_DIR, app_id, "solutions", f"{solution_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_apps() -> list[str]:
    """Return a sorted list of registered application IDs (directory names under apps/)."""
    if not os.path.isdir(_APPS_DIR):
        return []
    return sorted(
        d for d in os.listdir(_APPS_DIR)
        if os.path.isfile(os.path.join(_APPS_DIR, d, "app.json"))
    )


def list_solutions(app_id: str) -> list[str]:
    """Return a sorted list of solution IDs for the given application."""
    solutions_dir = os.path.join(_APPS_DIR, app_id, "solutions")
    if not os.path.isdir(solutions_dir):
        return []
    return sorted(
        f.removesuffix(".json")
        for f in os.listdir(solutions_dir)
        if f.endswith(".json")
    )


# ---------------------------------------------------------------------------
# Agent visibility
# ---------------------------------------------------------------------------

def get_visible_agents(app_id: str) -> list[str]:
    """Return the list of agent IDs visible to an application.

    Includes:
    - Base shared_agents (always)
    - The app's own shared_agents
    - All primary_agents from the app's solutions
    """
    agents: set[str] = set()

    # Always include base shared agents
    try:
        base = load_app_manifest("base")
        agents.update(base.get("shared_agents", []))
    except FileNotFoundError:
        pass

    # App's own shared agents
    if app_id != "base":
        try:
            app = load_app_manifest(app_id)
            agents.update(app.get("shared_agents", []))
        except FileNotFoundError:
            pass

    # Primary agents from all solutions
    for sol_id in list_solutions(app_id):
        try:
            sol = load_solution_manifest(app_id, sol_id)
            agents.update(sol.get("primary_agents", []))
        except FileNotFoundError:
            pass

    return sorted(agents)


# ---------------------------------------------------------------------------
# DB connection (safe — returns None if unavailable)
# ---------------------------------------------------------------------------

def _get_conn_safe():
    """Return DB connection or None (no crash if DB down)."""
    try:
        import psycopg2
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
        url = os.environ.get(
            "DOMAIN_DB_URL",
            "postgresql://postgres:memorystorepg@localhost:5432/domain",
        )
        return psycopg2.connect(url)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Semantic resolution (primary path)
# ---------------------------------------------------------------------------

def _semantic_resolve(
    conn,
    query: str,
    dataset_columns: Optional[list[str]] = None,
) -> Optional[tuple[str, Optional[str]]]:
    """Resolve query to (app_id, solution_id) using embedding similarity.

    Returns None if no confident match (triggers keyword fallback).
    """
    try:
        from hooks.embedding_registry import embed_texts, cosine_rank
    except Exception:
        return None

    # Build embed text
    parts = [query]
    if dataset_columns:
        parts.append(f"Columns: {', '.join(dataset_columns[:30])}")
    embed_text = ". ".join(parts)

    try:
        q_vec = embed_texts([embed_text])[0]
    except Exception:
        return None

    # Fetch solution embeddings
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT app_id, solution_id, embedding FROM solution_embeddings"
            )
            rows = cur.fetchall()
    except Exception:
        return None

    if not rows:
        return None

    # Cosine rank
    candidates = [(i, row[2]) for i, row in enumerate(rows)]
    ranked = cosine_rank(q_vec, candidates)
    if not ranked:
        return None

    # Column signal boost: +0.10 per match (capped +0.30)
    col_text = " ".join(dataset_columns).lower() if dataset_columns else ""
    boosted: list[tuple[int, float]] = []
    for idx, score in ranked:
        if col_text:
            app_id = rows[idx][0]
            sol_id = rows[idx][1]
            try:
                sol = load_solution_manifest(app_id, sol_id)
                signal_boost = sum(
                    0.10 for s in sol.get("column_signals", [])
                    if s.lower() in col_text
                )
                score += min(signal_boost, 0.30)
            except FileNotFoundError:
                pass
        boosted.append((idx, score))
    boosted.sort(key=lambda x: x[1], reverse=True)

    top_idx, top_score = boosted[0]
    if top_score < 0.55:
        return None

    # Ambiguity check: if top two within 0.05 delta, use LLM to disambiguate
    if len(boosted) >= 2:
        second_idx, second_score = boosted[1]
        if top_score - second_score < 0.05:
            winner = _llm_disambiguate(
                query, dataset_columns,
                (rows[top_idx][0], rows[top_idx][1]),
                (rows[second_idx][0], rows[second_idx][1]),
            )
            if winner:
                return winner

    return (rows[top_idx][0], rows[top_idx][1])


def _llm_disambiguate(
    query: str,
    dataset_columns: Optional[list[str]],
    option_a: tuple[str, str],
    option_b: tuple[str, str],
) -> Optional[tuple[str, Optional[str]]]:
    """Use LLM to pick between two close-scoring solutions."""
    try:
        from hooks.llm_client import call_llm
    except ImportError:
        return None

    cols_str = ", ".join(dataset_columns[:20]) if dataset_columns else "unknown"
    prompt = (
        f"A user asked: \"{query}\"\n"
        f"Dataset columns: {cols_str}\n\n"
        f"Which solution best matches?\n"
        f"A: {option_a[0]}/{option_a[1]}\n"
        f"B: {option_b[0]}/{option_b[1]}\n\n"
        f"Reply with just A or B."
    )
    resp = call_llm(prompt, max_tokens=10)
    if resp and resp.strip().upper().startswith("A"):
        return option_a
    if resp and resp.strip().upper().startswith("B"):
        return option_b
    return None


# ---------------------------------------------------------------------------
# Keyword resolution (fallback)
# ---------------------------------------------------------------------------

def _keyword_resolve(
    query: str,
    dataset_columns: Optional[list[str]] = None,
) -> tuple[str, Optional[str]]:
    """Original keyword-based resolution logic."""
    query_lower = query.lower()
    query_tokens = set(re.split(r"[\s,._\-/]+", query_lower))

    col_tokens: set[str] = set()
    col_text = ""
    if dataset_columns:
        col_text = " ".join(dataset_columns).lower()
        col_tokens = set(re.split(r"[\s,._\-/]+", col_text))

    best: Optional[tuple[str, str, float]] = None

    for app_id in list_apps():
        app_domains: list[str] = []
        try:
            app_manifest = load_app_manifest(app_id)
            app_domains = [d.lower() for d in app_manifest.get("domains", [])]
        except FileNotFoundError:
            pass

        for sol_id in list_solutions(app_id):
            try:
                sol = load_solution_manifest(app_id, sol_id)
            except FileNotFoundError:
                continue

            score = 0.0
            keyword_hits: set[str] = set()

            for pattern in sol.get("intent_patterns", []):
                pattern_lower = pattern.lower()
                if pattern_lower in query_lower:
                    score += 3.0
                else:
                    pattern_tokens = set(re.split(r"[\s_\-]+", pattern_lower))
                    keyword_hits |= (query_tokens & pattern_tokens)

            score += len(keyword_hits)

            for signal in sol.get("column_signals", []):
                signal_lower = signal.lower()
                if signal_lower in col_text:
                    score += 2.0

            if col_tokens and app_domains:
                for domain in app_domains:
                    domain_words = set(re.split(r"[\s/]+", domain))
                    if domain_words & col_tokens:
                        score += 1.0

            if score > 0 and (best is None or score > best[2]):
                best = (app_id, sol_id, score)

    if best:
        return (best[0], best[1])

    return ("base", None)


# ---------------------------------------------------------------------------
# Query resolution (public API)
# ---------------------------------------------------------------------------

def resolve_app_from_query(
    query: str,
    dataset_columns: Optional[list[str]] = None,
) -> tuple[str, Optional[str]]:
    """Match a user query against solution manifests across all apps.

    Returns (app_id, solution_id). Falls back to ("base", None) if no match.

    Resolution order:
    1. Semantic matching (BGE-M3 embeddings + column signal boost + LLM disambiguation)
    2. Keyword fallback (substring + token overlap + column signals)
    """
    # Try semantic path first
    conn = _get_conn_safe()
    if conn:
        try:
            result = _semantic_resolve(conn, query, dataset_columns)
            if result:
                return result
        except Exception as exc:
            log.debug("Semantic resolve failed: %s", exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Keyword fallback
    return _keyword_resolve(query, dataset_columns)
