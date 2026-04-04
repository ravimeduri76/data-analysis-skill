"""
seed_embeddings.py — Pre-compute and store all static embeddings for semantic matching.

Idempotent: uses INSERT ... ON CONFLICT DO UPDATE so re-runs are safe.
Run once after db_migrate.py to populate the 4 embedding tables.

Usage:
    python hooks/seed_embeddings.py          # seeds everything
    python -c "from hooks.seed_embeddings import seed_all_embeddings; ..."

Public API
----------
seed_all_embeddings(conn)
seed_solution_embeddings(conn)
seed_domain_embeddings(conn)
seed_plan_pattern_embeddings(conn)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_HOOKS_DIR)
_APPS_DIR = os.path.join(_BASE_DIR, "apps")
_SOLUTIONS_PATH = os.path.join(_HOOKS_DIR, "domain_solutions.json")


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


_DEFAULT_URL = "postgresql://postgres:memorystorepg@localhost:5432/domain"


def _get_conn():
    import psycopg2
    _load_env()
    url = os.environ.get("DOMAIN_DB_URL", _DEFAULT_URL)
    return psycopg2.connect(url)


# ---------------------------------------------------------------------------
# Solution embeddings (from apps/*/solutions/*.json)
# ---------------------------------------------------------------------------

def seed_solution_embeddings(conn) -> int:
    """Embed all solution manifests and store in solution_embeddings. Returns count."""
    from hooks.embedding_registry import embed_texts

    if not os.path.isdir(_APPS_DIR):
        log.warning("No apps directory found at %s", _APPS_DIR)
        return 0

    records: list[tuple[str, str, str]] = []  # (app_id, solution_id, embed_text)

    for app_id in sorted(os.listdir(_APPS_DIR)):
        sol_dir = os.path.join(_APPS_DIR, app_id, "solutions")
        if not os.path.isdir(sol_dir):
            continue
        for fname in sorted(os.listdir(sol_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(sol_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    sol = json.load(f)
            except Exception:
                continue

            sol_id = sol.get("solution_id", fname.removesuffix(".json"))
            name = sol.get("name", sol_id)
            desc = sol.get("description", "")
            patterns = ", ".join(sol.get("intent_patterns", []))
            signals = ", ".join(sol.get("column_signals", []))

            embed_text = f"{name}. {desc}"
            if patterns:
                embed_text += f". Patterns: {patterns}"
            if signals:
                embed_text += f". Signals: {signals}"

            records.append((app_id, sol_id, embed_text))

    if not records:
        return 0

    texts = [r[2] for r in records]
    embeddings = embed_texts(texts)

    with conn.cursor() as cur:
        for (app_id, sol_id, embed_text), emb in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO solution_embeddings (app_id, solution_id, embed_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (app_id, solution_id)
                DO UPDATE SET embed_text = EXCLUDED.embed_text,
                              embedding = EXCLUDED.embedding,
                              created_at = NOW()
                """,
                (app_id, sol_id, embed_text, emb.tolist()),
            )
    conn.commit()
    log.info("Seeded %d solution embeddings", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# Domain keyword embeddings (domains, solutions from registry, lifecycle)
# ---------------------------------------------------------------------------

def seed_domain_embeddings(conn) -> int:
    """Embed domain keywords, solution registry entries, and lifecycle stages. Returns count."""
    from hooks.embedding_registry import embed_texts

    records: list[tuple[str, str, str]] = []  # (domain_name, category, embed_text)

    # 1. Domain keywords + profiles
    from hooks.domain_detector import _DOMAIN_KEYWORDS, _DOMAIN_PROFILES
    for domain, keywords in _DOMAIN_KEYWORDS:
        profile = _DOMAIN_PROFILES.get(domain, {})
        kw_str = ", ".join(keywords[:30])
        lens = profile.get("analysis_lens", "")
        embed_text = f"Domain: {domain}. Keywords: {kw_str}"
        if lens:
            embed_text += f". {lens}"
        records.append((domain, "domain", embed_text))

    # 2. Solution registry entries
    if os.path.exists(_SOLUTIONS_PATH):
        try:
            with open(_SOLUTIONS_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

        for domain, solutions in registry.items():
            if domain.startswith("_") or not isinstance(solutions, list):
                continue
            for sol in solutions:
                if not isinstance(sol, dict):
                    continue
                name = sol.get("name", "")
                desc = sol.get("description", "")
                triggers = ", ".join(sol.get("triggers", []))
                embed_text = f"Solution: {name}. {desc}"
                if triggers:
                    embed_text += f". Triggers: {triggers}"
                key = f"{domain}::{name}"
                records.append((key, "solution", embed_text))

    # 3. Lifecycle stages
    from hooks.domain_detector import _LIFECYCLE_STAGE_KEYWORDS
    for stage, keywords in _LIFECYCLE_STAGE_KEYWORDS.items():
        kw_str = ", ".join(keywords)
        embed_text = f"Lifecycle: {stage}. Keywords: {kw_str}"
        records.append((stage, "lifecycle", embed_text))

    if not records:
        return 0

    texts = [r[2] for r in records]
    embeddings = embed_texts(texts)

    with conn.cursor() as cur:
        for (domain_name, category, embed_text), emb in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO domain_keyword_embeddings (domain_name, category, embed_text, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (domain_name, category)
                DO UPDATE SET embed_text = EXCLUDED.embed_text,
                              embedding = EXCLUDED.embedding,
                              created_at = NOW()
                """,
                (domain_name, category, embed_text, emb.tolist()),
            )
    conn.commit()
    log.info("Seeded %d domain keyword embeddings", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# Domain alias embeddings (for _resolve_domain)
# ---------------------------------------------------------------------------

def _seed_domain_alias_embeddings(conn) -> int:
    """Embed domain alias texts and all canonical domain names. Returns count."""
    from hooks.embedding_registry import embed_texts
    from hooks.domain_graph import _DOMAIN_ALIASES

    records: list[tuple[str, str]] = []  # (alias_text, canonical)

    # Alias → canonical mappings
    for alias, canonical in _DOMAIN_ALIASES.items():
        records.append((alias, canonical))

    # Also embed all canonical names pointing to themselves
    canonical_names = set(_DOMAIN_ALIASES.values())
    for name in sorted(canonical_names):
        records.append((name, name))

    if not records:
        return 0

    texts = [r[0] for r in records]
    embeddings = embed_texts(texts)

    with conn.cursor() as cur:
        for (alias_text, canonical), emb in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO domain_alias_embeddings (alias_text, canonical, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (alias_text)
                DO UPDATE SET canonical = EXCLUDED.canonical,
                              embedding = EXCLUDED.embedding
                """,
                (alias_text, canonical, emb.tolist()),
            )
    conn.commit()
    log.info("Seeded %d domain alias embeddings", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# Plan pattern embeddings (for _pattern_match)
# ---------------------------------------------------------------------------

def seed_plan_pattern_embeddings(conn) -> int:
    """Embed all intent_patterns from execution_plans. Returns count."""
    from hooks.embedding_registry import embed_texts

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, intent_patterns FROM execution_plans WHERE intent_patterns IS NOT NULL"
        )
        rows = cur.fetchall()

    records: list[tuple[int, str]] = []  # (plan_id, pattern_text)
    for plan_id, patterns in rows:
        if not patterns:
            continue
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern:
                records.append((plan_id, pattern))

    if not records:
        return 0

    texts = [r[1] for r in records]
    embeddings = embed_texts(texts)

    with conn.cursor() as cur:
        for (plan_id, pattern_text), emb in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO plan_pattern_embeddings (plan_id, pattern_text, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (plan_id, pattern_text)
                DO UPDATE SET embedding = EXCLUDED.embedding
                """,
                (plan_id, pattern_text, emb.tolist()),
            )
    conn.commit()
    log.info("Seeded %d plan pattern embeddings", len(records))
    return len(records)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def seed_all_embeddings(conn=None) -> dict[str, int]:
    """Seed all static embedding tables. Returns counts per category."""
    close_conn = False
    if conn is None:
        conn = _get_conn()
        close_conn = True

    # Ensure tables exist
    from hooks.db_migrate import run_migration
    run_migration(conn)

    results = {}
    try:
        results["solutions"] = seed_solution_embeddings(conn)
        results["domains"] = seed_domain_embeddings(conn)
        results["aliases"] = _seed_domain_alias_embeddings(conn)
        results["plan_patterns"] = seed_plan_pattern_embeddings(conn)
    finally:
        if close_conn:
            conn.close()

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    counts = seed_all_embeddings()
    print(f"Seeded embeddings: {counts}")
