"""
harmonizer.py -- AG09 SchemaHarmonizer reference implementation.

Generic harmonizer that takes a DataFrame + DCP (Domain Context Pack).
Domain knowledge comes from DCP, not hardcoded rules.

Public API
----------
harmonize(df, dcp, target_schema=None) -> (DataFrame, dict)
apply_entity_resolution(series, graph_entities) -> (Series, dict)
apply_controlled_vocabulary(series, terminology, property_values) -> (Series, dict)
check_invariants(df, invariants) -> list[dict]
completeness_report(df, property_model) -> dict
"""

from __future__ import annotations

import difflib
import re
from typing import Any

import pandas as pd


# ── Internal helpers ───────────────────────────────────────────────────────


def _build_alias_map(graph_entities: list[dict]) -> dict[str, str]:
    """Build case-insensitive alias → canonical name lookup from graph entities."""
    alias_map: dict[str, str] = {}
    for entity in graph_entities:
        canonical = entity.get("name", "")
        if not canonical:
            continue
        alias_map[canonical.lower()] = canonical
        for alt in entity.get("related_terms", []):
            alias_map[alt.lower()] = canonical
        desc = entity.get("description", "")
        if desc:
            alias_map[desc.lower()] = canonical
    return alias_map


def _build_vocab_map(
    terminology: list[dict],
    property_values: list[str] | None = None,
) -> dict[str, str]:
    """Build canonical term lookup from terminology + property model values."""
    vocab_map: dict[str, str] = {}
    for term_entry in terminology:
        canonical = term_entry.get("term", "")
        if not canonical:
            continue
        vocab_map[canonical.lower()] = canonical
        for alt in term_entry.get("related_terms", []):
            vocab_map[alt.lower()] = canonical
    if property_values:
        for val in property_values:
            vocab_map[val.lower()] = val
    return vocab_map


def _clean_text_value(value: str) -> str:
    """Collapse whitespace, strip artifacts, fix encoding, smart title case."""
    if not isinstance(value, str):
        return value
    # Fix common encoding artifacts
    value = value.replace("\u00e2\u0080\u0099", "'")
    value = value.replace("\u00e2\u0080\u0093", "-")
    value = value.replace("\u00c2\u00a0", " ")
    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()
    # Remove trailing punctuation artifacts (but keep meaningful ones)
    value = re.sub(r"[;,]+$", "", value).strip()
    # Smart title case: skip small words unless they're the first word
    if value and not any(c.isupper() for c in value[1:]):
        words = value.split()
        small_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        result = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        value = " ".join(result)
    return value


def _get_property_values(col_name: str, property_model: list[dict]) -> list[str] | None:
    """Find enumerated values for a column from the property model."""
    col_lower = col_name.lower()
    for group in property_model:
        for prop in group.get("properties", []):
            prop_name = prop.get("name", "").lower()
            if prop_name == col_lower or col_lower in prop_name or prop_name in col_lower:
                values = prop.get("enumerated_values", prop.get("values", []))
                if values:
                    return values
    return None


# ── Public API ─────────────────────────────────────────────────────────────


def apply_entity_resolution(
    series: pd.Series,
    graph_entities: list[dict],
) -> tuple[pd.Series, dict]:
    """Map aliases to canonical names using exact → containment → fuzzy match.

    Returns (resolved_series, report_dict).
    """
    if not graph_entities:
        return series, {"resolved": 0, "unresolved": 0, "method": "none"}

    alias_map = _build_alias_map(graph_entities)
    canonical_names = list(set(alias_map.values()))

    resolved_count = 0
    unresolved_count = 0
    result = series.copy()

    for idx, val in series.items():
        if pd.isna(val) or not isinstance(val, str) or not val.strip():
            continue

        val_lower = val.strip().lower()

        # Exact match
        if val_lower in alias_map:
            result.at[idx] = alias_map[val_lower]
            resolved_count += 1
            continue

        # Containment match: check if any alias is contained in the value
        matched = False
        for alias, canonical in alias_map.items():
            if alias in val_lower or val_lower in alias:
                result.at[idx] = canonical
                resolved_count += 1
                matched = True
                break

        if matched:
            continue

        # Fuzzy match via difflib (cutoff=0.8)
        close = difflib.get_close_matches(val_lower, alias_map.keys(), n=1, cutoff=0.8)
        if close:
            result.at[idx] = alias_map[close[0]]
            resolved_count += 1
        else:
            unresolved_count += 1

    return result, {
        "resolved": resolved_count,
        "unresolved": unresolved_count,
        "total": resolved_count + unresolved_count,
        "method": "exact+containment+fuzzy",
    }


def apply_controlled_vocabulary(
    series: pd.Series,
    terminology: list[dict],
    property_values: list[str] | None = None,
) -> tuple[pd.Series, dict]:
    """Normalize values using DCP terminology + property model enumerated lists.

    Returns (normalized_series, report_dict).
    """
    if not terminology and not property_values:
        return series, {"normalized": 0, "unmatched": 0, "method": "none"}

    vocab_map = _build_vocab_map(terminology, property_values)

    normalized_count = 0
    unmatched_count = 0
    result = series.copy()

    for idx, val in series.items():
        if pd.isna(val) or not isinstance(val, str) or not val.strip():
            continue

        val_lower = val.strip().lower()

        if val_lower in vocab_map:
            result.at[idx] = vocab_map[val_lower]
            normalized_count += 1
        else:
            # Try cleaning then matching
            cleaned = _clean_text_value(val).lower()
            if cleaned in vocab_map:
                result.at[idx] = vocab_map[cleaned]
                normalized_count += 1
            else:
                unmatched_count += 1

    return result, {
        "normalized": normalized_count,
        "unmatched": unmatched_count,
        "total": normalized_count + unmatched_count,
        "method": "vocabulary_lookup",
    }


def check_invariants(df: pd.DataFrame, invariants: list[dict]) -> list[dict]:
    """Parse invariant rules, check against df, return violations.

    Each invariant dict should have at least: {rule: str, description: str}.
    Supported rule patterns:
    - "column NOT NULL" → check for nulls
    - "column IN (val1, val2, ...)" → check membership
    - "column UNIQUE" → check uniqueness
    """
    violations: list[dict] = []

    for inv in invariants:
        rule = inv.get("rule", inv.get("description", ""))
        if not rule:
            continue

        rule_upper = rule.upper().strip()

        # Pattern: <col> NOT NULL
        not_null_match = re.match(r"(\w+)\s+NOT\s+NULL", rule_upper)
        if not_null_match:
            col = not_null_match.group(1)
            # Case-insensitive column match
            actual_col = _find_column(df, col)
            if actual_col and df[actual_col].isnull().any():
                null_count = int(df[actual_col].isnull().sum())
                violations.append({
                    "invariant": rule,
                    "column": actual_col,
                    "violation": f"{null_count} null values found",
                    "severity": "error",
                })
            continue

        # Pattern: <col> IN (val1, val2, ...)
        in_match = re.match(r"(\w+)\s+IN\s*\((.+)\)", rule_upper)
        if in_match:
            col = in_match.group(1)
            vals_str = in_match.group(2)
            allowed = {v.strip().strip("'\"") for v in vals_str.split(",")}
            actual_col = _find_column(df, col)
            if actual_col:
                actual_vals = set(df[actual_col].dropna().astype(str).str.upper())
                bad_vals = actual_vals - allowed
                if bad_vals:
                    violations.append({
                        "invariant": rule,
                        "column": actual_col,
                        "violation": f"Values outside allowed set: {sorted(bad_vals)[:5]}",
                        "severity": "warning",
                    })
            continue

        # Pattern: <col> UNIQUE
        unique_match = re.match(r"(\w+)\s+UNIQUE", rule_upper)
        if unique_match:
            col = unique_match.group(1)
            actual_col = _find_column(df, col)
            if actual_col:
                dupes = int(df[actual_col].duplicated().sum())
                if dupes > 0:
                    violations.append({
                        "invariant": rule,
                        "column": actual_col,
                        "violation": f"{dupes} duplicate values found",
                        "severity": "error",
                    })
            continue

    return violations


def _find_column(df: pd.DataFrame, col_name: str) -> str | None:
    """Case-insensitive column lookup."""
    col_lower = col_name.lower()
    for c in df.columns:
        if c.lower() == col_lower:
            return c
    return None


def completeness_report(df: pd.DataFrame, property_model: list[dict]) -> dict:
    """Fill rates per property group from DCP property model.

    Returns {group_name: {property: fill_rate, ...}, ...}.
    """
    report: dict[str, dict[str, float]] = {}
    total_rows = len(df)
    if total_rows == 0:
        return report

    for group in property_model:
        group_name = group.get("entity", group.get("name", "unknown"))
        group_report: dict[str, float] = {}
        for prop in group.get("properties", []):
            prop_name = prop.get("name", "")
            actual_col = _find_column(df, prop_name)
            if actual_col:
                fill_rate = round(1.0 - df[actual_col].isnull().sum() / total_rows, 4)
            else:
                fill_rate = 0.0
            group_report[prop_name] = fill_rate
        if group_report:
            report[group_name] = group_report

    return report


def harmonize(
    df: pd.DataFrame,
    dcp: dict,
    target_schema: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Full harmonization pipeline: entity resolution → controlled vocab →
    grammar cleanup → invariant check → completeness.

    Returns (harmonized_df, report_dict).
    """
    result_df = df.copy()
    report: dict[str, Any] = {
        "entity_resolution": {},
        "vocabulary": {},
        "grammar_cleaned": 0,
        "invariant_violations": [],
        "completeness": {},
    }

    graph_entities = dcp.get("graph_entities", [])
    terminology = dcp.get("terminology", [])
    property_model = dcp.get("property_model", [])
    invariants = dcp.get("invariants", [])

    # 1. Entity resolution on string columns
    if graph_entities:
        for col in result_df.select_dtypes(include=["object"]).columns:
            resolved, er_report = apply_entity_resolution(result_df[col], graph_entities)
            if er_report["resolved"] > 0:
                result_df[col] = resolved
                report["entity_resolution"][col] = er_report

    # 2. Controlled vocabulary on string columns
    if terminology:
        for col in result_df.select_dtypes(include=["object"]).columns:
            prop_values = _get_property_values(col, property_model) if property_model else None
            normalized, cv_report = apply_controlled_vocabulary(
                result_df[col], terminology, prop_values
            )
            if cv_report["normalized"] > 0:
                result_df[col] = normalized
                report["vocabulary"][col] = cv_report

    # 3. Grammar cleanup on remaining string columns
    cleaned_count = 0
    for col in result_df.select_dtypes(include=["object"]).columns:
        for idx, val in result_df[col].items():
            if pd.isna(val) or not isinstance(val, str):
                continue
            cleaned = _clean_text_value(val)
            if cleaned != val:
                result_df.at[idx, col] = cleaned
                cleaned_count += 1
    report["grammar_cleaned"] = cleaned_count

    # 4. Invariant check
    if invariants:
        report["invariant_violations"] = check_invariants(result_df, invariants)

    # 5. Completeness report
    if property_model:
        report["completeness"] = completeness_report(result_df, property_model)

    return result_df, report
