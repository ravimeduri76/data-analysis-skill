"""Sampling comparison hook — scores multiple sample DataFrames on diversity preservation.

All logic is pure pandas / numpy — no LLM inference.

Typical use case: you have a full dataset and 2–4 samples produced by different
techniques (random, PAM, CLARA, stratified, …) and want to know which sample
best represents the original data for profiling purposes.

Public API
----------
compare_samples(dfs)            -> pd.DataFrame   scorecard (one row per sample)
cardinality_report(dfs)         -> pd.DataFrame   per-column cardinality retention %
numeric_range_report(dfs)       -> pd.DataFrame   per-column numeric range coverage %
null_completeness_report(dfs)   -> pd.DataFrame   per-column null% per sample
sampling_commentary(dfs)        -> str            plain-English ranking + suggestions
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _common_cat_cols(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Categorical columns present in all samples."""
    sets = [set(df.select_dtypes(exclude="number").columns) for df in dfs.values()]
    common = sets[0].intersection(*sets[1:])
    # maintain original column order from the first sample
    first_cols = list(dfs[next(iter(dfs))].select_dtypes(exclude="number").columns)
    return [c for c in first_cols if c in common]


def _common_num_cols(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Numeric columns present in all samples."""
    sets = [set(df.select_dtypes(include="number").columns) for df in dfs.values()]
    common = sets[0].intersection(*sets[1:])
    first_cols = list(dfs[next(iter(dfs))].select_dtypes(include="number").columns)
    return [c for c in first_cols if c in common]


# ── public API ────────────────────────────────────────────────────────────────

def cardinality_report(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-column cardinality retention: sample_unique / combined_unique (%).

    A value of 100% means the sample has as many distinct values as all samples
    combined — i.e., full coverage of the known value space.
    """
    names = list(dfs.keys())
    cat_cols = _common_cat_cols(dfs)
    rows = []
    for col in cat_cols:
        combined = set().union(*[set(dfs[n][col].dropna()) for n in names])
        total = len(combined)
        row = {"column": col, "combined_unique": total}
        for name in names:
            sample_unique = dfs[name][col].nunique()
            row[f"{name}_unique"] = sample_unique
            row[f"{name}_retention_%"] = round(sample_unique / total * 100, 1) if total > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_range_report(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-column numeric range coverage: (sample_range) / (combined_range) (%).

    100% means the sample spans the full min–max extent seen across all samples.
    """
    names = list(dfs.keys())
    num_cols = _common_num_cols(dfs)
    rows = []
    for col in num_cols:
        all_vals = pd.concat([dfs[n][col].dropna() for n in names])
        if all_vals.empty:
            continue
        combined_range = all_vals.max() - all_vals.min()
        if combined_range == 0:
            continue
        row = {"column": col,
               "combined_min": round(float(all_vals.min()), 4),
               "combined_max": round(float(all_vals.max()), 4)}
        for name in names:
            s = dfs[name][col].dropna()
            if s.empty:
                row[f"{name}_coverage_%"] = 0.0
            else:
                cov = (s.max() - s.min()) / combined_range * 100
                row[f"{name}_coverage_%"] = round(float(cov), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def null_completeness_report(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-column null% for each sample, plus deviation from the first sample.

    The first key in dfs is treated as the reference (typically 'random').
    """
    names = list(dfs.keys())
    ref   = names[0]
    # common columns across all samples
    common_cols = list(
        set(dfs[ref].columns).intersection(*[set(dfs[n].columns) for n in names[1:]])
    )
    rows = []
    for col in common_cols:
        row = {"column": col}
        for name in names:
            row[f"{name}_null%"] = round(dfs[name][col].isna().mean() * 100, 1)
        # deviation from reference
        for name in names[1:]:
            row[f"{name}_vs_{ref}_diff_pp"] = round(
                abs(row[f"{name}_null%"] - row[f"{ref}_null%"]), 1
            )
        rows.append(row)
    df_out = pd.DataFrame(rows)
    # sort by highest null% in reference to surface worst columns first
    df_out = df_out.sort_values(f"{ref}_null%", ascending=False).reset_index(drop=True)
    return df_out


def compare_samples(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a diversity scorecard with one row per sample.

    Metrics:
      - rows                      : sample size
      - cols_with_data_%          : % columns that have at least 1 non-null value
      - avg_cat_cardinality_%     : avg categorical cardinality retention
      - median_cat_cardinality_%  : median categorical cardinality retention
      - cat_cols_full_coverage_%  : % cat cols at 100% cardinality retention
      - avg_num_range_coverage_%  : avg numeric range coverage
      - num_cols_full_coverage_%  : % numeric cols at 100% range coverage
      - wholly_null_cols          : count of columns that are entirely null
      - avg_null_pct              : average null% across all columns
      - composite_score           : weighted composite (0–100)
    """
    names = list(dfs.keys())
    card_df  = cardinality_report(dfs)
    range_df = numeric_range_report(dfs)

    rows = []
    for name in names:
        df = dfs[name]
        n_cols = df.shape[1]

        cols_with_data = int((df.notna().any()).sum())
        cols_with_data_pct = round(cols_with_data / n_cols * 100, 1) if n_cols > 0 else 0.0

        # cardinality metrics
        ret_col = f"{name}_retention_%"
        if ret_col in card_df.columns and not card_df[ret_col].empty:
            avg_card   = round(card_df[ret_col].mean(), 1)
            med_card   = round(card_df[ret_col].median(), 1)
            full_card  = round((card_df[ret_col] == 100).mean() * 100, 1)
        else:
            avg_card = med_card = full_card = 0.0

        # numeric range metrics
        cov_col = f"{name}_coverage_%"
        if cov_col in range_df.columns and not range_df[cov_col].empty:
            avg_range  = round(range_df[cov_col].mean(), 1)
            full_range = round((range_df[cov_col] == 100).mean() * 100, 1)
        else:
            avg_range = full_range = 0.0

        wholly_null = int((df.isna().all()).sum())
        avg_null    = round(df.isna().mean().mean() * 100, 1)

        # composite score: weighted average of key diversity metrics
        composite = round(
            avg_card   * 0.35 +
            avg_range  * 0.30 +
            cols_with_data_pct * 0.25 +
            (100 - avg_null) * 0.10,
            1
        )

        rows.append({
            "sample":                    name,
            "rows":                      df.shape[0],
            "avg_null_pct":              avg_null,
            "wholly_null_cols":          wholly_null,
            "cols_with_data_%":          cols_with_data_pct,
            "avg_cat_cardinality_%":     avg_card,
            "median_cat_cardinality_%":  med_card,
            "cat_cols_full_coverage_%":  full_card,
            "avg_num_range_coverage_%":  avg_range,
            "num_cols_full_coverage_%":  full_range,
            "composite_score":           composite,
        })

    return pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)


def sampling_commentary(dfs: dict[str, pd.DataFrame]) -> str:
    """Return a plain-English ranking and recommendation string.

    Covers: winner, per-metric breakdown, why cluster-based methods
    collapse diversity, and actionable next steps.
    """
    names  = list(dfs.keys())
    card   = compare_samples(dfs)
    null_r = null_completeness_report(dfs)
    ref    = names[0]

    lines = []
    lines.append("SAMPLING TECHNIQUE DIVERSITY REPORT")
    lines.append("=" * 60)

    # Ranking
    lines.append("\nRANKING (by composite diversity score):")
    for rank, row in card.iterrows():
        lines.append(
            f"  #{rank+1}  {row['sample'].upper():12s}  "
            f"composite={row['composite_score']:.1f}  "
            f"cat_cardinality={row['avg_cat_cardinality_%']:.1f}%  "
            f"num_range={row['avg_num_range_coverage_%']:.1f}%  "
            f"col_completeness={row['cols_with_data_%']:.1f}%"
        )

    winner = card.iloc[0]["sample"]
    lines.append(f"\nRECOMMENDED FOR PROFILING: {winner.upper()}")

    # Per-metric narrative
    lines.append("\nMETRIC BREAKDOWN:")

    lines.append("\n  Categorical cardinality retention (higher = more diverse):")
    for _, row in card.iterrows():
        bar = "#" * int(row["avg_cat_cardinality_%"] / 5)
        lines.append(
            f"    {row['sample']:12s}  {row['avg_cat_cardinality_%']:5.1f}%  "
            f"(median {row['median_cat_cardinality_%']:.1f}%,  "
            f"full coverage on {row['cat_cols_full_coverage_%']:.1f}% of cols)  {bar}"
        )

    lines.append("\n  Numeric range coverage (higher = wider spread preserved):")
    for _, row in card.iterrows():
        bar = "#" * int(row["avg_num_range_coverage_%"] / 5)
        lines.append(
            f"    {row['sample']:12s}  {row['avg_num_range_coverage_%']:5.1f}%  "
            f"(100% coverage on {row['num_cols_full_coverage_%']:.1f}% of cols)  {bar}"
        )

    lines.append("\n  Column completeness (% cols with at least 1 non-null value):")
    for _, row in card.iterrows():
        lines.append(
            f"    {row['sample']:12s}  {row['cols_with_data_%']:5.1f}%  "
            f"({row['wholly_null_cols']} cols entirely null)"
        )

    lines.append("\n  Average null% across all columns:")
    for _, row in card.iterrows():
        lines.append(f"    {row['sample']:12s}  {row['avg_null_pct']:5.1f}%")

    # Null deviation from reference
    if len(names) > 1:
        lines.append(f"\n  Null% deviation from '{ref}' (lower = more similar null pattern):")
        for other in names[1:]:
            diff_col = f"{other}_vs_{ref}_diff_pp"
            if diff_col in null_r.columns:
                avg_diff = round(null_r[diff_col].mean(), 1)
                wholly_lost = ((null_r[f"{other}_null%"] == 100) &
                               (null_r[f"{ref}_null%"] < 100)).sum()
                lines.append(
                    f"    {other:12s}  avg diff = {avg_diff} pp  |  "
                    f"{wholly_lost} cols 100% null in {other} but not in {ref}"
                )

    # Generic guidance based on results
    lines.append("\nINSIGHTS & RECOMMENDATIONS:")

    loser_scores = card[card["composite_score"] < card["composite_score"].max()]
    cluster_names = loser_scores["sample"].tolist()

    if cluster_names:
        lines.append(
            f"\n  Why {' and '.join(c.upper() for c in cluster_names)} score lower:"
        )
        lines.append(
            "  Cluster-based methods (PAM, CLARA, k-medoids variants) select"
        )
        lines.append(
            "  medoid points — the most central rows within each cluster. By design"
        )
        lines.append(
            "  they optimise for TYPICALITY, not breadth. Rare categories, sparse"
        )
        lines.append(
            "  attribute columns, and extreme-value rows are systematically excluded"
        )
        lines.append(
            "  because they lie at cluster peripheries. This collapses cardinality"
        )
        lines.append(
            "  and range coverage — exactly what profiling needs."
        )

    lines.append("\n  When to use each technique:")
    lines.append(
        "  - RANDOM       : best for profiling, EDA, and preserving full data diversity."
    )
    lines.append(
        "  - PAM / CLARA  : best for seeding clustering, training 'typical-case' models,"
    )
    lines.append(
        "                   or representative display samples where outliers are noise."
    )
    lines.append(
        "  - STRATIFIED   : best when you need both diversity AND guaranteed coverage"
    )
    lines.append(
        "                   of minority groups (stratify by key dimension columns)."
    )

    lines.append("\n  Next steps:")
    lines.append(f"  1. Use the {winner.upper()} sample for all profiling and EDA work.")
    lines.append(
        "  2. If any key dimension is under-represented even in random, apply"
    )
    lines.append(
        "     stratified sampling on that column to guarantee coverage."
    )
    lines.append(
        "  3. Columns with >80% nulls in ALL samples are sparse attributes —"
    )
    lines.append(
        "     filter them out before profiling to surface meaningful signals."
    )
    lines.append(
        "  4. Run data_quality.quality_commentary() on the winning sample"
    )
    lines.append(
        "     for a full per-column quality assessment."
    )

    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
