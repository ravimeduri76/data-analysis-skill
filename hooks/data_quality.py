"""Data quality hook — per-column scoring, issue detection, and commentary.

All logic is pure pandas / scipy — no LLM inference.

Public API
----------
quality_report(df)      -> pd.DataFrame   per-column quality table
quality_commentary(df)  -> str            plain-English findings ready for display
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import NamedTuple


# ── thresholds ────────────────────────────────────────────────────────────────
_NULL_WARN   = 20.0   # % nulls → warn
_NULL_CRIT   = 80.0   # % nulls → critical
_SKEW_WARN   = 2.0    # |skew| above this → flag
_IQR_MULT    = 3.0    # IQR multiplier for outlier detection
_DUP_WARN    = 1.0    # % duplicate rows → warn
_CONST_RATIO = 0.995  # single value covers this share → constant column


class _ColResult(NamedTuple):
    column: str
    dtype: str
    null_pct: float
    unique: int
    unique_pct: float
    is_constant: bool
    is_all_null: bool
    skewness: float | None
    outlier_pct: float | None
    issues: str          # pipe-separated issue tags
    quality_score: int   # 0 (worst) – 100 (best)


# ── internal helpers ──────────────────────────────────────────────────────────

def _outlier_pct(s: pd.Series) -> float | None:
    """IQR-based outlier percentage for numeric series."""
    clean = s.dropna()
    if len(clean) < 4:
        return None
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return None
    mask = (clean < q1 - _IQR_MULT * iqr) | (clean > q3 + _IQR_MULT * iqr)
    return round(mask.mean() * 100, 2)


def _skewness(s: pd.Series) -> float | None:
    clean = s.dropna()
    if len(clean) < 4:
        return None
    try:
        return round(float(clean.skew()), 3)
    except Exception:
        return None


def _score(null_pct: float, is_constant: bool, is_all_null: bool,
           skewness: float | None, outlier_pct: float | None) -> int:
    """Score a column 0–100. Deductions are additive."""
    if is_all_null:
        return 0
    score = 100
    # null penalties
    if null_pct >= _NULL_CRIT:
        score -= 50
    elif null_pct >= _NULL_WARN:
        score -= int((null_pct - _NULL_WARN) / (_NULL_CRIT - _NULL_WARN) * 30)
    # constant column
    if is_constant:
        score -= 20
    # high skew
    if skewness is not None and abs(skewness) >= _SKEW_WARN:
        score -= min(15, int((abs(skewness) - _SKEW_WARN) * 3))
    # outliers
    if outlier_pct is not None and outlier_pct > 5:
        score -= min(15, int(outlier_pct))
    return max(0, score)


def _issues(null_pct: float, is_constant: bool, is_all_null: bool,
            skewness: float | None, outlier_pct: float | None,
            is_numeric: bool, unique_pct: float) -> str:
    tags = []
    if is_all_null:
        tags.append("ALL_NULL")
    elif null_pct >= _NULL_CRIT:
        tags.append("CRITICAL_NULLS")
    elif null_pct >= _NULL_WARN:
        tags.append("HIGH_NULLS")
    if is_constant:
        tags.append("CONSTANT")
    if skewness is not None and abs(skewness) >= _SKEW_WARN:
        tags.append("HIGH_SKEW")
    if outlier_pct is not None and outlier_pct > 5:
        tags.append("OUTLIERS")
    if is_numeric and unique_pct < 0.5 and not is_constant:
        tags.append("LOW_CARDINALITY_NUMERIC")
    return " | ".join(tags) if tags else "OK"


# ── public API ────────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column quality DataFrame.

    Columns: column, dtype, null_pct, unique, unique_pct,
             is_constant, is_all_null, skewness, outlier_pct,
             issues, quality_score
    """
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        null_pct   = round(s.isna().mean() * 100, 2)
        unique     = int(s.nunique())
        unique_pct = round(unique / n * 100, 2) if n > 0 else 0.0
        is_all_null   = null_pct == 100.0
        top_freq      = s.value_counts(normalize=True).iloc[0] if unique > 0 else 1.0
        is_constant   = top_freq >= _CONST_RATIO

        skew_val    = _skewness(s) if is_numeric else None
        out_pct     = _outlier_pct(s) if is_numeric else None
        issue_str   = _issues(null_pct, is_constant, is_all_null,
                               skew_val, out_pct, is_numeric, unique_pct)
        score       = _score(null_pct, is_constant, is_all_null, skew_val, out_pct)

        rows.append({
            "column":       col,
            "dtype":        str(s.dtype),
            "null_pct":     null_pct,
            "unique":       unique,
            "unique_pct":   unique_pct,
            "is_constant":  is_constant,
            "is_all_null":  is_all_null,
            "skewness":     skew_val,
            "outlier_pct":  out_pct,
            "issues":       issue_str,
            "quality_score": score,
        })

    return pd.DataFrame(rows).sort_values("quality_score").reset_index(drop=True)


def quality_commentary(df: pd.DataFrame) -> str:
    """Return a plain-English data quality commentary string.

    Covers: overall score, null issues, constant columns, skew, outliers,
    duplicate rows, and actionable suggestions.
    """
    n_rows, n_cols = df.shape
    report = quality_report(df)

    overall = round(report["quality_score"].mean(), 1)
    grade   = ("A" if overall >= 85 else
               "B" if overall >= 70 else
               "C" if overall >= 55 else
               "D" if overall >= 40 else "F")

    lines = []
    lines.append(f"DATA QUALITY COMMENTARY")
    lines.append(f"{'='*60}")
    lines.append(f"Dataset: {n_rows:,} rows x {n_cols} columns")
    lines.append(f"Overall quality score: {overall}/100  (Grade: {grade})")
    lines.append("")

    # Duplicate rows
    dup_pct = round(df.duplicated().mean() * 100, 2)
    if dup_pct >= _DUP_WARN:
        lines.append(f"[WARNING] Duplicate rows: {dup_pct}% of rows are exact duplicates.")
    else:
        lines.append(f"[OK] Duplicate rows: {dup_pct}% (within acceptable range).")

    # All-null columns
    all_null_cols = report[report["is_all_null"]]["column"].tolist()
    if all_null_cols:
        lines.append(f"\n[CRITICAL] {len(all_null_cols)} column(s) are entirely null (100% missing):")
        for c in all_null_cols[:10]:
            lines.append(f"   - {c}")
        if len(all_null_cols) > 10:
            lines.append(f"   ... and {len(all_null_cols)-10} more.")
        lines.append("  -> Recommendation: Drop these columns before analysis.")

    # Critical nulls (80–99%)
    crit_null = report[(report["null_pct"] >= _NULL_CRIT) & (~report["is_all_null"])]
    if not crit_null.empty:
        lines.append(f"\n[CRITICAL] {len(crit_null)} column(s) with >=80% nulls:")
        for _, row in crit_null.head(10).iterrows():
            lines.append(f"   - {row['column']}: {row['null_pct']}% null")
        lines.append("  -> Recommendation: Treat as sparse attributes. Impute only if domain-justified.")

    # High nulls (20–79%)
    high_null = report[(report["null_pct"] >= _NULL_WARN) & (report["null_pct"] < _NULL_CRIT)]
    if not high_null.empty:
        lines.append(f"\n[WARNING] {len(high_null)} column(s) with 20–79% nulls:")
        for _, row in high_null.head(10).iterrows():
            lines.append(f"   - {row['column']}: {row['null_pct']}% null")
        if len(high_null) > 10:
            lines.append(f"   ... and {len(high_null)-10} more.")
        lines.append("  -> Recommendation: Investigate null pattern (MCAR/MAR/MNAR) before imputing.")

    # Constant columns
    const_cols = report[report["is_constant"] & ~report["is_all_null"]]["column"].tolist()
    if const_cols:
        lines.append(f"\n[WARNING] {len(const_cols)} constant column(s) (single dominant value >=99.5%):")
        for c in const_cols[:10]:
            top_val = df[c].mode().iloc[0] if df[c].notna().any() else "N/A"
            lines.append(f"   - {c}  (top value: '{top_val}')")
        lines.append("  -> Recommendation: Drop from models; offer zero predictive value.")

    # High skew numeric cols
    skew_cols = report[report["skewness"].notna() & (report["skewness"].abs() >= _SKEW_WARN)]
    if not skew_cols.empty:
        lines.append(f"\n[INFO] {len(skew_cols)} numeric column(s) with high skew (|skew| >= {_SKEW_WARN}):")
        for _, row in skew_cols.head(8).iterrows():
            lines.append(f"   - {row['column']}: skew = {row['skewness']}")
        lines.append("  -> Recommendation: Apply log/sqrt transform before modelling or visualising.")

    # Outlier-heavy cols
    out_cols = report[report["outlier_pct"].notna() & (report["outlier_pct"] > 5)]
    if not out_cols.empty:
        lines.append(f"\n[INFO] {len(out_cols)} numeric column(s) with >5% IQR outliers:")
        for _, row in out_cols.head(8).iterrows():
            lines.append(f"   - {row['column']}: {row['outlier_pct']}% outliers")
        lines.append("  -> Recommendation: Verify outliers are real (not data errors); cap or winsorise if needed.")

    # Low-cardinality numeric (likely encoded categoricals)
    low_card_num = report[report["issues"].str.contains("LOW_CARDINALITY_NUMERIC", na=False)]
    if not low_card_num.empty:
        lines.append(f"\n[INFO] {len(low_card_num)} numeric column(s) with very low cardinality (<0.5% unique):")
        for _, row in low_card_num.head(5).iterrows():
            lines.append(f"   - {row['column']}: {row['unique']} unique values")
        lines.append("  -> Recommendation: Consider treating as categorical (Dimension) rather than Fact.")

    # Summary
    ok_count = (report["issues"] == "OK").sum()
    lines.append(f"\nSUMMARY")
    lines.append(f"  Columns with no issues (OK): {ok_count} / {n_cols}")
    lines.append(f"  Columns with issues:         {n_cols - ok_count} / {n_cols}")

    # Score-band breakdown
    bands = {
        "Excellent (90-100)": ((report["quality_score"] >= 90)).sum(),
        "Good      (70-89)":  ((report["quality_score"] >= 70) & (report["quality_score"] < 90)).sum(),
        "Fair      (50-69)":  ((report["quality_score"] >= 50) & (report["quality_score"] < 70)).sum(),
        "Poor      (0-49)":   ((report["quality_score"] < 50)).sum(),
    }
    lines.append("\nColumn quality distribution:")
    for band, count in bands.items():
        bar = "#" * int(count / n_cols * 30)
        lines.append(f"  {band}: {count:>4} cols  {bar}")

    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
