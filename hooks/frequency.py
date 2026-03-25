"""Frequency / top-N hook — pure pandas, no LLM."""

import pandas as pd


def top_n(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    """Return value frequency table for col, top n rows."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    counts = df[col].value_counts().head(n).reset_index()
    counts.columns = [col, "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(2)
    return counts


def full_frequency(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return the full frequency table for col."""
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(2)
    return counts
