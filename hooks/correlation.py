"""Correlation hook — pure pandas/scipy, no LLM."""

import pandas as pd


def correlation_report(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Return correlation matrix for all numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for correlation.")
    return numeric.corr(method=method).round(4)


def top_correlations(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Return pairs of columns with |correlation| >= threshold (excluding self-pairs)."""
    import numpy as np
    corr = correlation_report(df).abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = (
        corr.where(mask)
        .stack()
        .reset_index()
    )
    pairs.columns = ["col_a", "col_b", "correlation"]
    return pairs[pairs["correlation"] >= threshold].sort_values(
        "correlation", ascending=False
    ).reset_index(drop=True)
