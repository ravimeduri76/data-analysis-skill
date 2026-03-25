"""Outlier detection hook — IQR and z-score methods, pure pandas/scipy, no LLM."""

import pandas as pd
import numpy as np


def outlier_report(
    df: pd.DataFrame,
    method: str = "iqr",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Return a summary of outlier counts per numeric column.

    method: 'iqr'    — values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
            'zscore' — values with |z| > z_threshold
    """
    numeric = df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        else:
            z = (s - s.mean()) / s.std()
            mask = z.abs() > z_threshold

        n_out = int(mask.sum())
        rows.append({
            "column":        col,
            "outlier_count": n_out,
            "outlier_pct":   round(n_out / len(s) * 100, 2) if len(s) else 0,
            "method":        method,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("outlier_count", ascending=False)
        .reset_index(drop=True)
    )


def get_outlier_rows(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Return the actual rows that are outliers for a specific column."""
    s = df[col].dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    else:
        z = (df[col] - df[col].mean()) / df[col].std()
        mask = z.abs() > z_threshold
    return df[mask].reset_index(drop=True)
