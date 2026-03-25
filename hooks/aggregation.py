"""Aggregation hook — group-by operations, pure pandas, no LLM."""

import pandas as pd
from typing import Union


_SUPPORTED_FUNCS = ("mean", "sum", "count", "min", "max", "median", "std", "nunique")


def group_agg(
    df: pd.DataFrame,
    group_cols: list[str],
    agg_col: str,
    func: str = "mean",
) -> pd.DataFrame:
    """Group df by group_cols and aggregate agg_col using func.

    Returns a sorted DataFrame (descending by the aggregated value).
    """
    if func not in _SUPPORTED_FUNCS:
        raise ValueError(f"func must be one of {_SUPPORTED_FUNCS}")
    missing = [c for c in group_cols + [agg_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    result = (
        df.groupby(group_cols)[agg_col]
        .agg(func)
        .reset_index()
        .rename(columns={agg_col: f"{func}_{agg_col}"})
        .sort_values(f"{func}_{agg_col}", ascending=False)
        .reset_index(drop=True)
    )
    return result


def multi_agg(
    df: pd.DataFrame,
    group_cols: list[str],
    agg_spec: dict[str, Union[str, list[str]]],
) -> pd.DataFrame:
    """Flexible multi-column aggregation.

    agg_spec example: {"revenue": ["sum", "mean"], "quantity": "sum"}
    """
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Group columns not found: {missing}")
    result = df.groupby(group_cols).agg(agg_spec)
    result.columns = ["_".join(c).strip("_") for c in result.columns]
    return result.reset_index().sort_values(result.columns[0], ascending=False)
