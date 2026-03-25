"""Column-level profiling — pure pandas, no LLM needed."""

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column profile table with dtype, null %, unique count, and stats."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column":   col,
            "dtype":    str(s.dtype),
            "count":    int(s.count()),
            "null_pct": round(s.isnull().mean() * 100, 2),
            "unique":   int(s.nunique()),
            "min":      _safe(s, "min"),
            "max":      _safe(s, "max"),
            "mean":     _safe(s, "mean"),
            "median":   _safe(s, "median"),
            "std":      _safe(s, "std"),
            "top_value": s.mode().iloc[0] if not s.mode().empty else None,
        })
    return pd.DataFrame(rows)


def _safe(series: pd.Series, func: str):
    try:
        val = getattr(series, func)()
        return round(float(val), 4) if isinstance(val, float) else val
    except Exception:
        return None
