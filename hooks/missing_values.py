"""Missing-value report hook — pure pandas, no LLM."""

import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame showing null count and percentage per column."""
    total = len(df)
    null_counts = df.isnull().sum()
    report = pd.DataFrame({
        "column":    null_counts.index,
        "null_count": null_counts.values,
        "null_pct":  (null_counts.values / total * 100).round(2),
        "dtype":     [str(df[c].dtype) for c in null_counts.index],
    })
    report = report[report["null_count"] > 0].sort_values("null_pct", ascending=False)
    if report.empty:
        print("No missing values found.")
    return report.reset_index(drop=True)
