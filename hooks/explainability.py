"""
explainability.py -- Provenance cards showing HOW each key finding was produced.

Public API
----------
InsightCard                                                       dataclass
explain_groupby(result_df, group_cols, agg_col, func, total_rows) -> list[InsightCard]
explain_quality(qr_df, total_rows)                                -> list[InsightCard]
explain_anomalies(summary_df, domain_ctx)                         -> list[InsightCard]
explain_missing(missing_df, total_rows)                           -> list[InsightCard]
render_explanation(cards, title)                                  -> str
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import pandas as pd


@dataclasses.dataclass
class InsightCard:
    finding: str
    metric: str
    source_hook: str
    columns_used: str
    rows_analyzed: int
    total_rows: int
    method: str
    confidence: str
    caveats: str
    sql_equivalent: str


def explain_groupby(
    result_df: pd.DataFrame,
    group_cols: list,
    agg_col: str,
    func: str,
    total_rows: int,
) -> List[InsightCard]:
    """One card per row of result_df."""
    if result_df is None or result_df.empty:
        return [InsightCard(
            finding="No aggregation results returned",
            metric="N/A",
            source_hook="hooks/aggregation.py::group_agg",
            columns_used=", ".join(list(group_cols) + [agg_col]),
            rows_analyzed=0,
            total_rows=total_rows,
            method=f"pandas groupby + {func}",
            confidence="low",
            caveats="Empty result_df -- check that group_cols and agg_col exist in the dataset",
            sql_equivalent="N/A",
        )]

    # Detect metric column dynamically (non-group col in result_df)
    metric_cols = [c for c in result_df.columns if c not in group_cols]
    metric_col = metric_cols[0] if metric_cols else f"{func}_{agg_col}"

    # Build SQL equivalent
    group_clause = ", ".join(group_cols)
    if func in ("median", "percentile"):
        sql_func = (
            f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {agg_col})"
            f" /* dialect: PostgreSQL / SQL Server */"
        )
    else:
        sql_func = f"{func.upper()}({agg_col})"

    sql = (
        f"SELECT {group_clause}, {sql_func} "
        f"FROM dataset GROUP BY {group_clause} ORDER BY 2 DESC"
    )

    cards: List[InsightCard] = []
    for _, row in result_df.iterrows():
        group_vals = {c: row[c] for c in group_cols}
        group_str = ", ".join(f"{k}={v}" for k, v in group_vals.items())
        metric_val = row[metric_col]
        pct = (metric_val / total_rows * 100) if total_rows > 0 else 0.0

        cards.append(InsightCard(
            finding=f"{func.upper()} of {agg_col} where {group_str} = {metric_val}",
            metric=f"{metric_col} = {metric_val} ({pct:.1f}% of total rows)",
            source_hook="hooks/aggregation.py::group_agg",
            columns_used=", ".join(list(group_cols) + [agg_col]),
            rows_analyzed=total_rows,
            total_rows=total_rows,
            method=f"pandas groupby + {func}",
            confidence="high",
            caveats="",
            sql_equivalent=sql,
        ))

    return cards


def explain_quality(qr_df: pd.DataFrame, total_rows: int) -> List[InsightCard]:
    """One card per column with issues != 'OK'; one overall summary card."""
    cards: List[InsightCard] = []

    if qr_df is None or qr_df.empty:
        return [InsightCard(
            finding="No quality report available",
            metric="N/A",
            source_hook="hooks/data_quality.py::quality_report",
            columns_used="all",
            rows_analyzed=total_rows,
            total_rows=total_rows,
            method="quality_report",
            confidence="low",
            caveats="Empty quality report",
            sql_equivalent="N/A",
        )]

    flagged = qr_df[qr_df["issues"] != "OK"]

    for _, row in flagged.iterrows():
        null_pct = float(row["null_pct"])
        if null_pct == 100.0:
            caveats = "Column is entirely null -- drop before analysis; offers zero information"
        elif null_pct >= 80.0:
            caveats = (
                f"Column has {null_pct}% nulls -- likely sparse attribute; "
                "impute only if domain-justified"
            )
        elif null_pct >= 20.0:
            caveats = (
                f"Column has {null_pct}% nulls -- investigate null pattern "
                "(MCAR/MAR/MNAR) before imputing"
            )
        else:
            caveats = f"Issue tags: {row['issues']}"

        cards.append(InsightCard(
            finding=f"Column '{row['column']}' has quality issues: {row['issues']}",
            metric=f"quality_score={row['quality_score']}, null_pct={null_pct}%",
            source_hook="hooks/data_quality.py::quality_report",
            columns_used=str(row["column"]),
            rows_analyzed=total_rows,
            total_rows=total_rows,
            method="IQR outlier detection + null/skew/constant checks",
            confidence="high",
            caveats=caveats,
            sql_equivalent=(
                f"SELECT COUNT(*) - COUNT({row['column']}) AS null_count, "
                f"COUNT(DISTINCT {row['column']}) AS unique_vals FROM dataset"
            ),
        ))

    # Overall summary card
    avg_score = round(float(qr_df["quality_score"].mean()), 1) if not qr_df.empty else 0.0
    grade = (
        "A" if avg_score >= 85 else
        "B" if avg_score >= 70 else
        "C" if avg_score >= 55 else
        "D" if avg_score >= 40 else "F"
    )
    ok_count = int((qr_df["issues"] == "OK").sum())
    total_cols = len(qr_df)

    cards.append(InsightCard(
        finding=f"Overall dataset quality grade: {grade} ({avg_score}/100)",
        metric=f"{ok_count}/{total_cols} columns are clean (OK)",
        source_hook="hooks/data_quality.py::quality_report",
        columns_used="all",
        rows_analyzed=total_rows,
        total_rows=total_rows,
        method="Composite score: null_pct + constant + skew + outlier deductions",
        confidence="high",
        caveats="Score is a heuristic aggregate -- domain context may change severity of individual flags",
        sql_equivalent="N/A -- computed from column-level statistics",
    ))

    return cards


def explain_anomalies(summary_df: pd.DataFrame, domain_ctx: dict) -> List[InsightCard]:
    """One card per row in summary_df where anomaly_count > 0."""
    cards: List[InsightCard] = []
    anomaly_guidance = str(domain_ctx.get("anomaly_guidance", ""))

    if summary_df is None or summary_df.empty:
        return [InsightCard(
            finding="No anomaly detection results available",
            metric="N/A",
            source_hook="hooks/anomaly_detection.py::detect_anomalies",
            columns_used="all",
            rows_analyzed=0,
            total_rows=0,
            method="detect_anomalies",
            confidence="low",
            caveats="Empty summary_df",
            sql_equivalent="N/A",
        )]

    flagged = summary_df[summary_df["anomaly_count"] > 0]

    for _, row in flagged.iterrows():
        method_str = str(row.get("method", "mod_z"))

        # Confidence based on method
        if "mod_z" in method_str or "iqr" in method_str:
            confidence = "high"
        elif "fuzzy" in method_str or "claude" in method_str.lower():
            confidence = "medium"
        else:
            confidence = "low"

        col_type = str(row.get("type", "numeric"))
        anom_count = int(row["anomaly_count"])
        anom_pct = float(row.get("anomaly_pct", 0.0))
        non_null = int(row.get("non_null", 0))

        if col_type == "numeric":
            anom_min = row.get("anomaly_min")
            anom_max = row.get("anomaly_max")
            col_mean = row.get("col_mean")
            metric_str = (
                f"{anom_count} anomalies ({anom_pct:.1f}%), "
                f"range=[{anom_min}, {anom_max}], mean={col_mean}"
            )
            sql = (
                f"SELECT {row['column']}, COUNT(*) FROM dataset "
                f"WHERE ABS({row['column']} - AVG({row['column']}) OVER()) "
                f"> 3 * STDDEV({row['column']}) OVER() "
                f"GROUP BY {row['column']}"
            )
        else:
            metric_str = f"{anom_count} anomalies ({anom_pct:.1f}%)"
            sql = (
                f"SELECT {row['column']}, COUNT(*) FROM dataset "
                f"WHERE {row['column']} NOT IN (<expected_values>) "
                f"GROUP BY {row['column']} ORDER BY 2 DESC"
            )

        caveats = anomaly_guidance if anomaly_guidance else "Review flagged values in context of domain"

        cards.append(InsightCard(
            finding=f"Column '{row['column']}' has {anom_count} anomalies ({anom_pct:.1f}%)",
            metric=metric_str,
            source_hook="hooks/anomaly_detection.py::detect_anomalies",
            columns_used=str(row["column"]),
            rows_analyzed=non_null,
            total_rows=non_null,
            method=method_str,
            confidence=confidence,
            caveats=caveats,
            sql_equivalent=sql,
        ))

    return cards


def explain_missing(missing_df: pd.DataFrame, total_rows: int) -> List[InsightCard]:
    """One card per row in missing_report output."""
    if missing_df is None or missing_df.empty:
        return []

    cards: List[InsightCard] = []

    for _, row in missing_df.iterrows():
        null_pct = float(row.get("null_pct", 0.0))
        null_count = int(row.get("null_count", 0))
        col = str(row["column"])

        if null_pct == 100.0:
            caveats = "Column is entirely null -- drop before analysis; offers zero information"
        elif null_pct > 80.0:
            caveats = (
                f"{null_pct:.1f}% null -- likely sparse attribute; "
                "impute only if domain-justified"
            )
        elif null_pct > 20.0:
            caveats = (
                f"{null_pct:.1f}% null -- investigate null pattern "
                "(MCAR/MAR/MNAR) before imputing"
            )
        else:
            caveats = f"{null_pct:.1f}% null -- low missingness; simple imputation likely safe"

        cards.append(InsightCard(
            finding=f"Column '{col}' has {null_count} missing values ({null_pct:.1f}%)",
            metric=f"null_count={null_count}, null_pct={null_pct:.1f}%",
            source_hook="hooks/missing_values.py::missing_report",
            columns_used=col,
            rows_analyzed=total_rows,
            total_rows=total_rows,
            method="pandas isnull().sum()",
            confidence="high",
            caveats=caveats,
            sql_equivalent=(
                f"SELECT COUNT(*) - COUNT({col}) AS null_count, "
                f"ROUND(100.0 * (COUNT(*) - COUNT({col})) / NULLIF(COUNT(*), 0), 2) AS null_pct "
                f"FROM dataset"
            ),
        ))

    return cards


def explain_timeseries(
    ts_config: object,
    comparison: object = None,
    forecast_result: object = None,
    domain_ctx: Optional[dict] = None,
    total_rows: int = 0,
) -> List[InsightCard]:
    """InsightCards for time-series analysis results.

    Parameters
    ----------
    ts_config       : TimeSeriesConfig from timeseries_prep
    comparison      : ModelComparison from timeseries_models
    forecast_result : ForecastResult from timeseries_models
    domain_ctx      : domain context dict
    total_rows      : total dataset row count
    """
    cards: List[InsightCard] = []
    n_obs = getattr(ts_config, "n_observations", total_rows)

    # Card 1: Time index detection
    time_col = getattr(ts_config, "time_col", "")
    freq = getattr(ts_config, "freq", None) or "irregular"
    freq_conf = getattr(ts_config, "freq_confidence", "low")
    cards.append(InsightCard(
        finding=f"Time index detected: '{time_col}' with frequency '{freq}'",
        metric=f"freq_confidence={freq_conf}, n_observations={n_obs}",
        source_hook="hooks/timeseries_prep.py::detect_time_index",
        columns_used=time_col,
        rows_analyzed=n_obs,
        total_rows=total_rows,
        method="datetime auto-detection + pd.infer_freq / mode-of-diff",
        confidence=freq_conf,
        caveats="" if freq_conf == "high" else "Frequency detection uncertain — verify manually",
        sql_equivalent="N/A",
    ))

    # Card 2: Gap assessment
    n_gaps = getattr(ts_config, "n_gaps", 0)
    gap_pct = getattr(ts_config, "gap_pct", 0.0)
    if n_gaps > 0:
        cards.append(InsightCard(
            finding=f"{n_gaps} gaps detected ({gap_pct}% of expected observations)",
            metric=f"n_gaps={n_gaps}, gap_pct={gap_pct}%",
            source_hook="hooks/timeseries_prep.py::detect_time_index",
            columns_used=time_col,
            rows_analyzed=n_obs,
            total_rows=total_rows,
            method="expected grid comparison",
            confidence="high" if freq_conf == "high" else "medium",
            caveats="Gap filling applied before modelling" if gap_pct > 0 else "",
            sql_equivalent="N/A",
        ))

    # Card 3: Best model + metrics
    if comparison is not None:
        best = getattr(comparison, "best_model_name", "N/A")
        rankings = getattr(comparison, "rankings", None)
        if rankings is not None and len(rankings) > 0:
            best_row = rankings.iloc[0]
            mape = best_row.get("MAPE", "N/A")
            rmse = best_row.get("RMSE", "N/A")
            cards.append(InsightCard(
                finding=f"Best time-series model: {best} (MAPE={mape}%)",
                metric=f"MAPE={mape}%, RMSE={rmse}",
                source_hook="hooks/timeseries_models.py::compare_models",
                columns_used=", ".join(getattr(ts_config, "value_cols", [])),
                rows_analyzed=n_obs,
                total_rows=total_rows,
                method="multi-model comparison (ES/ARIMA/Prophet)",
                confidence="high" if isinstance(mape, (int, float)) and mape < 10 else "medium",
                caveats="" if isinstance(mape, (int, float)) and mape < 20 else "High MAPE — consider additional features or longer history",
                sql_equivalent="N/A",
            ))

    # Card 4: Forecast summary
    if forecast_result is not None:
        horizon = getattr(forecast_result, "horizon", 0)
        model_name = getattr(forecast_result, "model_name", "unknown")
        point = getattr(forecast_result, "point_forecast", None)
        lower = getattr(forecast_result, "lower_bound", None)
        upper = getattr(forecast_result, "upper_bound", None)

        if point is not None and len(point) > 0:
            avg_width = float((upper - lower).mean()) if lower is not None and upper is not None else 0
            cards.append(InsightCard(
                finding=f"Forecast: {horizon} periods ahead using {model_name}",
                metric=f"avg_CI_width={avg_width:.2f}, range=[{float(point.iloc[0]):.2f}, {float(point.iloc[-1]):.2f}]",
                source_hook="hooks/timeseries_models.py::forecast",
                columns_used=", ".join(getattr(ts_config, "value_cols", [])),
                rows_analyzed=n_obs,
                total_rows=total_rows,
                method=f"{model_name} with prediction intervals",
                confidence="high" if avg_width < float(point.mean()) * 0.5 else "medium",
                caveats="Wide confidence intervals — forecast uncertainty is high" if avg_width > float(point.mean()) * 0.5 else "",
                sql_equivalent="N/A",
            ))

    # Card 5: Domain interpretation
    if domain_ctx:
        domain = domain_ctx.get("domain", "General")
        lens = domain_ctx.get("timeseries_lens", domain_ctx.get("analysis_lens", ""))
        if lens:
            cards.append(InsightCard(
                finding=f"Domain lens ({domain}): {lens}",
                metric="N/A",
                source_hook="hooks/domain_detector.py::detect_domain",
                columns_used="all",
                rows_analyzed=n_obs,
                total_rows=total_rows,
                method="domain-aware interpretation",
                confidence="medium",
                caveats="",
                sql_equivalent="N/A",
            ))

    return cards


def render_explanation(cards: List[InsightCard], title: str = "Analysis Explanation") -> str:
    """Render InsightCards as ASCII plain-text."""
    if not cards:
        return f"=== {title} ===\nNo insights to display.\n"

    lines = [f"=== {title} ===", f"Total insights: {len(cards)}", ""]

    for i, card in enumerate(cards, 1):
        rows_pct = (
            card.rows_analyzed / card.total_rows * 100
            if card.total_rows > 0
            else 0.0
        )
        lines.append(f"[{i}] {card.finding}")
        lines.append(f"    Metric     : {card.metric}")
        lines.append(f"    Hook       : {card.source_hook}")
        lines.append(f"    Columns    : {card.columns_used}")
        lines.append(
            f"    Rows       : {card.rows_analyzed:,} / {card.total_rows:,} ({rows_pct:.1f}%)"
        )
        lines.append(f"    Method     : {card.method}")
        lines.append(f"    Confidence : {card.confidence}")
        if card.caveats:
            lines.append(f"    Caveats    : {card.caveats}")
        if card.sql_equivalent and card.sql_equivalent != "N/A":
            lines.append(f"    SQL        : {card.sql_equivalent}")
        lines.append("")

    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
