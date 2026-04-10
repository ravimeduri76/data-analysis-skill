"""
timeseries_viz.py -- Time-series visualisation plots.

Public API
----------
plot_decomposition(series, time_index, seasonal_info, out_dir, title)  -> str
plot_forecast(series, forecast_result, out_dir, title)                 -> str
plot_model_comparison(comparison, out_dir)                             -> str
plot_residual_diagnostics(result, out_dir, title)                      -> str
plot_gap_fill(original, gap_fill_result, out_dir, title)               -> str
plot_stationarity(series, arima_result, out_dir)                       -> str
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Graceful imports ──────────────────────────────────────────────────────────

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, pacf
    _STL_OK = True
except ImportError:
    _STL_OK = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_plt():
    """Return pyplot with Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ts() -> str:
    """UTC timestamp for filenames (microsecond precision, no collisions)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]


def _ensure_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_decomposition(
    series: pd.Series,
    time_index: pd.DatetimeIndex,
    seasonal_info: object,
    out_dir: str,
    title: str = "",
) -> str:
    """4-panel decomposition: observed / trend / seasonal / residual.

    Parameters
    ----------
    series        : time-series values
    time_index    : aligned DatetimeIndex
    seasonal_info : SeasonalityInfo from detect_seasonality
    out_dir       : directory to save the plot
    title         : optional plot title

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    _ensure_dir(out_dir)
    plt = _get_plt()

    period = getattr(seasonal_info, "period", 7)
    s = series.copy()
    s.index = time_index

    # Fill NaN for STL
    s_filled = s.interpolate(method="linear").ffill().bfill()

    if _STL_OK and len(s_filled) >= 2 * period and period >= 2:
        p = period if period % 2 == 1 else period + 1
        try:
            stl = STL(s_filled, period=p, robust=True)
            result = stl.fit()
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
        except Exception:
            trend = s_filled.rolling(window=max(3, period), center=True, min_periods=1).mean()
            seasonal = s_filled - trend
            residual = pd.Series(0, index=s_filled.index)
    else:
        trend = s_filled.rolling(window=max(3, period), center=True, min_periods=1).mean()
        seasonal = s_filled - trend
        residual = pd.Series(0, index=s_filled.index)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    plot_title = title or "Time Series Decomposition"

    axes[0].plot(s.index, s.values, color="#4C72B0", linewidth=0.8)
    axes[0].set_ylabel("Observed")
    axes[0].set_title(plot_title)

    axes[1].plot(s.index, trend.values, color="#DD8452", linewidth=1.2)
    axes[1].set_ylabel("Trend")

    axes[2].plot(s.index, seasonal.values, color="#55A868", linewidth=0.8)
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(s.index, residual.values, color="#C44E52", linewidth=0.6, alpha=0.7)
    axes[3].axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    fig.tight_layout()
    path = os.path.join(out_dir, f"ts_decomposition_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def plot_forecast(
    series: pd.Series,
    forecast_result: object,
    out_dir: str,
    title: str = "",
) -> str:
    """Plot actual values + forecast with confidence bands.

    Parameters
    ----------
    series          : historical time-series values
    forecast_result : ForecastResult from forecast()
    out_dir         : directory to save the plot
    title           : optional plot title

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    _ensure_dir(out_dir)
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_title = title or f"Forecast — {forecast_result.model_name}"

    # Historical
    ax.plot(series.index, series.values, color="#4C72B0",
            linewidth=1.0, label="Actual")

    # Forecast
    fc = forecast_result.point_forecast
    lower = forecast_result.lower_bound
    upper = forecast_result.upper_bound

    # Create a forecast index that continues from the series
    if hasattr(series.index, "freq") and series.index.freq is not None:
        fc_index = pd.date_range(
            start=series.index[-1], periods=len(fc) + 1, freq=series.index.freq,
        )[1:]
    else:
        fc_index = np.arange(len(series), len(series) + len(fc))

    ax.plot(fc_index, fc.values, color="#DD8452", linewidth=1.5, label="Forecast")

    # Confidence band
    conf_pct = int(forecast_result.confidence_level * 100)
    ax.fill_between(
        fc_index, lower.values, upper.values,
        alpha=0.2, color="#DD8452", label=f"{conf_pct}% CI",
    )

    ax.set_title(plot_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")

    fig.tight_layout()
    path = os.path.join(out_dir, f"ts_forecast_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def plot_model_comparison(
    comparison: object,
    out_dir: str,
) -> str:
    """Grouped bar chart comparing model metrics.

    Parameters
    ----------
    comparison : ModelComparison from compare_models
    out_dir    : directory to save the plot

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    _ensure_dir(out_dir)
    plt = _get_plt()

    rankings = comparison.rankings
    metrics = ["MAE", "RMSE", "MAPE"]
    available = [m for m in metrics if m in rankings.columns]

    n_models = len(rankings)
    n_metrics = len(available)
    x = np.arange(n_models)
    width = 0.8 / n_metrics

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.5), 5))

    for i, metric in enumerate(available):
        vals = rankings[metric].values.astype(float)
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(rankings["model"].values, rotation=30, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title("Model Comparison")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, f"ts_model_comparison_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def plot_residual_diagnostics(
    result: object,
    out_dir: str,
    title: str = "",
) -> str:
    """4-panel residual diagnostics: ACF / PACF / Q-Q / histogram.

    Parameters
    ----------
    result  : any model result with .residuals attribute
    out_dir : directory to save the plot
    title   : optional plot title

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    _ensure_dir(out_dir)
    plt = _get_plt()

    residuals = getattr(result, "residuals", pd.Series(dtype=float))
    res = residuals.dropna().values.astype(float)

    if len(res) < 5:
        # Minimal fallback
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient residuals for diagnostics",
                ha="center", va="center", transform=ax.transAxes)
        path = os.path.join(out_dir, f"ts_residuals_{_ts()}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return os.path.abspath(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_title = title or "Residual Diagnostics"
    fig.suptitle(plot_title, fontsize=13)

    # ACF
    max_lag = min(40, len(res) // 2 - 1)
    if _STL_OK and max_lag >= 2:
        acf_vals = acf(res, nlags=max_lag, fft=True)
        axes[0, 0].bar(range(len(acf_vals)), acf_vals, color="#4C72B0", width=0.3)
        ci = 1.96 / np.sqrt(len(res))
        axes[0, 0].axhline(y=ci, color="red", linestyle="--", linewidth=0.5)
        axes[0, 0].axhline(y=-ci, color="red", linestyle="--", linewidth=0.5)
    else:
        axes[0, 0].text(0.5, 0.5, "ACF requires statsmodels",
                        ha="center", va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("ACF of Residuals")
    axes[0, 0].set_xlabel("Lag")

    # PACF
    if _STL_OK and max_lag >= 2:
        try:
            pacf_vals = pacf(res, nlags=max_lag, method="ywm")
            axes[0, 1].bar(range(len(pacf_vals)), pacf_vals, color="#DD8452", width=0.3)
            axes[0, 1].axhline(y=ci, color="red", linestyle="--", linewidth=0.5)
            axes[0, 1].axhline(y=-ci, color="red", linestyle="--", linewidth=0.5)
        except Exception:
            axes[0, 1].text(0.5, 0.5, "PACF computation failed",
                            ha="center", va="center", transform=axes[0, 1].transAxes)
    else:
        axes[0, 1].text(0.5, 0.5, "PACF requires statsmodels",
                        ha="center", va="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("PACF of Residuals")
    axes[0, 1].set_xlabel("Lag")

    # Q-Q plot
    try:
        from scipy.stats import probplot
        probplot(res, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
    except ImportError:
        sorted_res = np.sort(res)
        theoretical = np.linspace(-3, 3, len(sorted_res))
        axes[1, 0].scatter(theoretical, sorted_res, s=5, alpha=0.5, color="#55A868")
        axes[1, 0].set_title("Q-Q Plot (simplified)")

    # Histogram
    axes[1, 1].hist(res, bins=min(50, max(10, len(res) // 10)),
                     color="#C44E52", edgecolor="white", alpha=0.8)
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].set_xlabel("Residual")
    axes[1, 1].set_ylabel("Frequency")

    # Add mean/std annotation
    res_mean = float(np.mean(res))
    res_std = float(np.std(res))
    axes[1, 1].axvline(x=res_mean, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].text(0.97, 0.95, f"mean={res_mean:.3f}\nstd={res_std:.3f}",
                    transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    path = os.path.join(out_dir, f"ts_residuals_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def plot_gap_fill(
    original: pd.Series,
    gap_fill_result: object,
    out_dir: str,
    title: str = "",
) -> str:
    """Plot original series with gap regions highlighted + GP uncertainty.

    Parameters
    ----------
    original        : original series (with NaN gaps)
    gap_fill_result : GapFillResult from gp_gap_fill or dtw_impute
    out_dir         : directory to save the plot
    title           : optional plot title

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    _ensure_dir(out_dir)
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_title = title or f"Gap Filling ({gap_fill_result.method})"

    filled = gap_fill_result.filled_series
    gap_mask = gap_fill_result.gap_mask

    # Plot observed (non-gap) values
    obs_mask = ~gap_mask
    ax.plot(original.index[obs_mask], original.values[obs_mask],
            color="#4C72B0", linewidth=1.0, label="Observed")

    # Plot filled gap values
    if gap_mask.any():
        ax.scatter(filled.index[gap_mask], filled.values[gap_mask],
                   color="#DD8452", s=15, zorder=5, label="Filled")

        # Highlight gap regions
        for idx in filled.index[gap_mask]:
            ax.axvline(x=idx, color="#DD8452", alpha=0.05, linewidth=2)

    # Uncertainty bands (GP)
    uncertainty = gap_fill_result.uncertainty
    if uncertainty is not None and gap_mask.any():
        gap_idx = filled.index[gap_mask]
        gap_vals = filled.values[gap_mask]
        gap_unc = uncertainty.values[gap_mask]
        ax.fill_between(
            gap_idx,
            gap_vals - 1.96 * gap_unc,
            gap_vals + 1.96 * gap_unc,
            alpha=0.2, color="#DD8452", label="95% CI",
        )

    ax.set_title(plot_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")

    # Annotate gap count
    n_gaps = gap_fill_result.n_gaps_filled
    mean_unc = gap_fill_result.mean_uncertainty
    ann = f"{n_gaps} gaps filled"
    if mean_unc > 0:
        ann += f" (mean unc={mean_unc:.3f})"
    ax.text(0.97, 0.03, ann, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    path = os.path.join(out_dir, f"ts_gap_fill_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def plot_stationarity(
    series: pd.Series,
    arima_result: object,
    out_dir: str,
) -> str:
    """Plot rolling mean/std with ADF test annotation.

    Parameters
    ----------
    series       : original time-series values
    arima_result : ARIMAResult from fit_arima
    out_dir      : directory to save the plot

    Returns
    -------
    Absolute path to the saved PNG file.
    """
    _ensure_dir(out_dir)
    plt = _get_plt()

    s = series.dropna().astype(float)
    window = max(7, len(s) // 20)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(s.index, s.values, color="#4C72B0", linewidth=0.6, alpha=0.6, label="Original")

    rolling_mean = s.rolling(window=window, center=True).mean()
    rolling_std = s.rolling(window=window, center=True).std()

    ax.plot(s.index, rolling_mean.values, color="#DD8452", linewidth=1.5,
            label=f"Rolling Mean (w={window})")

    ax2 = ax.twinx()
    ax2.plot(s.index, rolling_std.values, color="#55A868", linewidth=1.2,
             alpha=0.7, label=f"Rolling Std (w={window})")
    ax2.set_ylabel("Standard Deviation", color="#55A868")

    # ADF annotation
    adf_p = getattr(arima_result, "adf_pvalue", None)
    is_stat = getattr(arima_result, "is_stationary", None)
    order = getattr(arima_result, "order", None)

    ann_lines = []
    if adf_p is not None:
        ann_lines.append(f"ADF p-value: {adf_p:.6f}")
    if is_stat is not None:
        ann_lines.append(f"Stationary: {'Yes' if is_stat else 'No'}")
    if order is not None:
        ann_lines.append(f"ARIMA order: {order}")

    if ann_lines:
        ann_text = "\n".join(ann_lines)
        ax.text(0.02, 0.95, ann_text, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_title("Stationarity Analysis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    # Combined legend
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    path = os.path.join(out_dir, f"ts_stationarity_{_ts()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)
