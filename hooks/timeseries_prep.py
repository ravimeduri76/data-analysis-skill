"""
timeseries_prep.py -- Time-series preprocessing: detect structure, fill gaps,
                      treat outliers.

Public API
----------
detect_time_index(df, time_col, value_cols, domain_ctx) -> TimeSeriesConfig
detect_seasonality(series, freq, method)                -> SeasonalityInfo
gp_gap_fill(series, time_index, kernel, max_gap_pct)    -> GapFillResult
dtw_impute(series, reference_series, window)            -> GapFillResult
ts_outlier_treatment(series, method, threshold, seasonal_period) -> (cleaned, flags)
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Graceful imports ──────────────────────────────────────────────────────────

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    _GP_OK = True
except ImportError:
    _GP_OK = False

try:
    from tslearn.metrics import dtw_path
    _DTW_OK = True
except ImportError:
    _DTW_OK = False

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf
    _STL_OK = True
except ImportError:
    _STL_OK = False


def _require_gp() -> None:
    if not _GP_OK:
        raise ImportError(
            "scikit-learn is required for GP gap filling.\n"
            "Install with:  pip install scikit-learn"
        )


def _require_dtw() -> None:
    if not _DTW_OK:
        raise ImportError(
            "tslearn is required for DTW imputation.\n"
            "Install with:  pip install tslearn"
        )


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class TimeSeriesConfig:
    time_col: str
    value_cols: List[str]
    freq: Optional[str]
    freq_confidence: str          # "high", "medium", "low"
    is_regular: bool
    n_gaps: int
    gap_pct: float
    date_range: Tuple[str, str]   # (min_date_iso, max_date_iso)
    n_observations: int
    warnings: List[str]


@dataclasses.dataclass
class SeasonalityInfo:
    period: int
    strength: float
    seasonal_type: str            # "additive" or "multiplicative"
    method: str                   # "acf", "fft", "stl"
    confidence: str               # "high", "medium", "low"


@dataclasses.dataclass
class GapFillResult:
    filled_series: pd.Series
    original_series: pd.Series
    uncertainty: Optional[pd.Series]
    gap_mask: pd.Series
    n_gaps_filled: int
    mean_uncertainty: float
    method: str
    kernel_description: str


# ── Time-column hints ─────────────────────────────────────────────────────────

_TIME_HINTS: List[str] = [
    "date", "time", "timestamp", "datetime", "dt", "period",
    "month", "year", "week", "day", "hour",
]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_time_index(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    value_cols: Optional[List[str]] = None,
    domain_ctx: Optional[dict] = None,
) -> TimeSeriesConfig:
    """Auto-detect the datetime column and infer frequency.

    Parameters
    ----------
    df          : input DataFrame
    time_col    : explicit datetime column name (auto-detected if None)
    value_cols  : explicit value columns (auto-detected if None)
    domain_ctx  : domain context dict (for domain-aware hints)

    Returns
    -------
    TimeSeriesConfig with detected parameters and any warnings.
    """
    issues: List[str] = []

    # ── 1. Find time column ──────────────────────────────────────────────
    if time_col is None:
        time_col = _auto_detect_time_col(df, issues)
    elif time_col not in df.columns:
        issues.append(f"Specified time_col '{time_col}' not found in DataFrame")
        time_col = _auto_detect_time_col(df, issues)

    if time_col is None:
        return TimeSeriesConfig(
            time_col="",
            value_cols=[],
            freq=None,
            freq_confidence="low",
            is_regular=False,
            n_gaps=0,
            gap_pct=0.0,
            date_range=("", ""),
            n_observations=len(df),
            warnings=issues + ["No datetime column detected"],
        )

    # Ensure datetime dtype
    ts = pd.to_datetime(df[time_col], errors="coerce")
    n_valid = int(ts.notna().sum())
    if n_valid < 3:
        issues.append(f"Only {n_valid} valid datetime values in '{time_col}'")
        return TimeSeriesConfig(
            time_col=time_col,
            value_cols=[],
            freq=None,
            freq_confidence="low",
            is_regular=False,
            n_gaps=0,
            gap_pct=0.0,
            date_range=("", ""),
            n_observations=n_valid,
            warnings=issues,
        )

    ts_clean = ts.dropna().sort_values()

    # ── 2. Find value columns ────────────────────────────────────────────
    if value_cols is None:
        value_cols = [
            c for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
    if not value_cols:
        issues.append("No numeric value columns found")

    # ── 3. Infer frequency ───────────────────────────────────────────────
    freq, freq_confidence = _infer_frequency(ts_clean, issues)

    # ── 4. Compute gaps ──────────────────────────────────────────────────
    n_gaps = 0
    gap_pct = 0.0
    is_regular = True

    if freq is not None:
        try:
            expected = pd.date_range(
                start=ts_clean.iloc[0], end=ts_clean.iloc[-1], freq=freq,
            )
            n_expected = len(expected)
            n_gaps = max(0, n_expected - n_valid)
            gap_pct = round(n_gaps / n_expected * 100, 2) if n_expected > 0 else 0.0
            is_regular = n_gaps == 0
        except Exception:
            issues.append("Could not compute gap count against expected grid")
    else:
        is_regular = False

    date_range = (
        str(ts_clean.iloc[0].isoformat()),
        str(ts_clean.iloc[-1].isoformat()),
    )

    return TimeSeriesConfig(
        time_col=time_col,
        value_cols=value_cols,
        freq=freq,
        freq_confidence=freq_confidence,
        is_regular=is_regular,
        n_gaps=n_gaps,
        gap_pct=gap_pct,
        date_range=date_range,
        n_observations=n_valid,
        warnings=issues,
    )


def detect_seasonality(
    series: pd.Series,
    freq: Optional[str] = None,
    method: str = "auto",
) -> SeasonalityInfo:
    """Detect seasonality using ACF peak detection, FFT, or STL.

    Parameters
    ----------
    series  : time-series values (NaN-dropped, sorted by time)
    freq    : pandas freq string (used to set candidate periods)
    method  : "auto", "acf", "fft", or "stl"

    Returns
    -------
    SeasonalityInfo with detected period, strength, and type.
    """
    s = series.dropna().values.astype(float)
    if len(s) < 6:
        return SeasonalityInfo(
            period=1, strength=0.0, seasonal_type="additive",
            method="none", confidence="low",
        )

    # Candidate periods from freq
    candidate_periods = _candidate_periods_from_freq(freq, len(s))

    # ── Try ACF first ────────────────────────────────────────────────────
    if method in ("auto", "acf"):
        result = _acf_seasonality(s, candidate_periods)
        if result is not None:
            return result

    # ── FFT fallback ─────────────────────────────────────────────────────
    if method in ("auto", "fft"):
        result = _fft_seasonality(s, candidate_periods)
        if result is not None:
            return result

    # ── STL fallback (requires statsmodels) ──────────────────────────────
    if method in ("auto", "stl") and _STL_OK and len(s) >= 14:
        result = _stl_seasonality(s, candidate_periods)
        if result is not None:
            return result

    return SeasonalityInfo(
        period=1, strength=0.0, seasonal_type="additive",
        method="none", confidence="low",
    )


def gp_gap_fill(
    series: pd.Series,
    time_index: pd.DatetimeIndex,
    kernel: object = None,
    max_gap_pct: float = 0.30,
) -> GapFillResult:
    """Fill gaps using Gaussian Process regression with uncertainty.

    Falls back to linear interpolation if sklearn is unavailable or
    gap_pct > max_gap_pct.

    Parameters
    ----------
    series      : values aligned to time_index (NaN = gap)
    time_index  : DatetimeIndex (must be same length as series)
    kernel      : sklearn GP kernel (auto-selected if None)
    max_gap_pct : max allowed gap fraction for GP (else linear fallback)

    Returns
    -------
    GapFillResult with filled values and uncertainty bands.
    """
    original = series.copy()
    gap_mask = series.isna()
    n_gaps = int(gap_mask.sum())

    if n_gaps == 0:
        return GapFillResult(
            filled_series=series.copy(),
            original_series=original,
            uncertainty=pd.Series(0.0, index=series.index),
            gap_mask=gap_mask,
            n_gaps_filled=0,
            mean_uncertainty=0.0,
            method="none",
            kernel_description="no gaps",
        )

    gap_frac = n_gaps / len(series)

    # Fallback: linear interpolation
    if not _GP_OK or gap_frac > max_gap_pct:
        method = "linear_interpolation"
        if not _GP_OK:
            method += " (sklearn unavailable)"
        elif gap_frac > max_gap_pct:
            method += f" (gap_pct {gap_frac:.1%} > max {max_gap_pct:.1%})"

        filled = series.interpolate(method="linear").ffill().bfill()
        return GapFillResult(
            filled_series=filled,
            original_series=original,
            uncertainty=None,
            gap_mask=gap_mask,
            n_gaps_filled=n_gaps,
            mean_uncertainty=0.0,
            method=method,
            kernel_description="N/A",
        )

    # GP regression
    # Convert time to numeric (days from start)
    t_numeric = (time_index - time_index[0]).total_seconds().values / 86400.0
    t_numeric = t_numeric.reshape(-1, 1)

    obs_mask = ~gap_mask.values
    X_obs = t_numeric[obs_mask]
    y_obs = series.values[obs_mask].astype(float)

    # Normalise y for numerical stability
    y_mean = float(np.mean(y_obs))
    y_std = float(np.std(y_obs)) or 1.0
    y_norm = (y_obs - y_mean) / y_std

    if kernel is None:
        kernel = 1.0 * RBF(length_scale=30.0, length_scale_bounds=(1.0, 365.0)) + \
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, normalize_y=False,
        )
        gp.fit(X_obs, y_norm)

    X_all = t_numeric
    y_pred_norm, y_std_pred = gp.predict(X_all, return_std=True)

    # Denormalise
    y_pred = y_pred_norm * y_std + y_mean
    uncertainty = y_std_pred * y_std

    # Fill only gaps
    filled_values = series.values.copy().astype(float)
    filled_values[gap_mask.values] = y_pred[gap_mask.values]

    filled_series = pd.Series(filled_values, index=series.index, name=series.name)
    uncertainty_series = pd.Series(uncertainty, index=series.index)

    kernel_desc = str(gp.kernel_) if hasattr(gp, "kernel_") else str(kernel)

    return GapFillResult(
        filled_series=filled_series,
        original_series=original,
        uncertainty=uncertainty_series,
        gap_mask=gap_mask,
        n_gaps_filled=n_gaps,
        mean_uncertainty=round(float(uncertainty[gap_mask.values].mean()), 4) if n_gaps > 0 else 0.0,
        method="gaussian_process",
        kernel_description=kernel_desc,
    )


def dtw_impute(
    series: pd.Series,
    reference_series: pd.Series,
    window: Optional[int] = None,
) -> GapFillResult:
    """Impute gaps by DTW-aligning to a reference series.

    Falls back to GP gap fill if tslearn unavailable.

    Parameters
    ----------
    series           : target series with gaps (NaN)
    reference_series : complete reference series for alignment
    window           : Sakoe-Chiba band width (None = auto)

    Returns
    -------
    GapFillResult
    """
    original = series.copy()
    gap_mask = series.isna()
    n_gaps = int(gap_mask.sum())

    if n_gaps == 0:
        return GapFillResult(
            filled_series=series.copy(),
            original_series=original,
            uncertainty=pd.Series(0.0, index=series.index),
            gap_mask=gap_mask,
            n_gaps_filled=0,
            mean_uncertainty=0.0,
            method="none",
            kernel_description="no gaps",
        )

    if not _DTW_OK:
        # Fall back to GP
        if hasattr(series.index, "freq") or isinstance(series.index, pd.DatetimeIndex):
            return gp_gap_fill(series, series.index)
        # Linear fallback
        filled = series.interpolate(method="linear").ffill().bfill()
        return GapFillResult(
            filled_series=filled,
            original_series=original,
            uncertainty=None,
            gap_mask=gap_mask,
            n_gaps_filled=n_gaps,
            mean_uncertainty=0.0,
            method="linear_interpolation (tslearn unavailable)",
            kernel_description="N/A",
        )

    # Prepare arrays
    s_vals = series.fillna(0).values.astype(float).reshape(-1, 1)
    r_vals = reference_series.values.astype(float).reshape(-1, 1)

    if window is None:
        window = max(5, len(series) // 10)

    try:
        path, _ = dtw_path(s_vals, r_vals, global_constraint="sakoe_chiba",
                           sakoe_chiba_radius=window)
    except Exception:
        # Fallback without constraint
        path, _ = dtw_path(s_vals, r_vals)

    # Build mapping: for each gap index in series, find aligned ref index
    filled_values = series.values.copy().astype(float)
    for idx_s, idx_r in path:
        if gap_mask.iloc[idx_s]:
            filled_values[idx_s] = reference_series.values[idx_r]

    filled_series = pd.Series(filled_values, index=series.index, name=series.name)

    return GapFillResult(
        filled_series=filled_series,
        original_series=original,
        uncertainty=None,
        gap_mask=gap_mask,
        n_gaps_filled=n_gaps,
        mean_uncertainty=0.0,
        method="dtw_alignment",
        kernel_description=f"sakoe_chiba_radius={window}",
    )


def ts_outlier_treatment(
    series: pd.Series,
    method: str = "stl_residual",
    threshold: float = 3.0,
    seasonal_period: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Detect and treat outliers in a time series.

    Parameters
    ----------
    series          : input time series (may have NaN)
    method          : "stl_residual", "rolling_zscore", or "seasonal_iqr"
    threshold       : z-score or IQR multiplier threshold
    seasonal_period : required for STL; auto-guessed if None

    Returns
    -------
    (cleaned_series, flags) where flags is a boolean Series (True = outlier).
    """
    s = series.copy()
    flags = pd.Series(False, index=s.index)
    valid_mask = s.notna()

    if valid_mask.sum() < 6:
        return s, flags

    # ── STL residual method ──────────────────────────────────────────────
    if method == "stl_residual" and _STL_OK:
        if seasonal_period is None:
            seasonal_period = _guess_seasonal_period(len(s))
        if seasonal_period < 2 or valid_mask.sum() < 2 * seasonal_period:
            # Fall back to rolling z-score
            return ts_outlier_treatment(s, method="rolling_zscore",
                                        threshold=threshold)
        try:
            filled = s.interpolate(method="linear").ffill().bfill()
            period = seasonal_period if seasonal_period % 2 == 1 else seasonal_period + 1
            stl = STL(filled, period=period, robust=True)
            result = stl.fit()
            residuals = result.resid
            res_mean = float(residuals.mean())
            res_std = float(residuals.std()) or 1.0
            z_scores = (residuals - res_mean).abs() / res_std
            outlier_mask = z_scores > threshold
            flags = outlier_mask & valid_mask
            # Replace outliers with trend + seasonal
            s[flags] = (result.trend + result.seasonal)[flags]
            return s, flags
        except Exception:
            return ts_outlier_treatment(s, method="rolling_zscore",
                                        threshold=threshold)

    # ── Rolling z-score method ───────────────────────────────────────────
    if method in ("rolling_zscore", "stl_residual"):
        window = max(7, len(s) // 20)
        rolling_mean = s.rolling(window=window, center=True, min_periods=3).mean()
        rolling_std = s.rolling(window=window, center=True, min_periods=3).std()
        rolling_std = rolling_std.replace(0, np.nan).ffill().bfill().fillna(1.0)
        z_scores = (s - rolling_mean).abs() / rolling_std
        outlier_mask = z_scores > threshold
        flags = outlier_mask & valid_mask
        s[flags] = rolling_mean[flags]
        return s, flags

    # ── Seasonal IQR method ──────────────────────────────────────────────
    if method == "seasonal_iqr":
        if seasonal_period is None:
            seasonal_period = _guess_seasonal_period(len(s))
        # Group by position in cycle
        positions = np.arange(len(s)) % seasonal_period
        for pos in range(seasonal_period):
            mask = positions == pos
            vals = s[mask].dropna()
            if len(vals) < 4:
                continue
            q1 = float(vals.quantile(0.25))
            q3 = float(vals.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            pos_outliers = mask & valid_mask & ((s < lower) | (s > upper))
            flags |= pos_outliers
            median_val = float(vals.median())
            s[pos_outliers] = median_val
        return s, flags

    # Unknown method: no-op
    return s, flags


# ── Private helpers ───────────────────────────────────────────────────────────

def _auto_detect_time_col(df: pd.DataFrame, issues: List[str]) -> Optional[str]:
    """Auto-detect the best datetime column."""
    # 1. Already-datetime columns
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        return dt_cols[0]

    # 2. Column name hints
    candidates: List[Tuple[str, int]] = []
    for c in df.columns:
        cl = c.lower().replace("_", " ").replace("-", " ")
        score = sum(1 for h in _TIME_HINTS if h in cl)
        if score > 0:
            candidates.append((c, score))

    # Try to parse candidates as datetime
    candidates.sort(key=lambda x: -x[1])
    for col_name, _ in candidates:
        try:
            parsed = pd.to_datetime(df[col_name], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                return col_name
        except Exception:
            continue

    # 3. Brute-force: try all object/string columns
    for c in df.select_dtypes(include=["object", "string"]).columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.7:
                issues.append(f"Auto-detected '{c}' as time column (parsed from string)")
                return c
        except Exception:
            continue

    issues.append("No datetime column could be detected")
    return None


def _infer_frequency(
    ts_clean: pd.Series, issues: List[str],
) -> Tuple[Optional[str], str]:
    """Infer pandas frequency string from a sorted datetime series."""
    if len(ts_clean) < 3:
        return None, "low"

    # Try pd.infer_freq
    try:
        idx = pd.DatetimeIndex(ts_clean.values)
        freq = pd.infer_freq(idx)
        if freq is not None:
            return freq, "high"
    except Exception:
        pass

    # Mode-of-diff fallback
    diffs = ts_clean.diff().dropna()
    if len(diffs) == 0:
        return None, "low"

    mode_diff = diffs.mode()
    if len(mode_diff) == 0:
        return None, "low"

    median_diff = mode_diff.iloc[0]
    days = median_diff.total_seconds() / 86400.0

    freq_map = [
        (0.04, "h"),    # ~1 hour
        (0.5, "h"),     # sub-daily
        (1.5, "D"),     # daily
        (8, "W"),       # weekly
        (20, "SMS"),    # semi-monthly
        (35, "MS"),     # monthly
        (100, "QS"),    # quarterly
        (400, "YS"),    # yearly
    ]

    for upper, f in freq_map:
        if days <= upper:
            # Check how many diffs agree
            agreement = (diffs == median_diff).mean()
            conf = "high" if agreement > 0.9 else "medium" if agreement > 0.6 else "low"
            return f, conf

    issues.append(f"Median diff={days:.1f} days does not match a standard frequency")
    return None, "low"


def _candidate_periods_from_freq(
    freq: Optional[str], n: int,
) -> List[int]:
    """Return candidate seasonal periods for a given frequency."""
    if freq is None:
        return [7, 12, 24, 52, 365]

    fl = freq.lower().rstrip("s")
    candidates: Dict[str, List[int]] = {
        "h": [24, 168],         # daily, weekly
        "d": [7, 30, 365],      # weekly, monthly, yearly
        "w": [4, 13, 52],       # monthly, quarterly, yearly
        "sm": [2, 6, 24],       # monthly, quarterly, yearly
        "m": [3, 6, 12],        # quarterly, semi-annual, yearly
        "q": [4],               # yearly
        "y": [],                # no sub-annual
    }

    for key, periods in candidates.items():
        if fl.startswith(key):
            return [p for p in periods if p < n // 2]

    return [7, 12, 24, 52]


def _acf_seasonality(
    s: np.ndarray, candidate_periods: List[int],
) -> Optional[SeasonalityInfo]:
    """Detect seasonality via ACF peak detection."""
    if not _STL_OK:
        return None

    n = len(s)
    max_lag = min(n // 2 - 1, max(candidate_periods) + 2) if candidate_periods else n // 2 - 1
    if max_lag < 3:
        return None

    try:
        acf_vals = acf(s, nlags=max_lag, fft=True)
    except Exception:
        return None

    # Find peaks among candidate periods
    best_period = 1
    best_acf = 0.0

    for p in candidate_periods:
        if p < 2 or p >= len(acf_vals):
            continue
        val = acf_vals[p]
        if val > best_acf:
            best_acf = val
            best_period = p

    if best_acf < 0.1:
        return None

    strength = float(best_acf)
    conf = "high" if strength > 0.5 else "medium" if strength > 0.25 else "low"

    # Detect additive vs multiplicative
    seasonal_type = _detect_seasonal_type(s, best_period)

    return SeasonalityInfo(
        period=best_period,
        strength=strength,
        seasonal_type=seasonal_type,
        method="acf",
        confidence=conf,
    )


def _fft_seasonality(
    s: np.ndarray, candidate_periods: List[int],
) -> Optional[SeasonalityInfo]:
    """Detect seasonality via FFT peak detection."""
    n = len(s)
    if n < 6:
        return None

    # Remove trend (simple differencing)
    detrended = np.diff(s)
    fft_vals = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(len(detrended))

    # Find dominant frequency
    if len(fft_vals) < 3:
        return None

    # Skip DC component
    fft_vals[0] = 0

    best_period = 1
    best_power = 0.0
    total_power = float(np.sum(fft_vals ** 2)) or 1.0

    for i, (f, p) in enumerate(zip(freqs, fft_vals)):
        if f <= 0:
            continue
        period = int(round(1.0 / f))
        if period < 2 or period > n // 2:
            continue
        power = float(p ** 2)
        if candidate_periods and period not in candidate_periods:
            # Allow ±1 tolerance
            if not any(abs(period - cp) <= 1 for cp in candidate_periods):
                continue
        if power > best_power:
            best_power = power
            best_period = period

    strength = best_power / total_power if total_power > 0 else 0.0
    if strength < 0.05:
        return None

    conf = "high" if strength > 0.3 else "medium" if strength > 0.1 else "low"
    seasonal_type = _detect_seasonal_type(s, best_period)

    return SeasonalityInfo(
        period=best_period,
        strength=round(strength, 4),
        seasonal_type=seasonal_type,
        method="fft",
        confidence=conf,
    )


def _stl_seasonality(
    s: np.ndarray, candidate_periods: List[int],
) -> Optional[SeasonalityInfo]:
    """Detect seasonality via STL decomposition."""
    if not _STL_OK or len(candidate_periods) == 0:
        return None

    best_result = None
    best_strength = 0.0

    for period in candidate_periods:
        if period < 2 or len(s) < 2 * period:
            continue
        p = period if period % 2 == 1 else period + 1
        try:
            stl = STL(pd.Series(s), period=p, robust=True)
            result = stl.fit()
            var_resid = float(np.var(result.resid))
            var_detrended = float(np.var(result.seasonal + result.resid))
            if var_detrended > 0:
                strength = max(0.0, 1.0 - var_resid / var_detrended)
            else:
                strength = 0.0
            if strength > best_strength:
                best_strength = strength
                best_result = (period, strength)
        except Exception:
            continue

    if best_result is None or best_strength < 0.1:
        return None

    period, strength = best_result
    conf = "high" if strength > 0.6 else "medium" if strength > 0.3 else "low"
    seasonal_type = _detect_seasonal_type(s, period)

    return SeasonalityInfo(
        period=period,
        strength=round(strength, 4),
        seasonal_type=seasonal_type,
        method="stl",
        confidence=conf,
    )


def _detect_seasonal_type(s: np.ndarray, period: int) -> str:
    """Determine if seasonality is additive or multiplicative."""
    if len(s) < 2 * period or np.any(s <= 0):
        return "additive"

    # Compare std of seasonal differences vs seasonal ratios
    n_cycles = len(s) // period
    if n_cycles < 2:
        return "additive"

    try:
        cycles = s[: n_cycles * period].reshape(n_cycles, period)
        cycle_means = cycles.mean(axis=1, keepdims=True)
        cycle_means = np.where(cycle_means == 0, 1.0, cycle_means)

        diffs = cycles - cycle_means
        ratios = cycles / cycle_means

        cv_diffs = float(np.std(diffs.std(axis=0)))
        cv_ratios = float(np.std(ratios.std(axis=0)))

        return "multiplicative" if cv_ratios < cv_diffs else "additive"
    except Exception:
        return "additive"


def _guess_seasonal_period(n: int) -> int:
    """Guess a seasonal period based on series length."""
    if n >= 730:
        return 365
    if n >= 104:
        return 52
    if n >= 60:
        return 12
    if n >= 14:
        return 7
    return max(2, n // 3)
