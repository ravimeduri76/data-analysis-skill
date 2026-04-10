"""
timeseries_models.py -- Time-series model fitting, comparison, and forecasting.

Public API — Classical
----------------------
fit_exponential_smoothing(series, seasonal_periods, trend, seasonal, horizon) -> ESResult
fit_arima(series, order, seasonal_order, auto_order, horizon)                 -> ARIMAResult
fit_prophet(df, time_col, value_col, domain_ctx, holidays_country, horizon)   -> ProphetResult
compare_models(results, primary_metric)                                       -> ModelComparison
forecast(model_result, horizon, confidence)                                   -> ForecastResult
timeseries_summary(ts_config, comparison, forecast_result, domain_ctx)        -> str

Public API — Deep Learning
--------------------------
fit_tft(df, time_col, value_col, group_col, static_covariates, max_epochs, horizon) -> TFTResult
fit_bilstm(series, lookback, hidden_size, num_layers, max_epochs, horizon)          -> BiLSTMResult
"""

from __future__ import annotations

import dataclasses
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Graceful imports ──────────────────────────────────────────────────────────

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False

try:
    from pytorch_forecasting import TemporalFusionTransformer
    _TFT_OK = True
except ImportError:
    _TFT_OK = False

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


def _require_statsmodels() -> None:
    if not _STATSMODELS_OK:
        raise ImportError(
            "statsmodels is required for ES and ARIMA models.\n"
            "Install with:  pip install statsmodels"
        )


def _require_prophet() -> None:
    if not _PROPHET_OK:
        raise ImportError(
            "prophet is required for Prophet models.\n"
            "Install with:  pip install prophet"
        )


def _require_tft() -> None:
    if not _TFT_OK:
        raise ImportError(
            "pytorch-forecasting is required for TFT models.\n"
            "Install with:  pip install pytorch-forecasting torch"
        )


def _require_torch() -> None:
    if not _TORCH_OK:
        raise ImportError(
            "PyTorch is required for BiLSTM models.\n"
            "Install with:  pip install torch"
        )


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ESResult:
    fitted_values: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]      # MAE, RMSE, MAPE, AIC, BIC
    model: object
    params: Dict[str, Any]
    forecast: Optional[pd.Series]


@dataclasses.dataclass
class ARIMAResult:
    fitted_values: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]
    model: object
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    is_stationary: bool
    adf_pvalue: float
    forecast: Optional[pd.Series]


@dataclasses.dataclass
class ProphetResult:
    fitted_values: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]
    model: object
    components: Optional[pd.DataFrame]
    changepoints: List[str]
    forecast: Optional[pd.DataFrame]


@dataclasses.dataclass
class TFTResult:
    fitted_values: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]
    model: object
    attention_weights: Optional[Any]
    forecast: Optional[pd.Series]


@dataclasses.dataclass
class BiLSTMResult:
    fitted_values: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]
    model: object
    forecast: Optional[pd.Series]


@dataclasses.dataclass
class ModelComparison:
    rankings: pd.DataFrame
    best_model_name: str
    best_model_result: object
    all_results: Dict[str, object]


@dataclasses.dataclass
class ForecastResult:
    point_forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence_level: float
    horizon: int
    model_name: str


# ── Metric helpers ────────────────────────────────────────────────────────────

def _compute_metrics(
    actual: np.ndarray, fitted: np.ndarray,
    aic: float = np.nan, bic: float = np.nan,
) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, AIC, BIC."""
    mask = np.isfinite(actual) & np.isfinite(fitted)
    a = actual[mask]
    f = fitted[mask]
    if len(a) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "AIC": aic, "BIC": bic}

    errors = a - f
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: avoid division by zero
    nonzero = np.abs(a) > 1e-10
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs(errors[nonzero] / a[nonzero])) * 100)
    else:
        mape = np.nan

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE": round(mape, 2), "AIC": round(aic, 2), "BIC": round(bic, 2)}


# ── Public API — Classical ────────────────────────────────────────────────────

def fit_exponential_smoothing(
    series: pd.Series,
    seasonal_periods: Optional[int] = None,
    trend: str = "add",
    seasonal: str = "add",
    horizon: int = 0,
) -> ESResult:
    """Fit Holt-Winters exponential smoothing.

    Parameters
    ----------
    series           : time-series values (no NaN)
    seasonal_periods : e.g. 7 for daily with weekly seasonality
    trend            : "add", "mul", or None
    seasonal         : "add", "mul", or None
    horizon          : number of steps to forecast (0 = no forecast)
    """
    _require_statsmodels()

    s = series.dropna().astype(float)
    if len(s) < 4:
        raise ValueError("Need at least 4 observations for exponential smoothing")

    # If seasonal_periods not sufficient, disable seasonality
    if seasonal_periods is not None and len(s) < 2 * seasonal_periods:
        seasonal = None
        seasonal_periods = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            s,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)

    fitted = fit.fittedvalues
    residuals = s - fitted
    aic = float(fit.aic) if hasattr(fit, "aic") else np.nan
    bic = float(fit.bic) if hasattr(fit, "bic") else np.nan
    metrics = _compute_metrics(s.values, fitted.values, aic, bic)

    fcast = None
    if horizon > 0:
        fcast = fit.forecast(horizon)

    params = {
        "trend": trend,
        "seasonal": seasonal,
        "seasonal_periods": seasonal_periods,
        "smoothing_level": round(float(fit.params.get("smoothing_level", 0)), 4),
        "smoothing_trend": round(float(fit.params.get("smoothing_trend", 0)), 4),
        "smoothing_seasonal": round(float(fit.params.get("smoothing_seasonal", 0)), 4),
    }

    return ESResult(
        fitted_values=fitted,
        residuals=residuals,
        metrics=metrics,
        model=fit,
        params=params,
        forecast=fcast,
    )


def fit_arima(
    series: pd.Series,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    auto_order: bool = True,
    horizon: int = 0,
) -> ARIMAResult:
    """Fit ARIMA/SARIMA model.

    Parameters
    ----------
    series         : time-series values (no NaN)
    order          : (p, d, q) — auto-selected if None and auto_order=True
    seasonal_order : (P, D, Q, s) — auto-selected if None and auto_order=True
    auto_order     : if True, use AIC grid search for best (p,d,q)
    horizon        : number of steps to forecast (0 = no forecast)
    """
    _require_statsmodels()

    s = series.dropna().astype(float)
    if len(s) < 10:
        raise ValueError("Need at least 10 observations for ARIMA")

    # ADF test for stationarity
    adf_result = adfuller(s, autolag="AIC")
    adf_pvalue = float(adf_result[1])
    is_stationary = adf_pvalue < 0.05

    # Auto-detect d
    if order is None and auto_order:
        d = 0 if is_stationary else 1
        if d == 1:
            s_diff = s.diff().dropna()
            adf2 = adfuller(s_diff, autolag="AIC")
            if adf2[1] > 0.05:
                d = 2
        order = _auto_arima_order(s, d)
    elif order is None:
        order = (1, 1, 1)

    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            s, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=200)

    fitted = fit.fittedvalues
    residuals = s - fitted
    aic = float(fit.aic)
    bic = float(fit.bic)
    metrics = _compute_metrics(s.values, fitted.values, aic, bic)

    fcast = None
    if horizon > 0:
        pred = fit.get_forecast(steps=horizon)
        fcast = pred.predicted_mean

    return ARIMAResult(
        fitted_values=fitted,
        residuals=residuals,
        metrics=metrics,
        model=fit,
        order=order,
        seasonal_order=seasonal_order,
        is_stationary=is_stationary,
        adf_pvalue=round(adf_pvalue, 6),
        forecast=fcast,
    )


def fit_prophet(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    domain_ctx: Optional[dict] = None,
    holidays_country: Optional[str] = None,
    horizon: int = 0,
) -> ProphetResult:
    """Fit Facebook Prophet model.

    Parameters
    ----------
    df               : DataFrame with time and value columns
    time_col         : datetime column name
    value_col        : numeric value column name
    domain_ctx       : domain context (for domain-aware holidays)
    holidays_country : ISO country code for holidays (e.g. "US")
    horizon          : number of periods to forecast
    """
    _require_prophet()

    # Prepare Prophet format: ds, y
    pdf = pd.DataFrame({
        "ds": pd.to_datetime(df[time_col]),
        "y": df[value_col].astype(float),
    }).dropna()

    if len(pdf) < 10:
        raise ValueError("Need at least 10 observations for Prophet")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kwargs: Dict[str, Any] = {}

        # Domain-aware holidays
        if holidays_country:
            kwargs["holidays"] = holidays_country
        elif domain_ctx:
            domain = domain_ctx.get("domain", "")
            if "retail" in domain.lower():
                kwargs.update({"holidays": "US"})

        m = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality="auto",
            daily_seasonality=False,
            **{k: v for k, v in kwargs.items() if k == "holidays"},
        )

        # Add country holidays if specified
        if "holidays" in kwargs:
            try:
                m.add_country_holidays(country_name=kwargs["holidays"])
            except Exception:
                pass

        m.fit(pdf)

    # In-sample prediction
    in_sample = m.predict(pdf[["ds"]])
    fitted = pd.Series(in_sample["yhat"].values, index=pdf.index)
    residuals = pdf["y"].values - fitted.values

    # Metrics
    metrics = _compute_metrics(
        pdf["y"].values, fitted.values,
        aic=np.nan, bic=np.nan,
    )

    # Changepoints
    changepoints = [str(cp) for cp in m.changepoints.tolist()] if hasattr(m, "changepoints") else []

    # Forecast
    fcast_df = None
    if horizon > 0:
        future = m.make_future_dataframe(periods=horizon)
        fcast_df = m.predict(future)

    # Components
    components = None
    try:
        components = in_sample[["ds", "trend", "yearly", "weekly"]].copy()
    except KeyError:
        try:
            components = in_sample[["ds", "trend"]].copy()
        except KeyError:
            pass

    return ProphetResult(
        fitted_values=fitted,
        residuals=pd.Series(residuals, index=pdf.index),
        metrics=metrics,
        model=m,
        components=components,
        changepoints=changepoints,
        forecast=fcast_df,
    )


def fit_tft(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    group_col: Optional[str] = None,
    static_covariates: Optional[List[str]] = None,
    max_epochs: int = 50,
    horizon: int = 0,
) -> TFTResult:
    """Fit Temporal Fusion Transformer.

    Falls back to Prophet if pytorch-forecasting unavailable.

    Parameters
    ----------
    df                : DataFrame
    time_col          : datetime column
    value_col         : target value column
    group_col         : panel data group column (optional)
    static_covariates : list of static feature columns
    max_epochs        : max training epochs
    horizon           : forecast horizon
    """
    if not _TFT_OK or not _TORCH_OK:
        # Fallback to Prophet
        if _PROPHET_OK:
            prophet_result = fit_prophet(df, time_col, value_col, horizon=horizon)
            return TFTResult(
                fitted_values=prophet_result.fitted_values,
                residuals=prophet_result.residuals,
                metrics={**prophet_result.metrics, "note": "Prophet fallback (TFT unavailable)"},
                model=prophet_result.model,
                attention_weights=None,
                forecast=prophet_result.forecast["yhat"].iloc[-horizon:] if prophet_result.forecast is not None and horizon > 0 else None,
            )
        # Fallback to ARIMA
        if _STATSMODELS_OK:
            arima_result = fit_arima(pd.Series(df[value_col].values), horizon=horizon)
            return TFTResult(
                fitted_values=arima_result.fitted_values,
                residuals=arima_result.residuals,
                metrics={**arima_result.metrics, "note": "ARIMA fallback (TFT unavailable)"},
                model=arima_result.model,
                attention_weights=None,
                forecast=arima_result.forecast,
            )
        raise ImportError(
            "TFT requires pytorch-forecasting + torch.\n"
            "Install with:  pip install pytorch-forecasting torch"
        )

    # Full TFT implementation
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    import pytorch_lightning as pl

    tdf = df[[time_col, value_col]].copy()
    tdf[time_col] = pd.to_datetime(tdf[time_col])
    tdf = tdf.sort_values(time_col).reset_index(drop=True)
    tdf["time_idx"] = np.arange(len(tdf))

    if group_col and group_col in df.columns:
        tdf["group"] = df[group_col].values
    else:
        tdf["group"] = "all"

    max_encoder_length = min(30, len(tdf) // 3)
    max_prediction_length = max(1, horizon if horizon > 0 else max_encoder_length // 3)

    training_cutoff = len(tdf) - max_prediction_length

    training = TimeSeriesDataSet(
        tdf[tdf.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=value_col,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=["group"]),
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=None,  # default QuantileLoss
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(tft, train_dataloaders=train_dataloader)

    # Predictions
    predictions = tft.predict(train_dataloader, return_x=True)
    pred_values = predictions.output.cpu().numpy().flatten()

    # Align fitted values
    n_fitted = min(len(pred_values), len(tdf))
    fitted = pd.Series(pred_values[:n_fitted], index=tdf.index[:n_fitted])
    actual = tdf[value_col].iloc[:n_fitted]
    residuals = actual - fitted
    metrics = _compute_metrics(actual.values, fitted.values)

    attention = None
    try:
        interpretation = tft.interpret_output(predictions.output, reduction="sum")
        attention = interpretation.get("attention", None)
    except Exception:
        pass

    return TFTResult(
        fitted_values=fitted,
        residuals=residuals,
        metrics=metrics,
        model=tft,
        attention_weights=attention,
        forecast=None,
    )


def fit_bilstm(
    series: pd.Series,
    lookback: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    max_epochs: int = 50,
    horizon: int = 0,
) -> BiLSTMResult:
    """Fit a Bidirectional LSTM for time-series forecasting.

    Falls back to ARIMA if torch unavailable.

    Parameters
    ----------
    series      : time-series values (no NaN)
    lookback    : number of past steps as input
    hidden_size : LSTM hidden dimension
    num_layers  : number of LSTM layers
    max_epochs  : training epochs
    horizon     : forecast steps
    """
    s = series.dropna().astype(float)

    if not _TORCH_OK:
        # Fallback to ARIMA
        if _STATSMODELS_OK:
            arima_result = fit_arima(s, horizon=horizon)
            return BiLSTMResult(
                fitted_values=arima_result.fitted_values,
                residuals=arima_result.residuals,
                metrics={**arima_result.metrics, "note": "ARIMA fallback (torch unavailable)"},
                model=arima_result.model,
                forecast=arima_result.forecast,
            )
        raise ImportError(
            "PyTorch is required for BiLSTM.\n"
            "Install with:  pip install torch"
        )

    values = s.values
    if len(values) < lookback + 2:
        lookback = max(3, len(values) // 3)

    # Normalise
    v_mean = float(np.mean(values))
    v_std = float(np.std(values)) or 1.0
    v_norm = (values - v_mean) / v_std

    # Create sequences
    X_list, y_list = [], []
    for i in range(lookback, len(v_norm)):
        X_list.append(v_norm[i - lookback: i])
        y_list.append(v_norm[i])

    X_arr = np.array(X_list, dtype=np.float32).reshape(-1, lookback, 1)
    y_arr = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    X_t = torch.from_numpy(X_arr)
    y_t = torch.from_numpy(y_arr)

    # Define model
    class _BiLSTM(nn.Module):
        def __init__(self, input_size: int, hidden: int, layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden, layers,
                batch_first=True, bidirectional=True,
            )
            self.fc = nn.Linear(hidden * 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = _BiLSTM(1, hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train
    model.train()
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(max_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    # Fitted values
    model.eval()
    with torch.no_grad():
        fitted_norm = model(X_t).numpy().flatten()

    fitted_vals = fitted_norm * v_std + v_mean
    fitted = pd.Series(
        np.concatenate([np.full(lookback, np.nan), fitted_vals]),
        index=s.index,
    )
    residuals = s - fitted

    metrics = _compute_metrics(
        s.values[lookback:], fitted_vals,
    )

    # Forecast
    fcast = None
    if horizon > 0:
        fcast_values = []
        last_seq = v_norm[-lookback:].copy()
        with torch.no_grad():
            for _ in range(horizon):
                inp = torch.from_numpy(
                    last_seq.reshape(1, lookback, 1).astype(np.float32)
                )
                pred = model(inp).item()
                fcast_values.append(pred * v_std + v_mean)
                last_seq = np.append(last_seq[1:], pred)

        fcast_index = pd.RangeIndex(start=len(s), stop=len(s) + horizon)
        fcast = pd.Series(fcast_values, index=fcast_index)

    return BiLSTMResult(
        fitted_values=fitted,
        residuals=residuals,
        metrics=metrics,
        model=model,
        forecast=fcast,
    )


# ── Model comparison & forecast ───────────────────────────────────────────────

def compare_models(
    results: Dict[str, object],
    primary_metric: str = "MAPE",
) -> ModelComparison:
    """Compare multiple model results and rank by primary metric.

    Parameters
    ----------
    results        : dict mapping model name -> result dataclass (ESResult, ARIMAResult, etc.)
    primary_metric : metric to rank by ("MAE", "RMSE", "MAPE", "AIC", "BIC")

    Returns
    -------
    ModelComparison with rankings DataFrame and best model.
    """
    rows = []
    for name, res in results.items():
        m = getattr(res, "metrics", {})
        rows.append({
            "model": name,
            "MAE": m.get("MAE", np.nan),
            "RMSE": m.get("RMSE", np.nan),
            "MAPE": m.get("MAPE", np.nan),
            "AIC": m.get("AIC", np.nan),
            "BIC": m.get("BIC", np.nan),
        })

    rankings = pd.DataFrame(rows)

    # Sort by primary metric (ascending — lower is better for all)
    if primary_metric in rankings.columns:
        rankings = rankings.sort_values(primary_metric, ascending=True, na_position="last")
    rankings = rankings.reset_index(drop=True)
    rankings.index = rankings.index + 1  # 1-based ranking
    rankings.index.name = "rank"

    best_name = rankings.iloc[0]["model"] if len(rankings) > 0 else ""
    best_result = results.get(best_name)

    return ModelComparison(
        rankings=rankings,
        best_model_name=best_name,
        best_model_result=best_result,
        all_results=results,
    )


def forecast(
    model_result: object,
    horizon: int,
    confidence: float = 0.95,
) -> ForecastResult:
    """Generate a forecast from any model result.

    Parameters
    ----------
    model_result : ESResult, ARIMAResult, ProphetResult, TFTResult, or BiLSTMResult
    horizon      : number of periods to forecast
    confidence   : confidence level for prediction intervals

    Returns
    -------
    ForecastResult with point forecast and confidence bounds.
    """
    alpha = 1.0 - confidence

    # ── ESResult ─────────────────────────────────────────────────────────
    if isinstance(model_result, ESResult):
        fit = model_result.model
        pred = fit.forecast(horizon)
        # ES doesn't natively produce PIs — use residual std
        res_std = float(model_result.residuals.std()) or 1.0
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        steps = np.arange(1, horizon + 1)
        width = z * res_std * np.sqrt(steps)
        lower = pred - width
        upper = pred + width
        return ForecastResult(
            point_forecast=pred,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=confidence,
            horizon=horizon,
            model_name="ExponentialSmoothing",
        )

    # ── ARIMAResult ──────────────────────────────────────────────────────
    if isinstance(model_result, ARIMAResult):
        fit = model_result.model
        pred = fit.get_forecast(steps=horizon)
        ci = pred.conf_int(alpha=alpha)
        return ForecastResult(
            point_forecast=pred.predicted_mean,
            lower_bound=ci.iloc[:, 0],
            upper_bound=ci.iloc[:, 1],
            confidence_level=confidence,
            horizon=horizon,
            model_name=f"ARIMA{model_result.order}",
        )

    # ── ProphetResult ────────────────────────────────────────────────────
    if isinstance(model_result, ProphetResult):
        m = model_result.model
        future = m.make_future_dataframe(periods=horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fcast = m.predict(future)
        tail = fcast.tail(horizon)
        return ForecastResult(
            point_forecast=pd.Series(tail["yhat"].values),
            lower_bound=pd.Series(tail[f"yhat_lower"].values),
            upper_bound=pd.Series(tail[f"yhat_upper"].values),
            confidence_level=confidence,
            horizon=horizon,
            model_name="Prophet",
        )

    # ── TFTResult / BiLSTMResult — use pre-computed forecast if available
    for cls, name in [(TFTResult, "TFT"), (BiLSTMResult, "BiLSTM")]:
        if isinstance(model_result, cls):
            fcast = model_result.forecast
            if fcast is not None and len(fcast) >= horizon:
                point = fcast.iloc[:horizon]
            else:
                point = pd.Series([np.nan] * horizon)
            # No native CI for DL models — use residual std
            res_std = float(model_result.residuals.dropna().std()) or 1.0
            try:
                from scipy.stats import norm
                z = norm.ppf(1 - alpha / 2)
            except ImportError:
                z = 1.96
            steps = np.arange(1, horizon + 1)
            width = z * res_std * np.sqrt(steps)
            return ForecastResult(
                point_forecast=point,
                lower_bound=point - width,
                upper_bound=point + width,
                confidence_level=confidence,
                horizon=horizon,
                model_name=name,
            )

    raise TypeError(f"Unsupported model result type: {type(model_result)}")


def timeseries_summary(
    ts_config: object,
    comparison: Optional[ModelComparison] = None,
    forecast_result: Optional[ForecastResult] = None,
    domain_ctx: Optional[dict] = None,
) -> str:
    """Generate an ASCII-safe narrative summary.

    Parameters
    ----------
    ts_config       : TimeSeriesConfig from detect_time_index
    comparison      : ModelComparison from compare_models
    forecast_result : ForecastResult from forecast()
    domain_ctx      : domain context dict

    Returns
    -------
    ASCII-safe plain-text summary string.
    """
    lines: List[str] = ["=== TIME SERIES ANALYSIS SUMMARY ===", ""]

    # Time index info
    cfg = ts_config
    lines.append(f"Time column : {cfg.time_col}")
    lines.append(f"Frequency   : {cfg.freq or 'irregular'} (confidence: {cfg.freq_confidence})")
    lines.append(f"Observations: {cfg.n_observations}")
    lines.append(f"Date range  : {cfg.date_range[0]} to {cfg.date_range[1]}")
    lines.append(f"Gaps        : {cfg.n_gaps} ({cfg.gap_pct}%)")
    lines.append(f"Regular     : {'Yes' if cfg.is_regular else 'No'}")
    if cfg.warnings:
        lines.append(f"Warnings    : {'; '.join(cfg.warnings)}")
    lines.append("")

    # Domain context
    if domain_ctx:
        lines.append(f"Domain      : {domain_ctx.get('domain', 'Unknown')}")
        lines.append("")

    # Model comparison
    if comparison is not None:
        lines.append("--- Model Rankings ---")
        for _, row in comparison.rankings.iterrows():
            lines.append(
                f"  {row['model']:25s}  MAE={row['MAE']:<10}  "
                f"RMSE={row['RMSE']:<10}  MAPE={row['MAPE']}%"
            )
        lines.append(f"\nBest model: {comparison.best_model_name}")
        lines.append("")

    # Forecast
    if forecast_result is not None:
        lines.append("--- Forecast ---")
        lines.append(f"Model    : {forecast_result.model_name}")
        lines.append(f"Horizon  : {forecast_result.horizon} periods")
        lines.append(f"Confidence: {forecast_result.confidence_level * 100:.0f}%")

        point = forecast_result.point_forecast
        if len(point) > 0:
            lines.append(f"Range    : {float(point.iloc[0]):.2f} to {float(point.iloc[-1]):.2f}")
            lower = forecast_result.lower_bound
            upper = forecast_result.upper_bound
            avg_width = float((upper - lower).mean())
            lines.append(f"Avg CI width: {avg_width:.2f}")
        lines.append("")

    lines.append("=" * 40)

    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")


# ── Private helpers ───────────────────────────────────────────────────────────

def _auto_arima_order(
    series: pd.Series, d: int, max_p: int = 5, max_q: int = 5,
) -> Tuple[int, int, int]:
    """AIC grid search for best (p, d, q)."""
    best_aic = np.inf
    best_order = (1, d, 1)

    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        series, order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False, maxiter=100)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
            except Exception:
                continue

    return best_order
