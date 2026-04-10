"""Full model battery: ES, ARIMA, Prophet, TFT, BiLSTM on aggregated respiratory admissions."""
import sys
import os
import warnings

sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timezone

OUT = r"C:\Users\Ravi\research\data-analysis\output\air_quality_ts"
os.makedirs(OUT, exist_ok=True)

# ── Load & aggregate ─────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Ravi\Downloads\air_quality_health_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

daily = df.groupby("date")["respiratory_admissions"].sum().reset_index()
daily = daily.sort_values("date").reset_index(drop=True)
print(f"Aggregated: {len(daily)} days, range {daily['date'].min().date()} to {daily['date'].max().date()}")
print(f"Mean={daily['respiratory_admissions'].mean():.1f}, Std={daily['respiratory_admissions'].std():.1f}")

# ── Preprocessing ────────────────────────────────────────────────────────
from hooks.timeseries_prep import detect_time_index, detect_seasonality, ts_outlier_treatment

ts_config = detect_time_index(daily)
print(f"\nTime col: {ts_config.time_col}, Freq: {ts_config.freq}")

series = daily["respiratory_admissions"].copy().astype(float)
time_index = pd.DatetimeIndex(daily["date"])

# Outlier treatment
cleaned, flags = ts_outlier_treatment(series, method="stl_residual", threshold=3.0)
print(f"Outliers treated: {flags.sum()}")
series = cleaned

# Seasonality
seasonality = detect_seasonality(series.dropna(), ts_config.freq)
print(f"Seasonality: period={seasonality.period}, strength={seasonality.strength:.4f}, type={seasonality.seasonal_type}")

clean = series.dropna()

# ── Train/test split for fair comparison ─────────────────────────────────
HORIZON = 15
train_series = clean.iloc[:-HORIZON]
test_series = clean.iloc[-HORIZON:]
train_df = daily.iloc[:-HORIZON].copy()
test_df = daily.iloc[-HORIZON:].copy()

print(f"\nTrain: {len(train_series)} days, Test: {len(test_series)} days")
print(f"Test actual: {test_series.values}")

# ── Fit all models on TRAIN, evaluate on TEST ────────────────────────────
from hooks.timeseries_models import (
    fit_exponential_smoothing, fit_arima, fit_prophet,
    fit_tft, fit_bilstm,
    compare_models, forecast, timeseries_summary,
)

results = {}

# 1. Exponential Smoothing
print("\n--- Exponential Smoothing ---")
sp = seasonality.period if seasonality.period > 1 and len(train_series) >= 2 * seasonality.period else None
try:
    es = fit_exponential_smoothing(train_series, seasonal_periods=sp, horizon=HORIZON)
    es_fcast = es.forecast
    results["ExponentialSmoothing"] = es
    print(f"  In-sample MAPE: {es.metrics['MAPE']}%")
    if es_fcast is not None:
        es_pred = es_fcast.values[:HORIZON]
        es_mae = np.mean(np.abs(test_series.values - es_pred))
        es_rmse = np.sqrt(np.mean((test_series.values - es_pred) ** 2))
        nz = np.abs(test_series.values) > 1e-10
        es_mape = np.mean(np.abs((test_series.values[nz] - es_pred[nz]) / test_series.values[nz])) * 100
        print(f"  Holdout MAE={es_mae:.2f}, RMSE={es_rmse:.2f}, MAPE={es_mape:.1f}%")
except Exception as exc:
    print(f"  FAILED: {exc}")

# 2. ARIMA
print("\n--- ARIMA ---")
try:
    arima = fit_arima(train_series, horizon=HORIZON)
    results["ARIMA"] = arima
    print(f"  Order: {arima.order}, Stationary: {arima.is_stationary}, ADF p={arima.adf_pvalue}")
    print(f"  In-sample MAPE: {arima.metrics['MAPE']}%")
    if arima.forecast is not None:
        ar_pred = arima.forecast.values[:HORIZON]
        ar_mae = np.mean(np.abs(test_series.values - ar_pred))
        ar_rmse = np.sqrt(np.mean((test_series.values - ar_pred) ** 2))
        nz = np.abs(test_series.values) > 1e-10
        ar_mape = np.mean(np.abs((test_series.values[nz] - ar_pred[nz]) / test_series.values[nz])) * 100
        print(f"  Holdout MAE={ar_mae:.2f}, RMSE={ar_rmse:.2f}, MAPE={ar_mape:.1f}%")
except Exception as exc:
    print(f"  FAILED: {exc}")

# 3. Prophet
print("\n--- Prophet ---")
try:
    prophet = fit_prophet(train_df, "date", "respiratory_admissions", horizon=HORIZON)
    results["Prophet"] = prophet
    print(f"  In-sample MAPE: {prophet.metrics['MAPE']}%")
    print(f"  Changepoints: {len(prophet.changepoints)}")
    if prophet.forecast is not None:
        # Prophet forecast includes in-sample + future
        pr_fcast = prophet.forecast.tail(HORIZON)
        pr_pred = pr_fcast["yhat"].values
        pr_mae = np.mean(np.abs(test_series.values - pr_pred))
        pr_rmse = np.sqrt(np.mean((test_series.values - pr_pred) ** 2))
        nz = np.abs(test_series.values) > 1e-10
        pr_mape = np.mean(np.abs((test_series.values[nz] - pr_pred[nz]) / test_series.values[nz])) * 100
        print(f"  Holdout MAE={pr_mae:.2f}, RMSE={pr_rmse:.2f}, MAPE={pr_mape:.1f}%")
except Exception as exc:
    print(f"  FAILED: {exc}")

# 4. BiLSTM
print("\n--- BiLSTM ---")
try:
    bilstm = fit_bilstm(train_series, lookback=30, hidden_size=64, num_layers=2,
                         max_epochs=100, horizon=HORIZON)
    results["BiLSTM"] = bilstm
    print(f"  In-sample MAPE: {bilstm.metrics.get('MAPE', 'N/A')}%")
    if bilstm.forecast is not None:
        bl_pred = bilstm.forecast.values[:HORIZON]
        bl_mae = np.mean(np.abs(test_series.values - bl_pred))
        bl_rmse = np.sqrt(np.mean((test_series.values - bl_pred) ** 2))
        nz = np.abs(test_series.values) > 1e-10
        bl_mape = np.mean(np.abs((test_series.values[nz] - bl_pred[nz]) / test_series.values[nz])) * 100
        print(f"  Holdout MAE={bl_mae:.2f}, RMSE={bl_rmse:.2f}, MAPE={bl_mape:.1f}%")
    else:
        print(f"  No forecast produced")
except Exception as exc:
    print(f"  FAILED: {exc}")

# 5. TFT
print("\n--- Temporal Fusion Transformer ---")
try:
    tft = fit_tft(train_df, "date", "respiratory_admissions",
                  max_epochs=30, horizon=HORIZON)
    results["TFT"] = tft
    note = tft.metrics.get("note", "")
    if note:
        print(f"  Note: {note}")
    print(f"  In-sample MAPE: {tft.metrics.get('MAPE', 'N/A')}%")
    if tft.forecast is not None:
        tf_pred = tft.forecast.values[:HORIZON] if len(tft.forecast) >= HORIZON else tft.forecast.values
        if len(tf_pred) == HORIZON:
            tf_mae = np.mean(np.abs(test_series.values - tf_pred))
            tf_rmse = np.sqrt(np.mean((test_series.values - tf_pred) ** 2))
            nz = np.abs(test_series.values) > 1e-10
            tf_mape = np.mean(np.abs((test_series.values[nz] - tf_pred[nz]) / test_series.values[nz])) * 100
            print(f"  Holdout MAE={tf_mae:.2f}, RMSE={tf_rmse:.2f}, MAPE={tf_mape:.1f}%")
        else:
            print(f"  Forecast length mismatch: got {len(tf_pred)}, need {HORIZON}")
    else:
        print(f"  No separate forecast produced")
except Exception as exc:
    print(f"  FAILED: {exc}")

# ── Compare all models (in-sample) ───────────────────────────────────────
print("\n" + "=" * 70)
comparison = compare_models(results, primary_metric="MAPE")
print("=== In-Sample Model Rankings ===")
print(comparison.rankings[["model", "MAE", "RMSE", "MAPE"]].to_string())
print(f"\nBest (in-sample): {comparison.best_model_name}")

# ── Holdout comparison table ─────────────────────────────────────────────
print("\n=== Holdout (15-day) Model Rankings ===")
holdout_rows = []
for name, res in results.items():
    fcast_vals = None
    if name == "ExponentialSmoothing" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON]
    elif name == "ARIMA" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON]
    elif name == "Prophet" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.tail(HORIZON)["yhat"].values
    elif name == "BiLSTM" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON]
    elif name == "TFT" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON] if len(res.forecast) >= HORIZON else None

    if fcast_vals is not None and len(fcast_vals) == HORIZON:
        actual = test_series.values
        mae = np.mean(np.abs(actual - fcast_vals))
        rmse = np.sqrt(np.mean((actual - fcast_vals) ** 2))
        nz = np.abs(actual) > 1e-10
        mape = np.mean(np.abs((actual[nz] - fcast_vals[nz]) / actual[nz])) * 100
        holdout_rows.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 1)})
    else:
        holdout_rows.append({"model": name, "MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"})

holdout_df = pd.DataFrame(holdout_rows)
# Sort by MAPE where available
holdout_df["_sort"] = holdout_df["MAPE"].apply(lambda x: float(x) if isinstance(x, (int, float)) else 999)
holdout_df = holdout_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
holdout_df.index = holdout_df.index + 1
holdout_df.index.name = "rank"
print(holdout_df.to_string())

# ── Best model 15-day production forecast (retrained on full data) ───────
print("\n=== Production Forecast (best model retrained on full data) ===")
best_name = comparison.best_model_name

# Retrain best on full data
if best_name == "ExponentialSmoothing":
    final = fit_exponential_smoothing(clean, seasonal_periods=sp, horizon=HORIZON)
    fcast = forecast(final, horizon=HORIZON)
elif best_name == "ARIMA":
    final = fit_arima(clean, horizon=HORIZON)
    fcast = forecast(final, horizon=HORIZON)
elif best_name == "Prophet":
    final = fit_prophet(daily, "date", "respiratory_admissions", horizon=HORIZON)
    fcast = forecast(final, horizon=HORIZON)
elif best_name == "BiLSTM":
    final = fit_bilstm(clean, lookback=30, hidden_size=64, num_layers=2,
                        max_epochs=100, horizon=HORIZON)
    fcast = forecast(final, horizon=HORIZON)
else:
    fcast = forecast(comparison.best_model_result, horizon=HORIZON)

print(f"Model: {fcast.model_name}, Horizon: {fcast.horizon}, Confidence: {fcast.confidence_level*100:.0f}%")
for i in range(HORIZON):
    lo = float(fcast.lower_bound.iloc[i])
    pt = float(fcast.point_forecast.iloc[i])
    hi = float(fcast.upper_bound.iloc[i])
    print(f"  Day {i+1:2d}: {pt:6.1f}  [{lo:6.1f} -- {hi:6.1f}]")

# ── Plots ────────────────────────────────────────────────────────────────
from hooks.timeseries_viz import (
    plot_forecast, plot_model_comparison, plot_residual_diagnostics,
)

p1 = plot_forecast(clean, fcast, OUT, f"All-Models Best ({best_name}) -- 15-Day Forecast")
p2 = plot_model_comparison(comparison, OUT)
print(f"\nPlots: {os.path.basename(p1)}, {os.path.basename(p2)}")

# ── Holdout overlay plot ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 6))

# Last 60 days of actuals
tail = daily.tail(60 + HORIZON)
train_tail = tail.iloc[:-HORIZON]
test_tail = tail.iloc[-HORIZON:]

ax.plot(train_tail["date"], train_tail["respiratory_admissions"],
        color="#4C72B0", lw=1.2, label="Train (last 60d)")
ax.plot(test_tail["date"], test_tail["respiratory_admissions"],
        color="black", lw=2, marker="o", markersize=4, label="Actual (holdout)")

colors = {"ExponentialSmoothing": "#DD8452", "ARIMA": "#55A868",
          "Prophet": "#C44E52", "BiLSTM": "#8172B2", "TFT": "#CCB974"}

for name, res in results.items():
    fcast_vals = None
    if name == "ExponentialSmoothing" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON]
    elif name == "ARIMA" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON]
    elif name == "Prophet" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.tail(HORIZON)["yhat"].values
    elif name == "BiLSTM" and hasattr(res, "forecast") and res.forecast is not None:
        fcast_vals = res.forecast.values[:HORIZON]
    elif name == "TFT" and hasattr(res, "forecast") and res.forecast is not None:
        fv = res.forecast.values[:HORIZON] if len(res.forecast) >= HORIZON else None
        if fv is not None and len(fv) == HORIZON:
            fcast_vals = fv

    if fcast_vals is not None and len(fcast_vals) == HORIZON:
        c = colors.get(name, "gray")
        ax.plot(test_tail["date"], fcast_vals, color=c, lw=1.5,
                linestyle="--", marker="s", markersize=3, label=name)

ax.set_title("All Models vs Holdout -- 15-Day Respiratory Admissions Forecast")
ax.set_ylabel("Respiratory Admissions (total)")
ax.set_xlabel("Date")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()

ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]
p3 = os.path.join(OUT, f"all_models_holdout_{ts}.png")
fig.savefig(p3, dpi=150)
plt.close(fig)
print(f"Holdout overlay: {os.path.basename(p3)}")
