"""Panel SARIMAX forecast: per-region with air quality covariates."""
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

OUT = r"C:\Users\Ravi\research\data-analysis\output\air_quality_ts"
os.makedirs(OUT, exist_ok=True)

# ── Load & prepare ───────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Ravi\Downloads\air_quality_health_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

EXOG_COLS = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "temperature", "humidity", "wind_speed", "precipitation",
    "hospital_visits", "emergency_visits", "mobility_index",
    "industrial_activity", "vehicle_count",
]
TARGET = "respiratory_admissions"

# Aggregate duplicates per region-date
agg_dict = {TARGET: "sum"}
for c in EXOG_COLS:
    agg_dict[c] = "mean"

regions = sorted(df["region"].unique())
print(f"Regions: {regions}")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

all_region_results = {}
panel_forecasts = {}

for region in regions:
    rdf = df[df["region"] == region].copy()

    # Aggregate duplicates per date
    rdf = rdf.groupby("date").agg(agg_dict).reset_index()
    rdf = rdf.sort_values("date").reset_index(drop=True)

    # Build complete daily grid
    full_dates = pd.date_range(rdf["date"].min(), rdf["date"].max(), freq="D")
    rdf = rdf.set_index("date").reindex(full_dates)
    rdf.index.name = "date"
    rdf = rdf.interpolate(method="linear").ffill().bfill()

    n = len(rdf)
    y = rdf[TARGET]
    X = rdf[EXOG_COLS]

    # Split: train = all but last 15, test = last 15
    train_y = y.iloc[:-15]
    train_X = X.iloc[:-15]
    test_y = y.iloc[-15:]
    test_X = X.iloc[-15:]

    # ADF test
    adf_p = adfuller(train_y.dropna(), autolag="AIC")[1]
    d = 0 if adf_p < 0.05 else 1

    # Grid search SARIMAX
    best_aic = np.inf
    best_order = (1, d, 1)
    for p in range(0, 3):
        for q in range(0, 3):
            if p == 0 and q == 0:
                continue
            try:
                m = SARIMAX(
                    train_y, exog=train_X, order=(p, d, q),
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                fit = m.fit(disp=False, maxiter=100)
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    # Refit best on training data
    model = SARIMAX(
        train_y, exog=train_X, order=best_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)

    # Validate
    pred = fit.get_forecast(steps=15, exog=test_X)
    pred_mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    actual = test_y.values
    predicted = pred_mean.values
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    nonzero = np.abs(actual) > 1e-10
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100

    print(
        f"{region:8s} SARIMAX{best_order} -- n={n}, "
        f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, ADF_p={adf_p:.4f}"
    )

    all_region_results[region] = {
        "order": best_order,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "test_actual": actual,
        "test_predicted": predicted,
    }

    # Full-data refit + 15-day forecast
    full_model = SARIMAX(
        y, exog=X, order=best_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    full_fit = full_model.fit(disp=False, maxiter=200)

    # Future exog: persistence (repeat last known values)
    future_exog = pd.DataFrame(
        [X.iloc[-1].values] * 15,
        columns=EXOG_COLS,
    )
    future_pred = full_fit.get_forecast(steps=15, exog=future_exog)
    future_mean = future_pred.predicted_mean
    future_ci = future_pred.conf_int(alpha=0.05)

    panel_forecasts[region] = {
        "point": future_mean.values,
        "lower": future_ci.iloc[:, 0].values,
        "upper": future_ci.iloc[:, 1].values,
    }


# ── Aggregated panel forecast ────────────────────────────────────────────
print("\n=== Aggregated Panel SARIMAX Forecast (15 days, all regions) ===")
agg_point = np.zeros(15)
agg_lower = np.zeros(15)
agg_upper = np.zeros(15)
for region in regions:
    agg_point += panel_forecasts[region]["point"]
    agg_lower += panel_forecasts[region]["lower"]
    agg_upper += panel_forecasts[region]["upper"]

for i in range(15):
    print(f"  Day {i + 1:2d}: {agg_point[i]:6.1f}  [{agg_lower[i]:6.1f} - {agg_upper[i]:6.1f}]")

# ── Validation summary ───────────────────────────────────────────────────
avg_mae = np.mean([r["mae"] for r in all_region_results.values()])
avg_rmse = np.mean([r["rmse"] for r in all_region_results.values()])
avg_mape = np.mean([r["mape"] for r in all_region_results.values()])
print(
    f"\nPanel validation (avg across regions): "
    f"MAE={avg_mae:.2f}, RMSE={avg_rmse:.2f}, MAPE={avg_mape:.1f}%"
)

# ── Per-region forecast ──────────────────────────────────────────────────
print("\n=== Per-Region 15-Day Forecast ===")
for region in regions:
    pf = panel_forecasts[region]
    order_str = str(all_region_results[region]["order"])
    print(f"\n  {region} (SARIMAX{order_str}):")
    for i in range(15):
        print(
            f"    Day {i + 1:2d}: {pf['point'][i]:5.1f}  "
            f"[{pf['lower'][i]:5.1f} - {pf['upper'][i]:5.1f}]"
        )


# ── Plot ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

last_date = df["date"].max()
fcast_dates = pd.date_range(last_date + timedelta(days=1), periods=15, freq="D")

# Aggregate historical
daily = df.groupby("date")[TARGET].sum().sort_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel SARIMAX
ax = axes[0]
ax.plot(daily.index[-60:], daily.values[-60:], color="#4C72B0", lw=1.2, label="Actual (last 60d)")
ax.plot(fcast_dates, agg_point, color="#DD8452", lw=2, label="Panel SARIMAX Forecast")
ax.fill_between(fcast_dates, agg_lower, agg_upper, alpha=0.2, color="#DD8452", label="95% CI")
ax.set_title("Panel SARIMAX (per-region with AQ covariates) -- 15-Day Forecast")
ax.set_ylabel("Total Respiratory Admissions")
ax.legend()
ax.grid(alpha=0.3)

# Per-region breakdown
ax2 = axes[1]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
for i, region in enumerate(regions):
    pf = panel_forecasts[region]
    ax2.plot(
        fcast_dates, pf["point"], color=colors[i], lw=1.5,
        label=region, marker="o", markersize=3,
    )
    ax2.fill_between(fcast_dates, pf["lower"], pf["upper"], alpha=0.1, color=colors[i])
ax2.set_title("Per-Region Forecast Breakdown")
ax2.set_ylabel("Respiratory Admissions")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]
p = os.path.join(OUT, f"panel_sarimax_forecast_{ts}.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"\nPanel forecast plot: {os.path.basename(p)}")


# ── Comparison ───────────────────────────────────────────────────────────
print("\n=== COMPARISON: Pure TS vs Panel SARIMAX ===")
print(f"                     Pure TS (ES)    Panel SARIMAX")
print(f"  Avg Point (Day 1):   38.4            {agg_point[0]:.1f}")
print(f"  Avg Point (Day 15):  38.2            {agg_point[14]:.1f}")
print(f"  Panel Validation MAE:  --              {avg_mae:.2f}")
print(f"  Panel Validation MAPE: --              {avg_mape:.1f}%")
print(f"  Pure TS MAPE (in-sample): 13.6%")
print(f"  Covariates used: {len(EXOG_COLS)}")
print(f"  Covariate list: {', '.join(EXOG_COLS)}")
