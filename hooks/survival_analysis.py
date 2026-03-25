"""
survival_analysis.py -- Survival and churn analysis hook.

Four modelling strategies:
  1. Non-parametric  — Kaplan-Meier (overall + stratified; log-rank test)
  2. Parametric      — Weibull + LogNormal fit with AIC comparison
  3. Semi-parametric — Cox Proportional Hazards (hazard ratios, C-index)
  4. Prediction      — survival probability at chosen time points for a new entity

Also handles:
  - Data preparation  (auto-detect duration/event columns, encode categoricals,
                       flag negative durations, impute missing covariates)
  - Plotting          (KM curves, model comparison overlay, Cox forest plot —
                       all saved as PNG to an output directory)
  - Plain-English summary

Requires:
    pip install lifelines

Public API
----------
prepare_survival_data(df, duration_col, event_col,
                      covariate_cols, categorical_cols)  -> SurvivalDataset

fit_kaplan_meier(dataset, stratify_col, time_points)     -> KMResult
fit_parametric(dataset)                                  -> ParametricResult
fit_cox(dataset, covariates, penalizer)                  -> CoxResult

predict_survival(dataset, time_points, km_result,
                 param_result, cox_result, new_entity)   -> dict

plot_km(km_result, out_dir, title)                       -> list[str]
plot_model_comparison(km_result, param_result, out_dir)  -> str
plot_cox(cox_result, out_dir)                            -> list[str]

predict_batch(dataset, cox_result, new_df,
              time_horizons, categorical_cols,
              id_cols, risk_tier_method)              -> pd.DataFrame

survival_summary(dataset, km_result, param_result,
                 cox_result, domain_ctx)                 -> str
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── lifelines (graceful import error) ────────────────────────────────────────

try:
    from lifelines import (
        KaplanMeierFitter,
        WeibullFitter,
        LogNormalFitter,
        CoxPHFitter,
    )
    from lifelines.statistics import logrank_test
    _LIFELINES_OK = True
except ImportError:
    _LIFELINES_OK = False


def _require_lifelines() -> None:
    if not _LIFELINES_OK:
        raise ImportError(
            "lifelines is required for survival analysis.\n"
            "Install with:  pip install lifelines"
        )


# ── Column name auto-detection ────────────────────────────────────────────────

_DURATION_HINTS = (
    "duration", "time", "tenure", "days", "months", "weeks",
    "period", "lifetime", "survival", "churn_time", "t_", "age",
)
_EVENT_HINTS = (
    "event", "churn", "died", "dead", "failure", "observed",
    "status", "outcome", "converted", "attrition", "churned",
)


def _auto_detect_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Infer duration and event column names from column names and dtypes."""
    duration_col: Optional[str] = None
    event_col: Optional[str] = None
    for col in df.columns:
        cl = col.lower().replace(" ", "_")
        if duration_col is None and any(h in cl for h in _DURATION_HINTS):
            if pd.api.types.is_numeric_dtype(df[col]):
                duration_col = col
        if event_col is None and any(h in cl for h in _EVENT_HINTS):
            # event should be binary or very low cardinality
            if df[col].nunique() <= 5:
                event_col = col
    return duration_col, event_col


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclasses.dataclass
class SurvivalDataset:
    """Validated and prepared dataset ready for survival modelling."""
    df: pd.DataFrame
    duration_col: str
    event_col: str
    covariate_cols: List[str]       # numeric columns ready for Cox (incl. encoded dummies)
    categorical_cols: List[str]     # original categorical column names
    n_total: int
    n_events: int
    n_censored: int
    warnings: List[str]
    median_duration: float


@dataclasses.dataclass
class KMResult:
    """Kaplan-Meier fit."""
    median_survival: float
    survival_at_times: Dict[float, float]   # time -> P(T > t)
    overall_kmf: object                     # KaplanMeierFitter
    group_kmfs: Dict[str, object]           # group label -> KaplanMeierFitter
    stratify_col: Optional[str]
    logrank_pvalue: Optional[float]
    n_total: int
    n_events: int


@dataclasses.dataclass
class ParametricResult:
    """Weibull and LogNormal parametric fit + AIC comparison."""
    weibull_aic: float
    lognormal_aic: float
    better_model: str               # 'Weibull' or 'LogNormal'
    weibull_lambda: float
    weibull_rho: float
    lognormal_mu: float
    lognormal_sigma: float
    weibull_fitter: object          # WeibullFitter
    lognormal_fitter: object        # LogNormalFitter
    median_weibull: float
    median_lognormal: float


@dataclasses.dataclass
class CoxResult:
    """Cox PH fit."""
    summary_df: pd.DataFrame        # covariate, hazard_ratio, hr_lower_95, hr_upper_95, p_value, z_score
    concordance: float              # C-index
    covariates_used: List[str]
    cph_fitter: object              # CoxPHFitter
    cox_df: pd.DataFrame            # data used to fit (duration + event + covariates)
    significant_covariates: List[str]   # p < 0.05


# ── 1. Data preparation ───────────────────────────────────────────────────────

def prepare_survival_data(
    df: pd.DataFrame,
    duration_col: Optional[str] = None,
    event_col: Optional[str] = None,
    covariate_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    start_date_col: Optional[str] = None,
    end_date_col: Optional[str] = None,
    analysis_date: Optional[pd.Timestamp] = None,
) -> SurvivalDataset:
    """Validate and prepare a DataFrame for survival analysis.

    Auto-detects duration and event columns when not specified.
    Encodes categorical covariates as one-hot dummies.
    Removes rows with zero/negative durations (these are impossible event times).
    Imputes missing covariate values with column medians.

    Parameters
    ----------
    df               : input DataFrame
    duration_col     : time-to-event or time-to-censoring column (numeric).
                       If omitted, computed from start_date_col / end_date_col,
                       or auto-detected by column name hints.
    event_col        : binary event indicator: 1 = event occurred, 0 = censored
    covariate_cols   : numeric columns to include as Cox covariates
    categorical_cols : categorical columns to one-hot encode for Cox
    start_date_col   : date column marking the start of the observation window
                       (e.g. "Membership Start Date"). Used to compute duration
                       when duration_col is not already in df.
    end_date_col     : date column for the end of the observation window
                       (e.g. "Membership Termination Date"). Rows where this is
                       null are treated as censored; their end date is set to
                       analysis_date.
    analysis_date    : reference date for censored observations when end_date_col
                       is null. Defaults to today if not provided.
    """
    df = df.copy()
    issues: List[str] = []

    # ── Compute duration from raw date columns if requested ───────────────────
    if start_date_col and start_date_col in df.columns and duration_col is None:
        _ad = analysis_date or pd.Timestamp("today").normalize()
        df[start_date_col] = pd.to_datetime(
            df[start_date_col], format="mixed", dayfirst=False, errors="coerce")
        if end_date_col and end_date_col in df.columns:
            df[end_date_col] = pd.to_datetime(
                df[end_date_col], format="mixed", dayfirst=False, errors="coerce")
            end_dates = df[end_date_col].fillna(_ad)
        else:
            end_dates = pd.Series([_ad] * len(df), index=df.index)
        df["_duration_days"] = (end_dates - df[start_date_col]).dt.days
        duration_col = "_duration_days"
        issues.append(
            f"Computed duration_days from '{start_date_col}' -> "
            f"'{end_date_col or 'analysis_date'}' "
            f"(analysis_date={_ad.date()})."
        )

    # ── Auto-detect columns ───────────────────────────────────────────────────
    if duration_col is None or event_col is None:
        auto_dur, auto_evt = _auto_detect_cols(df)
        if duration_col is None:
            if auto_dur is None:
                raise ValueError(
                    "Could not auto-detect a duration column. "
                    "Specify duration_col= explicitly (e.g. 'tenure_days', 'months_active')."
                )
            duration_col = auto_dur
            issues.append(f"Auto-detected duration column: '{duration_col}'")
        if event_col is None:
            if auto_evt is None:
                raise ValueError(
                    "Could not auto-detect an event column. "
                    "Specify event_col= explicitly (1 = event/churn occurred, 0 = censored)."
                )
            event_col = auto_evt
            issues.append(f"Auto-detected event column: '{event_col}'")

    # ── Coerce to numeric ─────────────────────────────────────────────────────
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    # ── Drop null durations ───────────────────────────────────────────────────
    null_dur = int(df[duration_col].isna().sum())
    if null_dur > 0:
        issues.append(f"Dropped {null_dur} rows with null duration.")
        df = df.dropna(subset=[duration_col])

    # ── Drop zero/negative durations (non-positive time is impossible) ────────
    neg_dur = int((df[duration_col] <= 0).sum())
    if neg_dur > 0:
        issues.append(
            f"Dropped {neg_dur} rows with duration <= 0 "
            f"(values ranged to {df.loc[df[duration_col] <= 0, duration_col].min():.2f}). "
            "Survival analysis requires strictly positive durations."
        )
        df = df[df[duration_col] > 0]

    # ── Validate event column ─────────────────────────────────────────────────
    unique_events = sorted(df[event_col].dropna().unique())
    non_binary = [v for v in unique_events if v not in (0, 1, 0.0, 1.0)]
    if non_binary:
        issues.append(
            f"Event column '{event_col}' contains non-binary values {non_binary[:5]}. "
            "Coercing: any value > 0 treated as 1 (event), 0 stays 0 (censored)."
        )
        df[event_col] = (df[event_col].fillna(0) > 0).astype(int)
    else:
        df[event_col] = df[event_col].fillna(0).astype(int)

    n_events = int(df[event_col].sum())
    if n_events == 0:
        issues.append(
            "WARNING: All observations are censored (event_col is all zeros). "
            "KM and Cox results will be unreliable — verify the event_col definition."
        )

    # ── Build covariate list ──────────────────────────────────────────────────
    categorical_cols = list(categorical_cols or [])
    exclude = {duration_col, event_col}

    if covariate_cols is None:
        # Auto-include all numeric columns (excluding duration and event)
        covariate_cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and not c.endswith(("_flag", "_score", "_detected_anomaly"))
        ]
    else:
        covariate_cols = [c for c in covariate_cols if c in df.columns and c not in exclude]

    # ── One-hot encode categorical covariates ─────────────────────────────────
    encoded_cols: List[str] = []
    for cat in categorical_cols:
        if cat not in df.columns:
            issues.append(f"Categorical column '{cat}' not found — skipped.")
            continue
        dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        encoded_cols.extend(dummies.columns.tolist())
        issues.append(f"One-hot encoded '{cat}' -> {dummies.columns.tolist()}")

    all_covariates = covariate_cols + encoded_cols

    # ── Impute missing covariate values with column median ────────────────────
    for col in all_covariates:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            med = float(df[col].median())
            df[col] = df[col].fillna(med)
            issues.append(f"Imputed {n_null} nulls in '{col}' with median {med:.4f}.")

    df = df.reset_index(drop=True)
    n_total = len(df)

    return SurvivalDataset(
        df=df,
        duration_col=duration_col,
        event_col=event_col,
        covariate_cols=all_covariates,
        categorical_cols=categorical_cols,
        n_total=n_total,
        n_events=n_events,
        n_censored=n_total - n_events,
        warnings=issues,
        median_duration=float(df[duration_col].median()),
    )


# ── 2. Kaplan-Meier ───────────────────────────────────────────────────────────

def fit_kaplan_meier(
    dataset: SurvivalDataset,
    stratify_col: Optional[str] = None,
    time_points: Optional[List[float]] = None,
) -> KMResult:
    """Fit a Kaplan-Meier survival curve, optionally stratified.

    If stratify_col has exactly two groups, a log-rank test p-value is computed.

    Parameters
    ----------
    dataset      : prepared SurvivalDataset
    stratify_col : column to split by (original categorical or any column in df)
    time_points  : time values at which to evaluate S(t); defaults to
                   [Q25, median, Q75, max] of the observed durations
    """
    _require_lifelines()
    df = dataset.df
    dur = dataset.duration_col
    evt = dataset.event_col

    # Default evaluation time points = duration quartiles
    if time_points is None:
        q = df[dur].quantile([0.25, 0.5, 0.75, 1.0]).values
        time_points = [float(v) for v in q]

    # ── Overall KM ────────────────────────────────────────────────────────────
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[dur], event_observed=df[evt], label="Overall")
    median_surv = float(kmf.median_survival_time_)
    survival_at = {t: float(kmf.predict(t)) for t in time_points}

    # ── Stratified KM ─────────────────────────────────────────────────────────
    group_kmfs: Dict[str, object] = {}
    logrank_pval: Optional[float] = None

    if stratify_col and stratify_col in df.columns:
        groups = df[stratify_col].dropna().unique()
        for g in sorted(groups, key=str):
            mask = df[stratify_col] == g
            if mask.sum() < 5:
                continue  # skip groups too small to estimate
            kmf_g = KaplanMeierFitter()
            kmf_g.fit(
                durations=df.loc[mask, dur],
                event_observed=df.loc[mask, evt],
                label=str(g),
            )
            group_kmfs[str(g)] = kmf_g

        # Log-rank test (only meaningful for exactly 2 groups)
        if len(group_kmfs) == 2:
            keys = list(group_kmfs.keys())
            mask0 = df[stratify_col].astype(str) == keys[0]
            mask1 = df[stratify_col].astype(str) == keys[1]
            try:
                lr = logrank_test(
                    df.loc[mask0, dur], df.loc[mask1, dur],
                    event_observed_A=df.loc[mask0, evt],
                    event_observed_B=df.loc[mask1, evt],
                )
                logrank_pval = float(lr.p_value)
            except Exception:
                pass

    return KMResult(
        median_survival=median_surv,
        survival_at_times=survival_at,
        overall_kmf=kmf,
        group_kmfs=group_kmfs,
        stratify_col=stratify_col,
        logrank_pvalue=logrank_pval,
        n_total=dataset.n_total,
        n_events=dataset.n_events,
    )


# ── 3. Parametric models ──────────────────────────────────────────────────────

def fit_parametric(dataset: SurvivalDataset) -> ParametricResult:
    """Fit Weibull and LogNormal parametric models; compare via AIC.

    Weibull rho interpretation:
      rho > 1  — increasing hazard (wear-out / ageing)
      rho = 1  — constant hazard (memoryless / exponential)
      rho < 1  — decreasing hazard (infant mortality / early failures)

    LogNormal is preferred when the hazard peaks at a mid-point then decreases
    (common in customer churn: early-life and late-life cohorts survive, mid-life churns most).
    """
    _require_lifelines()
    df = dataset.df
    dur = dataset.duration_col
    evt = dataset.event_col

    wf = WeibullFitter()
    wf.fit(df[dur], df[evt], label="Weibull")

    lnf = LogNormalFitter()
    lnf.fit(df[dur], df[evt], label="LogNormal")

    better = "Weibull" if float(wf.AIC_) <= float(lnf.AIC_) else "LogNormal"

    return ParametricResult(
        weibull_aic=float(wf.AIC_),
        lognormal_aic=float(lnf.AIC_),
        better_model=better,
        weibull_lambda=float(wf.lambda_),
        weibull_rho=float(wf.rho_),
        lognormal_mu=float(lnf.mu_),
        lognormal_sigma=float(lnf.sigma_),
        weibull_fitter=wf,
        lognormal_fitter=lnf,
        median_weibull=float(wf.median_survival_time_),
        median_lognormal=float(lnf.median_survival_time_),
    )


# ── 4. Cox Proportional Hazards ───────────────────────────────────────────────

def fit_cox(
    dataset: SurvivalDataset,
    covariates: Optional[List[str]] = None,
    penalizer: float = 0.1,
) -> CoxResult:
    """Fit a Cox Proportional Hazards model with L2 regularisation.

    Parameters
    ----------
    dataset    : prepared SurvivalDataset
    covariates : covariate column names (defaults to dataset.covariate_cols)
    penalizer  : L2 regularisation strength; increase to 0.5-1.0 if convergence warnings appear

    Returns
    -------
    CoxResult with hazard ratios, 95% CIs, p-values, and concordance index (C-index).
    C-index > 0.7 = good discrimination; > 0.8 = excellent.
    """
    _require_lifelines()

    if covariates is None:
        covariates = dataset.covariate_cols

    if not covariates:
        raise ValueError(
            "No covariates available for Cox model. "
            "Either include numeric columns in your data or specify covariates= explicitly. "
            "Categorical columns must be passed as categorical_cols= in prepare_survival_data()."
        )

    df = dataset.df
    dur = dataset.duration_col
    evt = dataset.event_col
    cox_df = df[[dur, evt] + covariates].copy()

    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(cox_df, duration_col=dur, event_col=evt)

    # ── Parse summary into a clean DataFrame ─────────────────────────────────
    raw = cph.summary.copy()
    raw.index.name = "covariate"
    raw = raw.reset_index()

    # Rename columns defensively (lifelines column names vary by version)
    rename_map: Dict[str, str] = {}
    for col in raw.columns:
        cl = col.lower()
        if col == "exp(coef)":
            rename_map[col] = "hazard_ratio"
        elif col == "exp(coef) lower 95%":
            rename_map[col] = "hr_lower_95"
        elif col == "exp(coef) upper 95%":
            rename_map[col] = "hr_upper_95"
        elif col == "p":
            rename_map[col] = "p_value"
        elif col == "z":
            rename_map[col] = "z_score"
    raw = raw.rename(columns=rename_map)

    keep = [c for c in ["covariate", "hazard_ratio", "hr_lower_95", "hr_upper_95",
                         "p_value", "z_score"] if c in raw.columns]
    summary = raw[keep].copy()

    sig_covars: List[str] = (
        summary.loc[summary["p_value"] < 0.05, "covariate"].tolist()
        if "p_value" in summary.columns else []
    )

    return CoxResult(
        summary_df=summary,
        concordance=float(cph.concordance_index_),
        covariates_used=covariates,
        cph_fitter=cph,
        cox_df=cox_df,
        significant_covariates=sig_covars,
    )


# ── 5. Prediction ─────────────────────────────────────────────────────────────

def predict_survival(
    dataset: SurvivalDataset,
    time_points: List[float],
    km_result: Optional[KMResult] = None,
    param_result: Optional[ParametricResult] = None,
    cox_result: Optional[CoxResult] = None,
    new_entity: Optional[Dict] = None,
) -> dict:
    """Predict survival probability at given time points.

    Population-level: KM, Weibull, LogNormal (no new_entity needed).
    Individual-level: Cox + new_entity dict of covariate values.

    Parameters
    ----------
    time_points  : e.g. [30, 90, 180, 365] — the time units used in duration_col
    new_entity   : dict of covariate values for one individual,
                   e.g. {"age": 45, "segment_B": 1.0}
                   Missing covariates are imputed with training medians.
    """
    predictions: dict = {"time_points": time_points}

    if km_result is not None:
        predictions["kaplan_meier"] = {
            t: float(km_result.overall_kmf.predict(t)) for t in time_points
        }

    if param_result is not None:
        predictions["weibull"] = {
            t: float(param_result.weibull_fitter.predict(t)) for t in time_points
        }
        predictions["lognormal"] = {
            t: float(param_result.lognormal_fitter.predict(t)) for t in time_points
        }

    if cox_result is not None and new_entity is not None:
        entity_df = pd.DataFrame([new_entity])
        # Impute missing covariates with training medians
        for col in cox_result.covariates_used:
            if col not in entity_df.columns:
                med = float(dataset.df[col].median())
                entity_df[col] = med
        entity_df = entity_df[cox_result.covariates_used]
        try:
            sf = cox_result.cph_fitter.predict_survival_function(entity_df)
            # sf is a DataFrame: index = time, columns = [0]
            sf_index = sf.index.values
            cox_preds: Dict[float, float] = {}
            for t in time_points:
                closest_idx = int(np.argmin(np.abs(sf_index - t)))
                cox_preds[t] = float(sf.iloc[closest_idx, 0])
            predictions["cox_individual"] = cox_preds
        except Exception as exc:
            predictions["cox_individual_error"] = str(exc)

    return predictions


# ── 5b. Batch prediction ──────────────────────────────────────────────────────

def predict_batch(
    dataset: SurvivalDataset,
    cox_result: CoxResult,
    new_df: pd.DataFrame,
    time_horizons: List[int],
    categorical_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None,
    risk_tier_method: str = "percentile",
    risk_horizon_col: Optional[str] = None,
) -> pd.DataFrame:
    """Score a batch of entities using the fitted Cox model.

    Handles encoding alignment automatically — new_df does not need to be
    pre-encoded; raw categorical columns are one-hot encoded here and aligned
    to the training schema.

    Parameters
    ----------
    dataset          : SurvivalDataset produced by prepare_survival_data
    cox_result       : CoxResult from fit_cox
    new_df           : raw DataFrame of entities to score (un-encoded)
    time_horizons    : list of time points to predict, e.g. [365, 730, 1825]
    categorical_cols : categorical columns to one-hot encode (same as used in
                       prepare_survival_data). Defaults to dataset.categorical_cols.
    id_cols          : columns from new_df to carry through to output unchanged
                       (e.g. member IDs, names). Auto-detected if None.
    risk_tier_method : 'percentile' (default) — top 5% = High, 5-20% = Medium,
                       bottom 80% = Low. OR 'absolute' — uses thresholds from
                       risk_tier_thresholds parameter.
    risk_horizon_col : which churn_prob_{N}yr column to tier on. Defaults to the
                       median time horizon in time_horizons.

    Returns
    -------
    pd.DataFrame with id_cols + churn_prob_{N}yr + survival_prob_{N}yr columns
    for each horizon, plus risk_tier column. Sorted by the risk horizon column
    descending (highest risk first).
    """
    _require_lifelines()

    cat_cols = categorical_cols if categorical_cols is not None else dataset.categorical_cols
    covariate_cols = cox_result.covariates_used
    train_medians  = dataset.df[covariate_cols].median()
    cph            = cox_result.cph_fitter

    # Auto-detect identifier columns if not specified
    if id_cols is None:
        id_cols = []
        for col in new_df.columns:
            cl = col.lower()
            if any(hint in cl for hint in ("id", "name", "account", "member", "facility", "org")):
                id_cols.append(col)
        # cap at 5 id columns
        id_cols = id_cols[:5]

    # One-hot encode using same categorical columns as training
    enc_df = new_df.copy()
    if cat_cols:
        enc_df = pd.get_dummies(enc_df, columns=[c for c in cat_cols if c in enc_df.columns],
                                drop_first=False)

    # Align to training covariate schema
    for col in covariate_cols:
        if col not in enc_df.columns:
            enc_df[col] = 0
    enc_covariates = enc_df[covariate_cols].fillna(train_medians)

    # Predict survival function for all entities: shape (time, n_entities)
    surv_funcs = cph.predict_survival_function(enc_covariates)

    # Extract probabilities at each horizon
    result_df = new_df[id_cols].copy().reset_index(drop=True) if id_cols else pd.DataFrame(index=range(len(enc_covariates)))

    for h in time_horizons:
        idx   = surv_funcs.index.searchsorted(h)
        idx   = min(idx, len(surv_funcs) - 1)
        probs = surv_funcs.iloc[idx].values
        label = h // 365 if h >= 365 else h
        unit  = "yr" if h >= 365 else "day"
        result_df[f"survival_prob_{label}{unit}"] = np.round(probs, 4)
        result_df[f"churn_prob_{label}{unit}"]    = np.round(1 - probs, 4)

    # Determine which column to tier on
    horizon_labels = [
        (f"churn_prob_{h // 365}yr" if h >= 365 else f"churn_prob_{h}day")
        for h in time_horizons
    ]
    if risk_horizon_col and risk_horizon_col in result_df.columns:
        tier_col = risk_horizon_col
    else:
        mid = len(horizon_labels) // 2
        tier_col = horizon_labels[mid]

    # Assign risk tiers
    if risk_tier_method == "percentile":
        p80 = result_df[tier_col].quantile(0.80)
        p95 = result_df[tier_col].quantile(0.95)
        def _tier(v: float) -> str:
            if v >= p95: return "High"
            if v >= p80: return "Medium"
            return "Low"
        result_df["risk_tier"] = result_df[tier_col].apply(_tier)
    else:
        # Fallback absolute thresholds
        def _tier_abs(v: float) -> str:
            if v >= 0.35: return "High"
            if v >= 0.20: return "Medium"
            return "Low"
        result_df["risk_tier"] = result_df[tier_col].apply(_tier_abs)

    result_df["_sort_key"] = result_df[tier_col]
    result_df = result_df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"]).reset_index(drop=True)
    return result_df


# ── 6. Plotting ───────────────────────────────────────────────────────────────

def _get_plt():
    """Return matplotlib.pyplot configured for non-interactive (Agg) backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_km(
    km_result: KMResult,
    out_dir: str,
    title: str = "",
) -> List[str]:
    """Plot overall and (optionally) stratified KM survival curves.

    Returns list of saved PNG file paths.
    """
    _require_lifelines()
    plt = _get_plt()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]
    paths: List[str] = []

    # Overall KM
    fig, ax = plt.subplots(figsize=(10, 6))
    km_result.overall_kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.axhline(0.5, color="crimson", linestyle="--", alpha=0.6, label="S(t) = 0.50 (median)")
    ax.set_title(title or "Kaplan-Meier Survival Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability  S(t)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, f"km_overall_{ts}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # Stratified KM
    if km_result.group_kmfs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, kmf_g in km_result.group_kmfs.items():
            kmf_g.plot_survival_function(ax=ax, ci_show=False)
        strat_title = f"KM Survival by {km_result.stratify_col or 'group'}"
        if km_result.logrank_pvalue is not None:
            sig = "p<0.05 *" if km_result.logrank_pvalue < 0.05 else "not significant"
            strat_title += f"\nlog-rank  p = {km_result.logrank_pvalue:.4f}  ({sig})"
        ax.axhline(0.5, color="crimson", linestyle="--", alpha=0.4)
        ax.set_title(strat_title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability  S(t)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = os.path.join(out_dir, f"km_stratified_{ts}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


def plot_model_comparison(
    km_result: KMResult,
    param_result: ParametricResult,
    out_dir: str,
) -> str:
    """Overlay KM, Weibull, and LogNormal survival curves on one chart.

    Lower AIC = better parametric fit. Returns path of saved PNG.
    """
    _require_lifelines()
    plt = _get_plt()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]

    fig, ax = plt.subplots(figsize=(11, 6))
    km_result.overall_kmf.plot_survival_function(ax=ax, ci_show=True, label="Kaplan-Meier (non-parametric)")
    param_result.weibull_fitter.plot_survival_function(
        ax=ax, ci_show=False,
        label=f"Weibull  (AIC = {param_result.weibull_aic:.1f})"
    )
    param_result.lognormal_fitter.plot_survival_function(
        ax=ax, ci_show=False,
        label=f"LogNormal  (AIC = {param_result.lognormal_aic:.1f})"
    )
    aic_diff = abs(param_result.weibull_aic - param_result.lognormal_aic)
    ax.set_title(
        f"Survival Model Comparison\n"
        f"Best parametric fit: {param_result.better_model}  (ΔAIC = {aic_diff:.1f})"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability  S(t)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    p = os.path.join(out_dir, f"model_comparison_{ts}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_cox(
    cox_result: CoxResult,
    out_dir: str,
) -> List[str]:
    """Plot two Cox charts:
      1. Hazard ratio forest plot (horizontal dot + CI per covariate)
      2. Baseline survival function S0(t)

    Returns list of saved PNG paths.
    """
    _require_lifelines()
    plt = _get_plt()
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:20]
    paths: List[str] = []

    summary = cox_result.summary_df
    n = len(summary)
    if n == 0:
        return paths

    # ── Forest plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.7 + 1.5)))
    y_pos = list(range(n))
    hr_vals = summary["hazard_ratio"].values

    colors = [
        "crimson" if c in cox_result.significant_covariates else "steelblue"
        for c in summary["covariate"]
    ]

    if "hr_lower_95" in summary.columns and "hr_upper_95" in summary.columns:
        lo = summary["hr_lower_95"].values
        hi = summary["hr_upper_95"].values
        xerr_lo = np.maximum(hr_vals - lo, 0.0)
        xerr_hi = np.maximum(hi - hr_vals, 0.0)
        for i, (hr, xl, xh, col) in enumerate(zip(hr_vals, xerr_lo, xerr_hi, colors)):
            ax.errorbar(hr, i, xerr=[[xl], [xh]], fmt="o",
                        color=col, capsize=5, markersize=7, linewidth=1.5)
    else:
        for i, (hr, col) in enumerate(zip(hr_vals, colors)):
            ax.plot(hr, i, "o", color=col, markersize=8)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="HR = 1.0 (no effect)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary["covariate"].tolist())
    ax.set_xlabel("Hazard Ratio  (HR > 1: higher risk,  HR < 1: protective)")
    ax.set_title(
        f"Cox PH  —  Hazard Ratios  (C-index = {cox_result.concordance:.3f})\n"
        f"Crimson = significant at p < 0.05"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, f"cox_hazard_ratios_{ts}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ── Baseline survival S0(t) ───────────────────────────────────────────────
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        cox_result.cph_fitter.baseline_survival_.plot(ax=ax2)
        ax2.set_title("Cox PH — Baseline Survival Function  S₀(t)\n(survival when all covariates = 0)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("S₀(t)")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        p2 = os.path.join(out_dir, f"cox_baseline_survival_{ts}.png")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        paths.append(p2)
    except Exception:
        pass

    return paths


# ── 7. Plain-English summary ──────────────────────────────────────────────────

def survival_summary(
    dataset: SurvivalDataset,
    km_result: KMResult,
    param_result: Optional[ParametricResult] = None,
    cox_result: Optional[CoxResult] = None,
    domain_ctx: Optional[dict] = None,
) -> str:
    """Return a structured plain-English summary of the full survival analysis."""
    lines: List[str] = []

    domain = (domain_ctx or {}).get("domain", "General")
    role = (domain_ctx or {}).get("role", "data analyst")

    lines += [
        "=" * 66,
        "SURVIVAL / CHURN ANALYSIS SUMMARY",
        "=" * 66,
        f"Domain   : {domain}",
        f"Role     : {role}",
        "",
        "-- DATASET ---------------------------------------------------------",
        f"  Total observations : {dataset.n_total:,}",
    ]
    if dataset.n_total > 0:
        lines += [
            f"  Events (observed)  : {dataset.n_events:,}  ({dataset.n_events / dataset.n_total * 100:.1f}%)",
            f"  Censored           : {dataset.n_censored:,}  ({dataset.n_censored / dataset.n_total * 100:.1f}%)",
        ]
    lines += [
        f"  Duration column    : '{dataset.duration_col}'",
        f"  Event column       : '{dataset.event_col}'",
        f"  Median duration    : {dataset.median_duration:.2f}",
    ]

    if dataset.warnings:
        lines += ["", "  Data prep notes:"]
        for w in dataset.warnings:
            lines.append(f"    - {w}")

    lines += [
        "",
        "-- KAPLAN-MEIER ----------------------------------------------------",
        f"  Median survival time : {km_result.median_survival:.2f}",
        "  Survival probabilities at evaluated time points:",
    ]
    for t, prob in km_result.survival_at_times.items():
        lines.append(f"    S(t={t:.1f}) = {prob:.4f}  ({prob * 100:.1f}% survive past this point)")

    if km_result.group_kmfs and km_result.stratify_col:
        lines += ["", f"  Stratified by '{km_result.stratify_col}':"]
        for grp, kmf_g in km_result.group_kmfs.items():
            med_g = float(kmf_g.median_survival_time_)
            lines.append(f"    Group '{grp}'  —  median survival = {med_g:.2f}")
        if km_result.logrank_pvalue is not None:
            sig_str = (
                "SIGNIFICANT difference between groups (alpha = 0.05)"
                if km_result.logrank_pvalue < 0.05
                else "no significant difference between groups"
            )
            lines.append(f"    Log-rank test p = {km_result.logrank_pvalue:.4f}  -> {sig_str}")

    if param_result is not None:
        lines += [
            "",
            "-- PARAMETRIC MODEL COMPARISON ------------------------------------",
            f"  Weibull    AIC = {param_result.weibull_aic:.2f}  "
            f"| lambda={param_result.weibull_lambda:.4f}, rho={param_result.weibull_rho:.4f}  "
            f"| median={param_result.median_weibull:.2f}",
            f"  LogNormal  AIC = {param_result.lognormal_aic:.2f}  "
            f"| mu={param_result.lognormal_mu:.4f}, sigma={param_result.lognormal_sigma:.4f}  "
            f"| median={param_result.median_lognormal:.2f}",
            f"  Best fit (lower AIC) : {param_result.better_model}",
            "",
            "  Hazard shape interpretation (Weibull rho / shape parameter):",
        ]
        rho = param_result.weibull_rho
        if rho > 1.2:
            lines.append(
                f"    rho={rho:.3f} > 1  ->  INCREASING hazard over time (wear-out / ageing).\n"
                "    Risk of churn/failure grows as time passes."
            )
        elif rho < 0.8:
            lines.append(
                f"    rho={rho:.3f} < 1  ->  DECREASING hazard over time (infant mortality / early failures).\n"
                "    Risk is highest early, then survivors become increasingly stable."
            )
        else:
            lines.append(
                f"    rho={rho:.3f} ~= 1  ->  roughly CONSTANT hazard (memoryless / exponential).\n"
                "    Churn/failure rate is stable over time."
            )

    if cox_result is not None:
        c_idx = cox_result.concordance
        c_qual = (
            "excellent discrimination" if c_idx > 0.8
            else "good discrimination" if c_idx > 0.7
            else "fair discrimination (C-index 0.6-0.7; consider adding more covariates)" if c_idx > 0.6
            else "poor discrimination -- covariates explain little churn/failure timing"
        )
        lines += [
            "",
            "-- COX PROPORTIONAL HAZARDS ---------------------------------------",
            f"  C-index (concordance) : {c_idx:.4f}  ({c_qual})",
            f"  Covariates used       : {len(cox_result.covariates_used)} total",
            "",
        ]

        # ── Significant covariates first (highlighted) ────────────────────────
        sig_df = cox_result.summary_df[
            cox_result.summary_df["covariate"].isin(cox_result.significant_covariates)
        ].sort_values("hazard_ratio", ascending=False)

        if not sig_df.empty:
            lines.append("  KEY DRIVERS (significant at p < 0.05)  -- HR > 1 = higher risk, HR < 1 = protective:")
            for _, row in sig_df.iterrows():
                covar = row.get("covariate", "?")
                hr    = row.get("hazard_ratio", float("nan"))
                pval  = row.get("p_value", float("nan"))
                lo    = row.get("hr_lower_95", float("nan"))
                hi    = row.get("hr_upper_95", float("nan"))
                direction = "RISK" if hr > 1 else "PROT"
                ci_str = f"[{lo:.3f}-{hi:.3f}]" if not (np.isnan(lo) or np.isnan(hi)) else ""
                lines.append(f"    [{direction}] {covar:<34s}  HR={hr:.3f} {ci_str}  p={pval:.2e}")
        else:
            lines.append("  No covariate reached significance at alpha=0.05 with current data/penalizer.")

        # ── All covariates compact table ───────────────────────────────────────
        lines += ["", "  All covariates (compact):"]
        for _, row in cox_result.summary_df.iterrows():
            covar = row.get("covariate", "?")
            hr    = row.get("hazard_ratio", float("nan"))
            pval  = row.get("p_value", float("nan"))
            sig_marker = " *" if covar in cox_result.significant_covariates else "  "
            lines.append(f"    {sig_marker}{covar:<34s}  HR={hr:.3f}  p={pval:.4f}")

    lines += ["", "=" * 66]
    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
