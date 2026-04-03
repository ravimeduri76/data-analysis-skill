"""Autoresearch-style ML experiment loop.

Consolidates the evaluate() function from automl scripts into a single
reusable hook and adds an autonomous experiment loop that tries multiple
models, tracks keep/discard/crash per trial, and advances an incumbent score.

With an Anthropic API key, enables LLM-guided experiment proposals:
the LLM reads its own history + feature importances + domain context,
proposes the next model/hyperparams/feature mutations, and rewrites a
mutable experiment_strategy.md between rounds (Karpathy autoresearch pattern).

Public API:
    select_target, build_feature_matrix, leakage_audit, evaluate,
    run_experiment_loop, append_results_tsv, render_experiment_summary,
    extract_feature_importances, apply_feature_mutations
"""
from __future__ import annotations

import json
import logging
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

log = logging.getLogger("ml_experiment")


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
def select_target(
    df: pd.DataFrame,
    col_classes: dict[str, list[str]],
    domain_ctx: dict[str, Any],
) -> Optional[dict]:
    """Auto-detect a prediction target from the dataset.

    Returns dict with keys: col, task_type, rationale, positive_rate
    or None if no suitable target is found.
    """
    id_cols = set(col_classes.get("identifiers", []))
    fact_cols = col_classes.get("facts", [])
    dim_cols = col_classes.get("dimensions", [])

    # Gather domain hints for preferred targets
    key_metrics = domain_ctx.get("key_metrics", [])
    hint_names = {m.lower() for m in key_metrics} if key_metrics else set()

    candidates: list[dict] = []

    # Scan fact columns for binary targets
    for col in fact_cols:
        if col in id_cols:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        nuniq = s.nunique()

        # Binary: exactly 2 unique values (0/1, True/False, yes/no)
        if nuniq == 2:
            vals = sorted(s.unique())
            # Try to interpret as 0/1
            try:
                mapped = s.map({vals[0]: 0, vals[1]: 1})
                pos_rate = mapped.mean()
            except Exception:
                pos_rate = 0.5

            score = 10  # base score for binary
            # Prefer 1-30% positive rate (imbalanced but not extreme)
            if 0.01 <= pos_rate <= 0.30:
                score += 5
            elif 0.30 < pos_rate <= 0.50:
                score += 2
            # Boost if column name matches domain hints
            if col.lower() in hint_names or any(h in col.lower() for h in hint_names):
                score += 3

            candidates.append({
                "col": col,
                "task_type": "binary",
                "rationale": f"Binary column ({nuniq} unique), positive_rate={pos_rate:.3f}",
                "positive_rate": round(pos_rate, 4),
                "score": score,
            })

        # Low-cardinality int as potential multiclass
        elif nuniq <= 10 and pd.api.types.is_integer_dtype(s):
            candidates.append({
                "col": col,
                "task_type": "multiclass",
                "rationale": f"Low-cardinality integer ({nuniq} classes)",
                "positive_rate": None,
                "score": 3,
            })

        # Continuous numeric for regression
        elif pd.api.types.is_numeric_dtype(s) and nuniq > 10:
            score = 2
            if col.lower() in hint_names or any(h in col.lower() for h in hint_names):
                score += 3
            candidates.append({
                "col": col,
                "task_type": "regression",
                "rationale": f"Continuous numeric ({nuniq} unique values)",
                "positive_rate": None,
                "score": score,
            })

    # Also scan dimension columns for binary flags
    for col in dim_cols:
        if col in id_cols:
            continue
        s = df[col].dropna()
        if len(s) == 0 or s.nunique() != 2:
            continue
        vals = sorted(s.unique())
        try:
            mapped = s.map({vals[0]: 0, vals[1]: 1})
            pos_rate = mapped.mean()
        except Exception:
            pos_rate = 0.5
        score = 8
        if 0.01 <= pos_rate <= 0.30:
            score += 5
        candidates.append({
            "col": col,
            "task_type": "binary",
            "rationale": f"Binary dimension ({vals}), positive_rate={pos_rate:.3f}",
            "positive_rate": round(pos_rate, 4),
            "score": score,
        })

    if not candidates:
        return None

    # Pick highest score; prefer binary over regression
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    best.pop("score", None)
    return best


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
def leakage_audit(X: pd.DataFrame, y: pd.Series) -> set[str]:
    """Detect perfectly collinear / leaky columns via cross-tab + overlap test.

    Returns set of column names to drop.
    """
    leaky: set[str] = set()
    y_binary = y.astype(int) if y.dtype != int else y

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            continue
        try:
            nonzero_mask = X[col] > 0
            if nonzero_mask.sum() == 0:
                continue

            # Cross-tab test: if feature>0 implies target=1 (no FP at all)
            target_when_nonzero = y_binary[nonzero_mask]
            if (target_when_nonzero == 0).sum() == 0:
                leaky.add(col)
                continue

            # Overlap test: if count(feature>0) == count(target=1)
            if nonzero_mask.sum() == (y_binary == 1).sum():
                leaky.add(col)
        except Exception:
            pass

    if leaky:
        log.warning("Leakage audit: dropping %d columns: %s", len(leaky), leaky)
    return leaky


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build clean (X, y, feature_names) from raw DataFrame.

    - Drops target_col and id_cols
    - One-hot encodes categoricals (max 20 unique; drop_first=True)
    - Fills numeric NaN with median, categorical NaN with 'MISSING'
    - Runs leakage_audit and drops leaky columns
    """
    drop_cols = {target_col} | set(id_cols or [])
    y_raw = df[target_col].copy()

    # Encode y for binary
    if y_raw.nunique() == 2:
        vals = sorted(y_raw.dropna().unique())
        y = y_raw.map({vals[0]: 0, vals[1]: 1}).fillna(0).astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Separate numeric and categorical
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Fill numeric NaN with median
    for c in num_cols:
        median_val = X[c].median()
        X[c] = X[c].fillna(median_val if pd.notna(median_val) else 0)

    # One-hot encode categoricals with <= 20 unique
    cat_to_encode = [c for c in cat_cols if X[c].nunique() <= 20]
    cat_to_drop = [c for c in cat_cols if X[c].nunique() > 20]

    if cat_to_drop:
        X = X.drop(columns=cat_to_drop)
        log.info("Dropped %d high-cardinality categoricals: %s", len(cat_to_drop), cat_to_drop[:5])

    if cat_to_encode:
        X[cat_to_encode] = X[cat_to_encode].fillna("MISSING")
        X = pd.get_dummies(X, columns=cat_to_encode, drop_first=True, dtype=int)

    # Leakage audit
    leaky = leakage_audit(X, y)
    if leaky:
        X = X.drop(columns=list(leaky))

    feature_names = X.columns.tolist()
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Evaluation (verbatim from automl_mortality_v7.py:77-91)
# ---------------------------------------------------------------------------
def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray,
             y_pred: np.ndarray | None = None) -> dict:
    """Compute classification metrics for a single model.

    Returns dict with: Model, ROC_AUC, PR_AUC, F1_pos, Recall,
    Specificity, MCC, TP, FP, TN, FN.
    """
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    f1p = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = (y_pred[y_true == 1] == 1).mean()
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return dict(
        Model=name, ROC_AUC=round(roc, 4), PR_AUC=round(pr, 4),
        F1_pos=round(f1p, 4), Recall=round(rec, 4),
        Specificity=round(spec, 4), MCC=round(mcc, 4),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
    )


def evaluate_regression(name: str, y_true: np.ndarray,
                        y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return dict(Model=name, R2=round(r2, 4), RMSE=round(rmse, 4))


# ---------------------------------------------------------------------------
# Model roster
# ---------------------------------------------------------------------------
def _build_model_roster_binary() -> list[dict]:
    """Build the default model roster for binary classification."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    roster: list[dict] = [
        {
            "name": "LogisticRegression",
            "model": Pipeline([
                ("scaler", RobustScaler()),
                ("lr", LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42,
                )),
            ]),
            "description": "LR baseline (balanced, RobustScaler)",
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            "description": "RF n=200 balanced",
        },
    ]

    # LightGBM (optional)
    try:
        import lightgbm as lgb
        roster.append({
            "name": "LightGBM",
            "model": lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05,
                is_unbalance=True, random_state=42,
                verbose=-1, n_jobs=-1,
            ),
            "description": "LGB is_unbalance lr=0.05",
        })
    except ImportError:
        log.info("LightGBM not installed -- skipping")

    # XGBoost (optional)
    try:
        import xgboost as xgb
        roster.append({
            "name": "XGBoost_aucpr",
            "model": xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05,
                eval_metric="aucpr", use_label_encoder=False,
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            "description": "XGB eval_metric=aucpr",
        })
        roster.append({
            "name": "XGBoost_logloss",
            "model": xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05,
                eval_metric="logloss", use_label_encoder=False,
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            "description": "XGB eval_metric=logloss",
        })
    except ImportError:
        log.info("XGBoost not installed -- skipping")

    # MLP
    roster.append({
        "name": "MLP",
        "model": Pipeline([
            ("scaler", RobustScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, random_state=42,
            )),
        ]),
        "description": "MLP (128,64) early_stopping",
    })

    return roster


def _build_model_roster_regression() -> list[dict]:
    """Build the default model roster for regression."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    roster: list[dict] = [
        {
            "name": "Ridge",
            "model": Pipeline([
                ("scaler", RobustScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]),
            "description": "Ridge baseline (alpha=1.0)",
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=-1,
            ),
            "description": "RF n=200 regressor",
        },
    ]

    try:
        import lightgbm as lgb
        roster.append({
            "name": "LightGBM",
            "model": lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05,
                random_state=42, verbose=-1, n_jobs=-1,
            ),
            "description": "LGB regressor lr=0.05",
        })
    except ImportError:
        pass

    try:
        import xgboost as xgb
        roster.append({
            "name": "XGBoost",
            "model": xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05,
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            "description": "XGB regressor",
        })
    except ImportError:
        pass

    roster.append({
        "name": "MLP",
        "model": Pipeline([
            ("scaler", RobustScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, random_state=42,
            )),
        ]),
        "description": "MLP (128,64) regressor",
    })

    return roster


def _generate_variations(top_models: list[dict], task_type: str,
                         n_variations: int) -> list[dict]:
    """Generate hyperparameter variations of the top-performing models."""
    import copy
    rng = np.random.RandomState(int(time.time()) % 2**31)
    variations: list[dict] = []

    for entry in top_models[:2]:
        for i in range(n_variations):
            new_entry = copy.deepcopy(entry)
            model = new_entry["model"]
            name_base = new_entry["name"]

            # Perturb tree-based models
            if hasattr(model, "n_estimators"):
                model.n_estimators = int(rng.choice([100, 200, 400, 500]))
                if hasattr(model, "max_depth"):
                    model.max_depth = int(rng.choice([3, 5, 8, 12, None]))
                if hasattr(model, "learning_rate"):
                    model.learning_rate = float(rng.choice([0.01, 0.03, 0.1, 0.15]))
                desc = f"n_est={model.n_estimators}"
                if hasattr(model, "learning_rate"):
                    desc += f" lr={model.learning_rate}"
            else:
                # Skip variation for non-tree models
                continue

            new_entry["name"] = f"{name_base}_v{i+1}"
            new_entry["description"] = f"{name_base} variation ({desc})"
            variations.append(new_entry)
            if len(variations) >= n_variations:
                break
        if len(variations) >= n_variations:
            break

    return variations


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------
def extract_feature_importances(
    model: Any,
    feature_names: list[str],
    top_k: int = 15,
) -> list[tuple[str, float]]:
    """Extract top-k feature importances from a fitted model.

    Unwraps Pipeline to get the inner estimator. Supports tree-based
    (feature_importances_) and linear (coef_) models.
    Returns list of (name, importance) sorted descending.
    """
    estimator = model
    # Unwrap Pipeline
    if hasattr(estimator, "steps"):
        estimator = estimator.steps[-1][1]

    importances: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return []

    indices = np.argsort(importances)[::-1][:top_k]
    return [(feature_names[i], round(float(importances[i]), 6)) for i in indices]


# ---------------------------------------------------------------------------
# Feature mutation engine
# ---------------------------------------------------------------------------
_MUTATION_TYPES = {"ratio", "interaction", "bin", "log", "drop"}


def apply_feature_mutations(
    X: pd.DataFrame,
    y: pd.Series,
    mutations: list[dict],
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply LLM-proposed feature mutations to X (from original copy).

    Allowed types: ratio, interaction, bin, log, drop.
    Runs leakage_audit on result. Invalid mutations are skipped with a log warning.

    Returns (X_mutated, new_feature_names, descriptions).
    """
    X_mut = X.copy()
    descriptions: list[str] = []

    for mut in mutations:
        mtype = mut.get("type", "")
        if mtype not in _MUTATION_TYPES:
            log.warning("Unknown mutation type '%s' -- skipped", mtype)
            continue

        try:
            if mtype == "ratio":
                num, den = mut["numerator"], mut["denominator"]
                if num not in X_mut.columns or den not in X_mut.columns:
                    log.warning("ratio: column not found (%s/%s) -- skipped", num, den)
                    continue
                safe_den = X_mut[den].clip(lower=1)
                col_name = f"{num}_div_{den}"
                X_mut[col_name] = X_mut[num] / safe_den
                descriptions.append(f"ratio({num}/{den})")

            elif mtype == "interaction":
                col_a, col_b = mut["col_a"], mut["col_b"]
                if col_a not in X_mut.columns or col_b not in X_mut.columns:
                    log.warning("interaction: column not found (%s*%s) -- skipped", col_a, col_b)
                    continue
                col_name = f"{col_a}_x_{col_b}"
                X_mut[col_name] = X_mut[col_a] * X_mut[col_b]
                descriptions.append(f"interaction({col_a}*{col_b})")

            elif mtype == "bin":
                col = mut["col"]
                if col not in X_mut.columns:
                    log.warning("bin: column '%s' not found -- skipped", col)
                    continue
                bins = max(2, min(10, int(mut.get("bins", 5))))
                col_name = f"{col}_binned"
                X_mut[col_name] = pd.qcut(X_mut[col], q=bins, labels=False, duplicates="drop")
                X_mut[col_name] = X_mut[col_name].fillna(0).astype(int)
                descriptions.append(f"bin({col}, q={bins})")

            elif mtype == "log":
                col = mut["col"]
                if col not in X_mut.columns or not pd.api.types.is_numeric_dtype(X_mut[col]):
                    log.warning("log: column '%s' not found or not numeric -- skipped", col)
                    continue
                col_name = f"log1p_{col}"
                X_mut[col_name] = np.log1p(X_mut[col].clip(lower=0))
                descriptions.append(f"log1p({col})")

            elif mtype == "drop":
                cols = mut.get("cols", [])
                existing = [c for c in cols if c in X_mut.columns]
                if len(existing) > len(X_mut.columns) * 0.5:
                    log.warning("drop: refusing to drop >50%% of features (%d/%d)", len(existing), len(X_mut.columns))
                    continue
                X_mut = X_mut.drop(columns=existing)
                descriptions.append(f"drop({existing})")

        except Exception as exc:
            log.warning("Mutation %s failed: %s -- skipped", mtype, exc)
            continue

    # Leakage audit on mutated features
    leaky = leakage_audit(X_mut, y)
    if leaky:
        X_mut = X_mut.drop(columns=list(leaky))
        descriptions.append(f"leakage_audit_dropped({list(leaky)})")

    new_names = X_mut.columns.tolist()
    return X_mut, new_names, descriptions


# ---------------------------------------------------------------------------
# LLM-guided experiment proposal
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-6"


def propose_next_experiment(
    strategy_md: str,
    results_tsv: str,
    importances: list[tuple[str, float]],
    feature_names: list[str],
    domain_ctx: dict | None,
    task_type: str,
    primary_metric: str,
    anthropic_api_key: str,
) -> dict | None:
    """Ask Claude to propose the next experiment based on history.

    Returns parsed dict: {model, hyperparams, mutations, hypothesis, confidence}
    or None on any failure (falls back to random variations).
    """
    domain_summary = ""
    if domain_ctx:
        domain_summary = (
            f"Domain: {domain_ctx.get('domain', 'Unknown')}\n"
            f"Summary: {domain_ctx.get('summary', 'N/A')}\n"
            f"Key metrics: {domain_ctx.get('key_metrics', [])}"
        )

    imp_text = "\n".join(f"  {name}: {val:.4f}" for name, val in importances[:15])
    feat_text = ", ".join(feature_names[:50])

    prompt = f"""You are an ML research assistant. Based on the experiment history below,
propose the SINGLE best next experiment to improve {primary_metric} for a {task_type} task.

## Strategy Document (current hypotheses)
{strategy_md[:2000]}

## Recent Results (TSV, last 20 rows)
{results_tsv[-2000:]}

## Top Feature Importances
{imp_text}

## Available Features (first 50)
{feat_text}

## Domain Context
{domain_summary}

Reply with ONLY valid JSON (no markdown fences):
{{
  "model": "RandomForest|LightGBM|XGBoost|LogisticRegression|MLP",
  "hyperparams": {{"n_estimators": 300, "learning_rate": 0.05, ...}},
  "mutations": [
    {{"type": "ratio", "numerator": "colA", "denominator": "colB"}},
    {{"type": "interaction", "col_a": "colA", "col_b": "colB"}},
    {{"type": "bin", "col": "colA", "bins": 5}},
    {{"type": "log", "col": "colA"}},
    {{"type": "drop", "cols": ["colA", "colB"]}}
  ],
  "hypothesis": "Short description of what this experiment tests",
  "confidence": "high|medium|low"
}}

Include 0-3 mutations. Only reference columns from the available features list.
Focus on what the previous results suggest would improve {primary_metric}."""

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_api_key)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        proposal = json.loads(text)
        log.info("LLM proposal: model=%s, mutations=%d, hypothesis=%s",
                 proposal.get("model"), len(proposal.get("mutations", [])),
                 proposal.get("hypothesis", "")[:80])
        return proposal
    except Exception as exc:
        log.warning("propose_next_experiment failed: %s -- falling back to variations", exc)
        return None


# ---------------------------------------------------------------------------
# Build model from LLM proposal
# ---------------------------------------------------------------------------
_ALLOWED_HYPERPARAMS = {
    "RandomForest": {
        "n_estimators": (50, 1000), "max_depth": (2, 30),
        "min_samples_split": (2, 20), "min_samples_leaf": (1, 10),
    },
    "LightGBM": {
        "n_estimators": (50, 1000), "learning_rate": (0.005, 0.3),
        "max_depth": (-1, 30), "num_leaves": (15, 255),
        "min_child_samples": (5, 100), "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.0, 10.0),
    },
    "XGBoost": {
        "n_estimators": (50, 1000), "learning_rate": (0.005, 0.3),
        "max_depth": (2, 15), "min_child_weight": (1, 20),
        "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (0.0, 10.0), "reg_lambda": (0.0, 10.0),
    },
    "LogisticRegression": {
        "C": (0.001, 100.0), "max_iter": (100, 5000),
    },
    "MLP": {
        "max_iter": (100, 2000),
    },
}


def _build_model_from_proposal(proposal: dict, task_type: str) -> dict | None:
    """Validate and construct a model entry from an LLM proposal.

    Clamps hyperparams to allowed ranges, drops unknown params.
    Returns roster entry dict {name, model, description} or None on failure.
    """
    model_name = proposal.get("model", "")
    raw_params = proposal.get("hyperparams", {})
    hypothesis = proposal.get("hypothesis", "")

    allowed = _ALLOWED_HYPERPARAMS.get(model_name)
    if allowed is None:
        log.warning("Unknown model '%s' in proposal -- skipping", model_name)
        return None

    # Clamp params
    clean_params: dict[str, Any] = {}
    for k, v in raw_params.items():
        if k not in allowed:
            continue
        lo, hi = allowed[k]
        try:
            v = type(lo)(v)  # cast to same type as bounds
            v = max(lo, min(hi, v))
        except (TypeError, ValueError):
            continue
        clean_params[k] = v

    is_clf = task_type in ("binary", "multiclass")

    try:
        if model_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            cls = RandomForestClassifier if is_clf else RandomForestRegressor
            defaults = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
            if is_clf:
                defaults["class_weight"] = "balanced"
            defaults.update(clean_params)
            model = cls(**defaults)

        elif model_name == "LightGBM":
            import lightgbm as lgb
            cls = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
            defaults = {"n_estimators": 300, "learning_rate": 0.05,
                        "random_state": 42, "verbose": -1, "n_jobs": -1}
            if is_clf:
                defaults["is_unbalance"] = True
            defaults.update(clean_params)
            model = cls(**defaults)

        elif model_name == "XGBoost":
            import xgboost as xgb
            cls = xgb.XGBClassifier if is_clf else xgb.XGBRegressor
            defaults = {"n_estimators": 300, "learning_rate": 0.05,
                        "random_state": 42, "n_jobs": -1, "verbosity": 0}
            if is_clf:
                defaults["eval_metric"] = "aucpr"
                defaults["use_label_encoder"] = False
            defaults.update(clean_params)
            model = cls(**defaults)

        elif model_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import RobustScaler
            defaults = {"class_weight": "balanced", "max_iter": 1000, "random_state": 42}
            defaults.update(clean_params)
            model = Pipeline([("scaler", RobustScaler()), ("lr", LogisticRegression(**defaults))])

        elif model_name == "MLP":
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import RobustScaler
            cls = MLPClassifier if is_clf else MLPRegressor
            defaults = {"hidden_layer_sizes": (128, 64), "max_iter": 500,
                        "early_stopping": True, "random_state": 42}
            defaults.update(clean_params)
            model = Pipeline([("scaler", RobustScaler()), ("mlp", cls(**defaults))])
        else:
            return None

    except Exception as exc:
        log.warning("Failed to build model from proposal: %s", exc)
        return None

    # Build description
    param_str = " ".join(f"{k}={v}" for k, v in clean_params.items())
    desc = f"LLM: {model_name} {param_str}".strip()
    if hypothesis:
        desc += f" | {hypothesis[:60]}"

    return {
        "name": f"LLM_{model_name}",
        "model": model,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# Strategy document I/O
# ---------------------------------------------------------------------------
def _load_strategy(path: Path) -> str:
    """Load experiment_strategy.md content, or empty string if missing."""
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _save_strategy(path: Path, content: str) -> None:
    """Write experiment_strategy.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _initial_strategy(
    feature_names: list[str],
    domain_ctx: dict | None,
    task_type: str,
    primary_metric: str,
) -> str:
    """Generate the initial experiment_strategy.md content."""
    domain = domain_ctx.get("domain", "Unknown") if domain_ctx else "Unknown"
    return (
        f"# Experiment Strategy\n\n"
        f"## Setup\n"
        f"- Task: {task_type}\n"
        f"- Primary metric: {primary_metric}\n"
        f"- Features: {len(feature_names)}\n"
        f"- Domain: {domain}\n\n"
        f"## Hypotheses\n"
        f"1. Tree-based models (RF, LGB, XGB) will outperform linear models on this dataset\n"
        f"2. Feature interactions between top importances may improve {primary_metric}\n"
        f"3. Log transforms on skewed numeric features may help linear models\n\n"
        f"## What Worked\n"
        f"_(after round 0)_\n\n"
        f"## What Didn't Work\n"
        f"_(after round 0)_\n"
    )


def _update_strategy(
    strategy_path: Path,
    round_results: list[dict],
    importances: list[tuple[str, float]],
    incumbent_metric: float,
    primary_metric: str,
    anthropic_api_key: str | None = None,
) -> str:
    """Rewrite experiment_strategy.md after a round of experiments.

    With API key: Claude summarizes results. Without: template update.
    """
    current = _load_strategy(strategy_path)

    # Build results summary
    results_text = "\n".join(
        f"- {t['model']}: {primary_metric}={t['metrics'].get(primary_metric, 'N/A')} "
        f"[{t['status']}] {t['description']}"
        for t in round_results
    )
    imp_text = "\n".join(f"  {name}: {val:.4f}" for name, val in importances[:10])

    if anthropic_api_key:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=anthropic_api_key)

            prompt = f"""You are updating an ML experiment strategy document.
Current strategy:
{current[:1500]}

Latest round results:
{results_text}

Top feature importances:
{imp_text}

Current best {primary_metric}: {incumbent_metric:.4f}

Rewrite the strategy document (max 800 words). Include:
1. Updated hypotheses based on what worked/didn't
2. Feature importance observations
3. Suggested next directions
Keep the same markdown structure (## Setup, ## Hypotheses, ## What Worked, ## What Didn't Work).
Output ONLY the markdown content, no code fences."""

            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            updated = resp.content[0].text.strip()
            _save_strategy(strategy_path, updated)
            return updated
        except Exception as exc:
            log.warning("LLM strategy update failed: %s -- using template", exc)

    # Template-based fallback
    updated = (
        f"{current}\n\n"
        f"## Round Update ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')})\n\n"
        f"### Results\n{results_text}\n\n"
        f"### Best {primary_metric}: {incumbent_metric:.4f}\n\n"
        f"### Top Importances\n{imp_text}\n"
    )
    # Cap at ~800 words
    words = updated.split()
    if len(words) > 800:
        updated = " ".join(words[:800]) + "\n\n_(truncated)_"
    _save_strategy(strategy_path, updated)
    return updated


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------
def run_experiment_loop(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    task_type: str = "binary",
    max_trials: int = 8,
    primary_metric: str | None = None,
    # LLM-guided autoresearch (all optional, backward compatible)
    anthropic_api_key: str | None = None,
    domain_ctx: dict | None = None,
    out_dir: str | Path | None = None,
    n_rounds: int = 2,
) -> dict:
    """Autoresearch-style experiment loop.

    Without anthropic_api_key: runs static roster + random variations (original behavior).
    With anthropic_api_key: after base roster, runs n_rounds of LLM-guided experiments
    where Claude reads history + feature importances + domain context, proposes next
    model/hyperparams/feature mutations, and rewrites experiment_strategy.md each round.

    Returns:
        dict with keys: trials, best_trial, incumbent_metric, tsv_rows,
        strategy_path, llm_proposals, best_feature_set, best_importances
    """
    if primary_metric is None:
        primary_metric = "PR_AUC" if task_type == "binary" else "R2"

    is_classification = task_type in ("binary", "multiclass")
    llm_enabled = bool(anthropic_api_key)

    # Strategy doc path
    strategy_path: Path | None = None
    if out_dir:
        strategy_path = Path(out_dir) / "experiment_strategy.md"

    # Build roster
    if is_classification:
        roster = _build_model_roster_binary()
    else:
        roster = _build_model_roster_regression()

    # Keep original X for LLM-guided rounds (mutations start fresh each time)
    X_original = X.copy()

    # Split data
    if is_classification:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42,
        )

    trials: list[dict] = []
    llm_proposals: list[dict | None] = []
    incumbent_score = -np.inf
    best_trial: dict = {}
    best_model_obj: Any = None
    best_feature_names: list[str] = list(feature_names)
    trial_id = 0

    def _run_trial(entry: dict, trial_X_train: pd.DataFrame,
                   trial_X_test: pd.DataFrame, trial_feature_names: list[str]) -> dict:
        nonlocal trial_id, incumbent_score, best_trial, best_model_obj, best_feature_names
        trial_id += 1
        tid = f"{trial_id:03d}"

        try:
            t0 = time.time()
            model = entry["model"]
            model.fit(trial_X_train, y_train)
            elapsed = time.time() - t0

            if is_classification:
                y_prob = model.predict_proba(trial_X_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                metrics = evaluate(entry["name"], y_test.values, y_prob, y_pred)
            else:
                y_pred = model.predict(trial_X_test)
                metrics = evaluate_regression(entry["name"], y_test.values, y_pred)

            score = metrics.get(primary_metric, -np.inf)

            if score > incumbent_score:
                status = "keep"
                incumbent_score = score
                best_trial = {
                    "trial_id": tid,
                    "metrics": metrics,
                    "description": entry["description"],
                    "elapsed_s": round(elapsed, 2),
                }
                best_model_obj = model
                best_feature_names = list(trial_feature_names)
            else:
                status = "discard"

            trial = {
                "trial_id": tid,
                "model": entry["name"],
                "metrics": metrics,
                "status": status,
                "description": entry["description"],
                "n_features": len(trial_feature_names),
                "elapsed_s": round(elapsed, 2),
            }

        except Exception as exc:
            log.error("Trial %s (%s) crashed: %s", tid, entry["name"], exc)
            trial = {
                "trial_id": tid,
                "model": entry["name"],
                "metrics": {},
                "status": "crash",
                "description": f"{entry['description']} -- ERROR: {exc}",
                "n_features": len(trial_feature_names),
                "elapsed_s": 0,
            }

        trials.append(trial)
        log.info(
            "Trial %s: %s -> %s=%s [%s]",
            tid, entry["name"],
            primary_metric,
            trial["metrics"].get(primary_metric, "N/A"),
            trial["status"],
        )
        return trial

    # -----------------------------------------------------------------------
    # Phase 1 (Round 0): run the base roster — unchanged
    # -----------------------------------------------------------------------
    base_budget = min(len(roster), max_trials)
    for entry in roster[:base_budget]:
        _run_trial(entry, X_train, X_test, feature_names)

    # Extract importances from best model so far
    current_importances = extract_feature_importances(
        best_model_obj, best_feature_names
    ) if best_model_obj else []

    # Write initial strategy doc
    if strategy_path:
        initial = _initial_strategy(feature_names, domain_ctx, task_type, primary_metric)
        _save_strategy(strategy_path, initial)
        log.info("Strategy doc initialized: %s", strategy_path)

    # -----------------------------------------------------------------------
    # Phase 2: LLM-guided rounds (or fallback to random variations)
    # -----------------------------------------------------------------------
    remaining = max_trials - len(trials)

    if remaining > 0 and llm_enabled and strategy_path:
        # LLM-guided rounds
        trials_per_round = max(1, remaining // n_rounds) if n_rounds > 0 else remaining
        llm_round = 0

        while remaining > 0 and llm_round < n_rounds:
            llm_round += 1
            round_budget = min(trials_per_round, remaining)
            round_trials_start = len(trials)

            # Build TSV text for proposal prompt
            tsv_text = _build_tsv_text(trials, is_classification)

            for _trial_in_round in range(round_budget):
                strategy_md = _load_strategy(strategy_path)

                proposal = propose_next_experiment(
                    strategy_md=strategy_md,
                    results_tsv=tsv_text,
                    importances=current_importances,
                    feature_names=best_feature_names,
                    domain_ctx=domain_ctx,
                    task_type=task_type,
                    primary_metric=primary_metric,
                    anthropic_api_key=anthropic_api_key,
                )
                llm_proposals.append(proposal)

                if proposal:
                    # Apply feature mutations (from original X, not compounding)
                    mutations = proposal.get("mutations", [])
                    if mutations:
                        X_mut, mut_names, mut_descs = apply_feature_mutations(
                            X_original, y, mutations, feature_names,
                        )
                        # Re-split with same random state
                        if is_classification:
                            Xt, Xv, _, _ = train_test_split(
                                X_mut, y, test_size=0.20, stratify=y, random_state=42,
                            )
                        else:
                            Xt, Xv, _, _ = train_test_split(
                                X_mut, y, test_size=0.20, random_state=42,
                            )
                        trial_fnames = mut_names
                        log.info("Applied %d mutations: %s", len(mut_descs), mut_descs)
                    else:
                        Xt, Xv, trial_fnames = X_train, X_test, feature_names

                    # Build model from proposal
                    entry = _build_model_from_proposal(proposal, task_type)
                    if entry:
                        # Append mutation info to description
                        if mutations and mut_descs:
                            entry["description"] += f" + {', '.join(mut_descs[:3])}"
                        _run_trial(entry, Xt, Xv, trial_fnames)
                    else:
                        # Proposal had bad model → fallback to variation
                        _run_fallback_variation(
                            trials, roster, task_type, remaining,
                            X_train, X_test, feature_names, _run_trial,
                        )
                else:
                    # LLM failed → fallback to variation
                    _run_fallback_variation(
                        trials, roster, task_type, remaining,
                        X_train, X_test, feature_names, _run_trial,
                    )

                # Update TSV text for next trial in same round
                tsv_text = _build_tsv_text(trials, is_classification)

            # End of round: update importances + strategy
            if best_model_obj:
                current_importances = extract_feature_importances(
                    best_model_obj, best_feature_names,
                )

            round_trials = trials[round_trials_start:]
            _update_strategy(
                strategy_path, round_trials, current_importances,
                incumbent_score, primary_metric, anthropic_api_key,
            )
            log.info("Round %d complete: %d trials, best %s=%.4f",
                     llm_round, len(round_trials), primary_metric, incumbent_score)

            remaining = max_trials - len(trials)

    elif remaining > 0:
        # Fallback: original random variation logic (no API key)
        completed = [t for t in trials if t["status"] != "crash" and t["metrics"]]
        if completed:
            completed.sort(
                key=lambda t: t["metrics"].get(primary_metric, -np.inf),
                reverse=True,
            )
            top_names = [t["model"] for t in completed[:2]]
            top_entries = [e for e in roster if e["name"] in top_names]
            variations = _generate_variations(top_entries, task_type, remaining)
            for v in variations[:remaining]:
                _run_trial(v, X_train, X_test, feature_names)

    # Final importances from best model
    best_importances = extract_feature_importances(
        best_model_obj, best_feature_names,
    ) if best_model_obj else []

    # Build TSV rows
    tsv_rows = _build_tsv_rows(trials, is_classification)

    return {
        "trials": trials,
        "best_trial": best_trial,
        "incumbent_metric": incumbent_score,
        "primary_metric": primary_metric,
        "task_type": task_type,
        "tsv_rows": tsv_rows,
        # New keys
        "strategy_path": str(strategy_path) if strategy_path else None,
        "llm_proposals": llm_proposals,
        "best_feature_set": best_feature_names,
        "best_importances": best_importances,
    }


def _build_tsv_text(trials: list[dict], is_classification: bool) -> str:
    """Build TSV text from trials for LLM prompt context."""
    rows = _build_tsv_rows(trials, is_classification)
    return "\n".join(rows[-20:])  # last 20 rows


def _build_tsv_rows(trials: list[dict], is_classification: bool) -> list[str]:
    """Build TSV row strings from trial dicts."""
    tsv_rows: list[str] = []
    for t in trials:
        m = t["metrics"]
        if is_classification:
            row = (
                f"{t['trial_id']}\t{t['model']}\t"
                f"{m.get('PR_AUC', '')}\t{m.get('MCC', '')}\t"
                f"{m.get('ROC_AUC', '')}\t{t['n_features']}\t"
                f"{t['status']}\t{t['description']}"
            )
        else:
            row = (
                f"{t['trial_id']}\t{t['model']}\t"
                f"{m.get('R2', '')}\t{m.get('RMSE', '')}\t"
                f"\t{t['n_features']}\t"
                f"{t['status']}\t{t['description']}"
            )
        tsv_rows.append(row)
    return tsv_rows


def _run_fallback_variation(
    trials: list[dict],
    roster: list[dict],
    task_type: str,
    remaining: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    run_trial_fn: Any,
) -> None:
    """Run a single random variation as fallback when LLM proposal fails."""
    completed = [t for t in trials if t["status"] != "crash" and t["metrics"]]
    if not completed:
        return
    completed.sort(
        key=lambda t: t["metrics"].get("PR_AUC", t["metrics"].get("R2", -np.inf)),
        reverse=True,
    )
    top_names = [t["model"] for t in completed[:2]]
    top_entries = [e for e in roster if e["name"] in top_names]
    if top_entries:
        variations = _generate_variations(top_entries, task_type, 1)
        if variations:
            run_trial_fn(variations[0], X_train, X_test, feature_names)


# ---------------------------------------------------------------------------
# TSV persistence (autoresearch-style ledger)
# ---------------------------------------------------------------------------
def append_results_tsv(
    trials: list[dict],
    tsv_path: str | Path,
    dataset_name: str = "",
    domain: str = "",
    target_col: str = "",
) -> str:
    """Append trial results to an autoresearch-style TSV ledger.

    Returns path to the TSV file.
    """
    tsv_path = Path(tsv_path)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    header = "trial\tdataset\tdomain\ttarget\tmodel\tPR_AUC\tMCC\tROC_AUC\tn_features\tstatus\tdescription\n"

    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0

    with open(tsv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        for t in trials:
            m = t["metrics"]
            line = (
                f"{t['trial_id']}\t{dataset_name}\t{domain}\t{target_col}\t"
                f"{t['model']}\t{m.get('PR_AUC', '')}\t{m.get('MCC', '')}\t"
                f"{m.get('ROC_AUC', '')}\t{t['n_features']}\t"
                f"{t['status']}\t{t['description']}\n"
            )
            f.write(line)

    log.info("TSV updated: %s (%d trials)", tsv_path, len(trials))
    return str(tsv_path)


# ---------------------------------------------------------------------------
# Summary rendering (ASCII-safe)
# ---------------------------------------------------------------------------
def render_experiment_summary(result: dict) -> str:
    """Render an ASCII-safe markdown summary of the experiment loop."""
    lines: list[str] = []
    lines.append("## ML Experiment Results")
    lines.append("")

    task = result.get("task_type", "binary")
    pm = result.get("primary_metric", "PR_AUC")
    best = result.get("best_trial", {})
    trials = result.get("trials", [])

    n_keep = sum(1 for t in trials if t["status"] == "keep")
    n_discard = sum(1 for t in trials if t["status"] == "discard")
    n_crash = sum(1 for t in trials if t["status"] == "crash")

    lines.append(f"**Task:** {task} | **Primary metric:** {pm}")
    lines.append(f"**Trials:** {len(trials)} total ({n_keep} keep, {n_discard} discard, {n_crash} crash)")
    lines.append("")

    if best and best.get("metrics"):
        bm = best["metrics"]
        lines.append(f"**Best model:** {bm.get('Model', 'N/A')}")
        lines.append(f"- {pm}: {bm.get(pm, 'N/A')}")
        if task == "binary":
            lines.append(f"- ROC_AUC: {bm.get('ROC_AUC', 'N/A')}")
            lines.append(f"- MCC: {bm.get('MCC', 'N/A')}")
            lines.append(f"- Recall: {bm.get('Recall', 'N/A')} | Specificity: {bm.get('Specificity', 'N/A')}")
        elif task == "regression":
            lines.append(f"- R2: {bm.get('R2', 'N/A')}")
            lines.append(f"- RMSE: {bm.get('RMSE', 'N/A')}")
        lines.append(f"- Description: {best.get('description', '')}")
        lines.append(f"- Training time: {best.get('elapsed_s', 0)}s")
    else:
        lines.append("**No successful trials.**")

    lines.append("")
    lines.append("### Leaderboard")
    lines.append("")

    if task == "binary":
        lines.append("| # | Model | PR_AUC | MCC | ROC_AUC | Status |")
        lines.append("|---|-------|--------|-----|---------|--------|")
        for t in trials:
            m = t["metrics"]
            lines.append(
                f"| {t['trial_id']} | {t['model']} | "
                f"{m.get('PR_AUC', '-')} | {m.get('MCC', '-')} | "
                f"{m.get('ROC_AUC', '-')} | {t['status']} |"
            )
    else:
        lines.append("| # | Model | R2 | RMSE | Status |")
        lines.append("|---|-------|----|------|--------|")
        for t in trials:
            m = t["metrics"]
            lines.append(
                f"| {t['trial_id']} | {t['model']} | "
                f"{m.get('R2', '-')} | {m.get('RMSE', '-')} | "
                f"{t['status']} |"
            )

    # LLM Research Notes section
    llm_proposals = result.get("llm_proposals", [])
    best_importances = result.get("best_importances", [])
    strategy_path = result.get("strategy_path")

    if llm_proposals or best_importances:
        lines.append("")
        lines.append("### LLM Research Notes")
        lines.append("")

        if llm_proposals:
            n_llm = sum(1 for p in llm_proposals if p is not None)
            lines.append(f"**LLM-guided trials:** {n_llm}/{len(llm_proposals)} proposals accepted")
            for i, p in enumerate(llm_proposals):
                if p:
                    lines.append(f"- Round {i+1}: {p.get('model', '?')} -- {p.get('hypothesis', 'N/A')[:80]}")

        if best_importances:
            lines.append("")
            lines.append("**Top feature importances (best model):**")
            for name, val in best_importances[:10]:
                lines.append(f"- {name}: {val:.4f}")

        if strategy_path:
            lines.append("")
            lines.append(f"**Strategy doc:** {strategy_path}")

    text = "\n".join(lines)
    # ASCII-safe
    text = text.encode("ascii", errors="replace").decode("ascii")
    return text
