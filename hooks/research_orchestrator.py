"""4-Phase Research Orchestrator for structured ML experimentation.

Wraps (not replaces) the existing ml_experiment.py with a proper research
methodology: Understand -> Design (creative DoE) -> Prioritize -> Execute.

Public API
----------
run_research_project      Main entry point — runs full 4-phase loop
phase1_understand         Phase 1: profile data, frame objective
phase2_design             Phase 2: design experiments across 7 surfaces
phase3_prioritize         Phase 3: rank experiments by impact/cost/diversity
phase4_execute            Phase 4: execute experiments with stopping criteria
compute_stopping_threshold   Blend of LLM reasoning + heuristic
adapt_stopping_threshold     Adjust threshold based on improvement history
update_research_project_md   Generate living research artifact
render_research_summary      ASCII-safe summary of research project
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass
class ImprovementVector:
    """A specific improvement direction within a research surface."""
    name: str
    description: str
    experiment_templates: list[dict] = field(default_factory=list)
    cost_estimate: float = 1.0  # relative cost (1=cheap, 5=expensive)


@dataclass
class ResearchSurface:
    """One of 7 research surfaces to explore."""
    id: str
    name: str
    vectors: list[ImprovementVector] = field(default_factory=list)


@dataclass
class Experiment:
    """A single planned or completed experiment."""
    id: str
    surface: str
    vector: str
    hypothesis: str
    design: dict = field(default_factory=dict)
    expected_impact: float = 0.0  # 0..1
    priority_score: float = 0.0
    status: str = "planned"  # planned | running | completed | skipped
    results: dict = field(default_factory=dict)
    triage: str = "PARK"  # WI-2C: ACT_NOW | RESEARCH_MORE | PARK | KILL


@dataclass
class ResearchProject:
    """Full state of a research project."""
    objective: str = ""
    surfaces: list[ResearchSurface] = field(default_factory=list)
    experiments: list[Experiment] = field(default_factory=list)
    baseline: dict = field(default_factory=dict)
    stopping_threshold: float = 0.0
    threshold_rationale: str = ""
    budget: dict = field(default_factory=dict)
    phase: str = "init"  # init | understand | design | execute | done
    iteration: int = 0
    improvement_history: list[dict] = field(default_factory=list)
    best_score: float = -np.inf
    best_model: str = ""
    primary_metric: str = ""
    task_type: str = ""
    domain: str = ""
    all_trials: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    contradictions: list[dict] = field(default_factory=list)  # WI-2B
    strategy_proposals: list[dict] = field(default_factory=list)  # WI-2D


# ===========================================================================
# Seed surfaces (7 hardcoded, LLM can extend)
# ===========================================================================
def _build_seed_surfaces() -> list[ResearchSurface]:
    """Build the 7 default research surfaces."""
    return [
        ResearchSurface(
            id="data", name="Data",
            vectors=[
                ImprovementVector(
                    "quality_fixes", "Fix data quality: imputation strategies",
                    [{"type": "impute", "strategy": s} for s in ["knn", "iterative", "median"]],
                    cost_estimate=1.5,
                ),
                ImprovementVector(
                    "quantity_augmentation", "Augment data: SMOTE, noise injection",
                    [{"type": "smote"}, {"type": "noise", "std": 0.01}],
                    cost_estimate=2.0,
                ),
            ],
        ),
        ResearchSurface(
            id="features", name="Features",
            vectors=[
                ImprovementVector(
                    "explicit", "Explicit features: ratios, interactions, differences",
                    [{"type": "ratio"}, {"type": "interaction"}, {"type": "difference"}],
                    cost_estimate=1.0,
                ),
                ImprovementVector(
                    "latent", "Latent features: PCA, target encoding, cluster membership",
                    [{"type": "pca", "n_components": 10}, {"type": "cluster", "n_clusters": 5}],
                    cost_estimate=2.0,
                ),
                ImprovementVector(
                    "engineering", "Feature engineering: log, bin, polynomial",
                    [{"type": "log"}, {"type": "bin"}, {"type": "polynomial", "degree": 2}],
                    cost_estimate=1.0,
                ),
            ],
        ),
        ResearchSurface(
            id="architectures", name="Architectures",
            vectors=[
                ImprovementVector(
                    "ensemble", "Ensembles: voting, stacking, blending",
                    [{"type": "voting", "voting": "soft"}, {"type": "stacking"}],
                    cost_estimate=3.0,
                ),
                ImprovementVector(
                    "mixture_of_experts", "Mixture of experts: partition + specialist",
                    [{"type": "moe"}],
                    cost_estimate=4.0,
                ),
            ],
        ),
        ResearchSurface(
            id="models", name="Models",
            vectors=[
                ImprovementVector(
                    "new_model_types", "New models: CatBoost, ExtraTrees, GBM, SVM, KNN",
                    [{"type": "extended_roster"}],
                    cost_estimate=2.0,
                ),
            ],
        ),
        ResearchSurface(
            id="parameters", name="Parameters",
            vectors=[
                ImprovementVector(
                    "bayesian_optimization", "Optuna Bayesian optimization on best models",
                    [{"type": "optuna", "n_trials": 20}],
                    cost_estimate=3.0,
                ),
                ImprovementVector(
                    "targeted_grid", "Targeted grid search on best model hyperparams",
                    [{"type": "grid_search"}],
                    cost_estimate=2.5,
                ),
            ],
        ),
        ResearchSurface(
            id="validation", name="Validation",
            vectors=[
                ImprovementVector(
                    "kfold_cv", "K-fold cross-validation for robust estimates",
                    [{"type": "kfold", "n_splits": 5}],
                    cost_estimate=2.0,
                ),
                ImprovementVector(
                    "bootstrap_ci", "Bootstrap confidence intervals",
                    [{"type": "bootstrap", "n_bootstrap": 200}],
                    cost_estimate=2.5,
                ),
                ImprovementVector(
                    "overfitting_detection", "Learning curves, train-test gap analysis",
                    [{"type": "learning_curve"}],
                    cost_estimate=2.0,
                ),
            ],
        ),
        ResearchSurface(
            id="interpretation", name="Interpretation",
            vectors=[
                ImprovementVector(
                    "shap_summary", "SHAP feature importance analysis",
                    [{"type": "shap"}],
                    cost_estimate=3.0,
                ),
                ImprovementVector(
                    "permutation_importance", "Permutation importance ranking",
                    [{"type": "permutation"}],
                    cost_estimate=1.5,
                ),
                ImprovementVector(
                    "threshold_analysis", "Optimal threshold tuning for classification",
                    [{"type": "threshold_tuning"}],
                    cost_estimate=1.0,
                ),
            ],
        ),
    ]


# ===========================================================================
# Phase 1: Understand
# ===========================================================================
def phase1_understand(
    df: pd.DataFrame,
    col_classes: dict[str, list[str]],
    domain_ctx: dict[str, Any],
    target_col: str,
    objective: str | None,
    quality_report_df: pd.DataFrame | None,
) -> dict:
    """Phase 1: Profile data, frame research objective.

    Returns dict with: objective, n_rows, n_cols, n_features, target_col,
    task_type, positive_rate, quality_summary, data_challenges.
    """
    from hooks.ml_experiment import select_target

    n_rows, n_cols = df.shape
    target_info = select_target(df, col_classes, domain_ctx)

    if target_info is None and target_col:
        # Manual target override
        y = df[target_col]
        nunique = y.nunique()
        if nunique == 2:
            task_type = "binary"
        elif nunique <= 20:
            task_type = "multiclass"
        else:
            task_type = "regression"
        positive_rate = float((y == 1).mean()) if task_type == "binary" else None
        target_info = {"col": target_col, "task_type": task_type,
                       "rationale": "user-specified", "positive_rate": positive_rate}
    elif target_info is None:
        return {"error": "No suitable prediction target found"}

    task_type = target_info["task_type"]
    target_col = target_info["col"]

    # Frame objective
    domain = domain_ctx.get("domain", "Unknown") if domain_ctx else "Unknown"
    if not objective:
        objective = (
            f"Maximize {_default_metric(task_type)} for {task_type} prediction "
            f"of '{target_col}' in the {domain} domain"
        )

    # Quality summary
    quality_summary = ""
    data_challenges: list[str] = []
    if quality_report_df is not None and not quality_report_df.empty:
        high_null = quality_report_df[quality_report_df["null_pct"] > 20]
        if len(high_null) > 0:
            data_challenges.append(f"{len(high_null)} columns with >20% nulls")
        low_quality = quality_report_df[quality_report_df["quality_score"] < 0.5]
        if len(low_quality) > 0:
            data_challenges.append(f"{len(low_quality)} low-quality columns (score<0.5)")
        constants = quality_report_df[quality_report_df["is_constant"] == True]
        if len(constants) > 0:
            data_challenges.append(f"{len(constants)} constant columns")
        quality_summary = "; ".join(data_challenges) if data_challenges else "Data quality looks good"

    return {
        "objective": objective,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target_col": target_col,
        "task_type": task_type,
        "positive_rate": target_info.get("positive_rate"),
        "quality_summary": quality_summary,
        "data_challenges": data_challenges,
        "domain": domain,
        "target_info": target_info,
    }


def _default_metric(task_type: str) -> str:
    if task_type == "binary":
        return "PR_AUC"
    elif task_type == "multiclass":
        return "Macro_F1"
    return "R2"


# ===========================================================================
# WI-2E: Prior findings helper
# ===========================================================================
def _format_prior_findings(findings: list[dict] | None) -> str:
    """Format prior experiment findings for LLM prompt injection."""
    if not findings:
        return "  (no prior lessons available)"
    lines = []
    for f in findings[:10]:
        ftype = f.get("finding_type", "neutral")
        surface = f.get("surface", "?")
        vector = f.get("vector", "?")
        score = f.get("metric_result")
        score_str = f"score={score:.4f}" if isinstance(score, (int, float)) else "score=N/A"
        finding = f.get("positive_finding") or f.get("negative_finding") or f.get("lesson_learned", "")
        if finding:
            lines.append(f"  - [{ftype}] {surface}/{vector} ({score_str}): {finding[:120]}")
    return "\n".join(lines) if lines else "  (no prior lessons available)"


def _build_prior_impact_adjustments(findings: list[dict] | None) -> dict[str, float]:
    """Build a {surface: adjustment} dict from prior findings.

    Positive findings on a surface boost expected_impact by +0.05.
    Negative findings reduce by -0.03.
    Returns net adjustment per surface.
    """
    if not findings:
        return {}
    adjustments: dict[str, float] = {}
    for f in findings:
        surface = f.get("surface", "")
        ftype = f.get("finding_type", "neutral")
        if ftype == "positive":
            adjustments[surface] = adjustments.get(surface, 0.0) + 0.05
        elif ftype == "negative":
            adjustments[surface] = adjustments.get(surface, 0.0) - 0.03
    # Clamp adjustments to [-0.15, +0.15]
    return {s: max(-0.15, min(0.15, v)) for s, v in adjustments.items()}


# ===========================================================================
# Phase 2: Design experiments (LLM DoE + heuristic fallback)
# ===========================================================================
def phase2_design(
    project: ResearchProject,
    baseline_result: dict,
    quality_df: pd.DataFrame | None,
    importances: list[tuple[str, float]],
    anthropic_api_key: str | None,
    prior_findings: list[dict] | None = None,
) -> list[Experiment]:
    """Phase 2: Design creative experiments across 7 research surfaces.

    WI-2E: prior_findings from experiment_ledger are used to boost/reduce
    expected_impact of experiments based on what worked/failed in past runs.

    Uses LLM to propose 8-15 experiments if API key is available,
    otherwise falls back to heuristic template instantiation.

    Returns list of Experiment objects.
    """
    experiments: list[Experiment] = []

    # Try LLM-guided design first
    if anthropic_api_key:
        llm_experiments = _llm_design_experiments(project, baseline_result,
                                                   quality_df, importances,
                                                   prior_findings)
        if llm_experiments:
            return llm_experiments

    # Heuristic fallback: template instantiation
    return _heuristic_design_experiments(project, baseline_result,
                                         quality_df, importances,
                                         prior_findings)


def _llm_design_experiments(
    project: ResearchProject,
    baseline_result: dict,
    quality_df: pd.DataFrame | None,
    importances: list[tuple[str, float]],
    prior_findings: list[dict] | None = None,
) -> list[Experiment] | None:
    """Use LLM to design creative experiments."""
    try:
        from hooks.llm_client import call_llm
    except ImportError:
        return None

    # Build quality summary
    quality_text = "No quality data available"
    if quality_df is not None and not quality_df.empty:
        quality_text = quality_df.head(15).to_string(index=False, max_colwidth=30)

    # Build importances text
    imp_text = "\n".join(f"  {name}: {val:.4f}" for name, val in importances[:10])
    if not imp_text:
        imp_text = "  (no importances available yet)"

    # Surface descriptions
    surface_text = ""
    for s in project.surfaces:
        vectors = ", ".join(v.name for v in s.vectors)
        surface_text += f"  {s.name}: {vectors}\n"

    # Previous experiments summary
    prev_text = ""
    completed = [e for e in project.experiments if e.status == "completed"]
    if completed:
        prev_text = "\n".join(
            f"  - [{e.surface}] {e.hypothesis} -> "
            f"{e.results.get('score', 'N/A')}"
            for e in completed[-10:]
        )

    baseline_score = baseline_result.get("incumbent_metric", 0)
    best_model = baseline_result.get("best_trial", {}).get("metrics", {}).get("Model", "N/A")

    prompt = f"""You are a machine learning research scientist designing experiments.

## Research Objective
{project.objective}

## Current State
- Task: {project.task_type}
- Primary metric: {project.primary_metric}
- Baseline score: {baseline_score:.4f} (model: {best_model})
- Stopping threshold: {project.stopping_threshold:.4f}
- Domain: {project.domain}
- Iteration: {project.iteration + 1}

## Data Quality
{quality_text}

## Top Feature Importances
{imp_text}

## Research Surfaces Available
{surface_text}

## Previous Experiments (if any)
{prev_text if prev_text else "  (none yet — this is the first iteration)"}

## Prior Experiment Lessons (from experiment ledger)
{_format_prior_findings(prior_findings) if prior_findings else "  (no prior lessons available)"}

## Instructions
Design 8-15 creative experiments across at least 4 different surfaces.
For each experiment, provide:
- surface: one of [data, features, architectures, models, parameters, validation, interpretation]
- vector: specific improvement direction
- hypothesis: what you expect and why
- design: specific parameters/configuration as a JSON object
- expected_impact: 0.0-1.0 (how much improvement you expect relative to headroom)
- reasoning: why this experiment is worth trying

Output a JSON array of experiment objects. Output ONLY valid JSON, no markdown fences.
Example format:
[
  {{"surface": "features", "vector": "explicit", "hypothesis": "Ratio of top-2 important features may capture nonlinear relationship", "design": {{"type": "ratio", "numerator": "feat_a", "denominator": "feat_b"}}, "expected_impact": 0.3, "reasoning": "Top features show high individual importance but may interact"}}
]"""

    response = call_llm(prompt, max_tokens=2500, expect_json=True)
    if not response:
        log.warning("Phase2: LLM returned empty response — falling back to heuristic")
        return None

    try:
        # Parse JSON response
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        proposals = json.loads(raw)
        if not isinstance(proposals, list):
            log.warning("Phase2: LLM response was not a JSON array (got %s) — falling back to heuristic",
                        type(proposals).__name__)
            return None
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Phase2: LLM response was not valid JSON: %s — first 200 chars: %s",
                    exc, response[:200])
        return None

    # Validate each proposal has required fields
    valid_surfaces = {"data", "features", "architectures", "models", "parameters", "validation", "interpretation"}
    experiments: list[Experiment] = []
    for i, p in enumerate(proposals[:15]):
        if not isinstance(p, dict):
            log.warning("Phase2: LLM proposal %d is not a dict, skipping", i)
            continue
        surface = p.get("surface", "features")
        if surface not in valid_surfaces:
            log.warning("Phase2: LLM proposal %d has invalid surface '%s', skipping", i, surface)
            continue
        design = p.get("design", {})
        if not isinstance(design, dict):
            log.warning("Phase2: LLM proposal %d has non-dict design, skipping", i)
            continue
        exp = Experiment(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            surface=surface,
            vector=p.get("vector", "unknown"),
            hypothesis=p.get("hypothesis", ""),
            design=design,
            expected_impact=min(1.0, max(0.0, float(p.get("expected_impact", 0.1)))),
        )
        experiments.append(exp)

    if len(experiments) < 4:
        log.warning("Phase2: LLM produced only %d valid experiments (need >=4) — falling back to heuristic",
                    len(experiments))
        return None

    log.info("Phase2: LLM designed %d experiments across %d surfaces",
             len(experiments), len(set(e.surface for e in experiments)))
    return experiments


def _heuristic_design_experiments(
    project: ResearchProject,
    baseline_result: dict,
    quality_df: pd.DataFrame | None,
    importances: list[tuple[str, float]],
    prior_findings: list[dict] | None = None,
) -> list[Experiment]:
    """Design experiments using template instantiation (no LLM needed).

    Experiments are ordered by expected impact (high-impact first) so that
    plateau detection doesn't kill the iteration before the best experiments run.

    WI-2E: prior_findings adjust expected_impact based on past experiment outcomes.
    """
    experiments: list[Experiment] = []

    # WI-2E: build impact adjustments from prior findings
    impact_adj = _build_prior_impact_adjustments(prior_findings)

    # Gather completed surface counts for diversity
    surface_counts: dict[str, int] = {}
    completed_designs: set[str] = set()
    for e in project.experiments:
        if e.status == "completed":
            surface_counts[e.surface] = surface_counts.get(e.surface, 0) + 1
            completed_designs.add(e.design.get("type", ""))

    top_features = [name for name, _ in importances[:10]] if importances else []
    baseline_score = baseline_result.get("incumbent_metric", 0)
    best_trial = baseline_result.get("best_trial", {})
    best_model_name = best_trial.get("metrics", {}).get("Model", "")
    task_type = project.task_type
    is_classification = task_type in ("binary", "multiclass")
    positive_rate = baseline_result.get("positive_rate")

    def _add(surface: str, vector: str, hypothesis: str, design: dict,
             impact: float, *, guard: bool = True) -> None:
        if not guard:
            return
        # WI-2E: adjust impact based on prior findings for this surface
        adjusted_impact = max(0.0, min(1.0, impact + impact_adj.get(surface, 0.0)))
        experiments.append(Experiment(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            surface=surface, vector=vector,
            hypothesis=hypothesis, design=design,
            expected_impact=adjusted_impact,
        ))

    # -----------------------------------------------------------------------
    # HIGH IMPACT (0.25-0.40) — run first
    # -----------------------------------------------------------------------

    # 1. Optuna Bayesian optimization — always try, no baseline gate
    _add("parameters", "bayesian_optimization",
         f"Bayesian optimization on {best_model_name or 'best model'} may find better hyperparameters",
         {"type": "optuna", "n_trials": 20, "model": best_model_name},
         impact=0.35,
         guard="optuna" not in completed_designs)

    # 2. Extended model roster (GBM, ExtraTrees, SVM, KNN, CatBoost)
    _add("models", "new_model_types",
         "Extended model roster (GBM, ExtraTrees, SVM, KNN) may find better architecture",
         {"type": "extended_roster"},
         impact=0.30,
         guard=surface_counts.get("models", 0) < 2)

    # 3. Feature selection — drop low-importance features to reduce noise
    if len(top_features) >= 5:
        # Keep top-60% features, drop the rest
        n_keep = max(3, int(len(top_features) * 0.6))
        low_imp = [name for name, _ in importances[n_keep:]] if importances else []
        if low_imp:
            _add("features", "selection",
                 f"Dropping {len(low_imp)} low-importance features may reduce noise",
                 {"type": "drop", "cols": low_imp},
                 impact=0.25,
                 guard="feature_selection" not in completed_designs)

    # 4. Class weight / threshold tuning for classification
    if is_classification and "threshold_tuning" not in completed_designs:
        _add("interpretation", "threshold_tuning",
             "Threshold optimization may improve F1/recall trade-off",
             {"type": "threshold_tuning"},
             impact=0.25)

    # 5. SMOTE for imbalanced binary
    if positive_rate is not None and positive_rate < 0.3:
        _add("data", "quantity_augmentation",
             f"SMOTE may help with class imbalance (positive_rate={positive_rate:.2f})",
             {"type": "smote"},
             impact=0.25,
             guard="smote" not in completed_designs)

    # -----------------------------------------------------------------------
    # MEDIUM IMPACT (0.15-0.24) — ensembles, data quality, targeted features
    # -----------------------------------------------------------------------

    # 6. Stacking ensemble
    _add("architectures", "ensemble",
         "Stacking with LR meta-learner may capture complementary model strengths",
         {"type": "stacking"},
         impact=0.22,
         guard="stacking" not in completed_designs)

    # 7. Voting ensemble
    _add("architectures", "ensemble",
         "Soft voting ensemble of diverse models may reduce variance",
         {"type": "voting", "voting": "soft"},
         impact=0.20,
         guard="voting" not in completed_designs)

    # 8. KNN imputation for high-null columns
    if quality_df is not None and not quality_df.empty:
        high_null = quality_df[quality_df["null_pct"] > 20]
        if len(high_null) > 0:
            _add("data", "quality_fixes",
                 "KNN imputation on high-null columns may preserve more signal than median fill",
                 {"type": "knn_impute", "columns": high_null["column"].tolist()[:5]},
                 impact=0.18,
                 guard="knn_impute" not in completed_designs)

    # 9. Feature interactions — top-1 * top-3 (skip adjacent-only pattern)
    if len(top_features) >= 4 and surface_counts.get("features", 0) < 4:
        # Cross top-1 with top-3,4 (non-adjacent, more likely to find novel signal)
        pairs = [(top_features[0], top_features[2]),
                 (top_features[0], top_features[3]),
                 (top_features[1], top_features[3])]
        for col_a, col_b in pairs:
            _add("features", "explicit",
                 f"Interaction {col_a} * {col_b} may capture non-adjacent synergy",
                 {"type": "interaction", "col_a": col_a, "col_b": col_b},
                 impact=0.15)

    # 10. Ratio features — top-1 / top-2 (only if both numeric)
    if len(top_features) >= 2:
        _add("features", "explicit",
             f"Ratio {top_features[0]} / {top_features[1]} may capture relative magnitude",
             {"type": "ratio", "numerator": top_features[0], "denominator": top_features[1]},
             impact=0.15,
             guard="ratio" not in completed_designs)

    # -----------------------------------------------------------------------
    # LOW IMPACT (0.0-0.14) — validation, interpretation, log transforms
    # -----------------------------------------------------------------------

    # 11. Log transforms on skewed features
    if top_features:
        _add("features", "engineering",
             "Log transforms on skewed numeric features may help linear models",
             {"type": "log_top_features", "features": top_features[:3]},
             impact=0.08,
             guard="log_top_features" not in completed_designs)

    # 12. K-fold CV for robust estimates
    _add("validation", "kfold_cv",
         "5-fold CV will give more robust performance estimate than single split",
         {"type": "kfold", "n_splits": 5, "model": best_model_name},
         impact=0.0,
         guard="kfold" not in completed_designs)

    # 13. Permutation importance — useful for next iteration's feature selection
    _add("interpretation", "permutation_importance",
         "Permutation importance may reveal features that hurt generalization",
         {"type": "permutation"},
         impact=0.05,
         guard="permutation" not in completed_designs)

    log.info("Phase2 heuristic: designed %d experiments across %d surfaces",
             len(experiments), len(set(e.surface for e in experiments)))
    return experiments


# ===========================================================================
# Phase 3: Prioritize experiments (with WI-2C triage)
# ===========================================================================
# ===========================================================================
# WI-2B: Contradiction detection (OB1 pattern: research-synthesis)
# ===========================================================================
def _detect_contradictions(experiments: list[Experiment]) -> list[dict]:
    """Find pairs of completed experiments on the same surface/vector with opposite outcomes.

    A contradiction is when one experiment improved the score and another on the same
    surface/vector did not (or vice versa).
    """
    completed = [e for e in experiments if e.status == "completed" and e.results]

    # Group by (surface, vector)
    groups: dict[tuple[str, str], list[Experiment]] = {}
    for e in completed:
        key = (e.surface, e.vector)
        groups.setdefault(key, []).append(e)

    contradictions: list[dict] = []
    for (surface, vector), group in groups.items():
        if len(group) < 2:
            continue

        positives = [e for e in group if e.results.get("best_score", -np.inf) > -np.inf
                     and e.results.get("best_score", -np.inf) > 0]
        negatives = [e for e in group if e.results.get("best_score", -np.inf) <= 0
                     or e.results.get("error")]

        # Check for mixed results: at least one experiment improved and one didn't
        scores = [e.results.get("best_score", -np.inf) for e in group
                  if isinstance(e.results.get("best_score"), (int, float))]
        if len(scores) >= 2:
            best = max(scores)
            worst = min(scores)
            # Significant divergence: best > 0 improvement but worst shows degradation or no improvement
            if best > worst and (best - worst) > 0.01:
                contradictions.append({
                    "surface": surface,
                    "vector": vector,
                    "exp_ids": [e.id for e in group],
                    "scores": scores,
                    "description": (
                        f"{surface}/{vector}: scores diverge ({worst:.4f} to {best:.4f}). "
                        f"{len(group)} experiments, range={best-worst:.4f}. "
                        f"Both findings preserved for investigation."
                    ),
                })

    return contradictions


_TRIAGE_ORDER = {"ACT_NOW": 0, "RESEARCH_MORE": 1, "PARK": 2, "KILL": 3}


def triage_experiment(exp: Experiment, project: ResearchProject) -> str:
    """WI-2C: Classify experiment urgency (OB1 pattern: panning-for-gold triage).

    Returns ACT_NOW | RESEARCH_MORE | PARK | KILL.
    """
    surface_completed = sum(
        1 for e in project.experiments
        if e.status == "completed" and e.surface == exp.surface
    )
    # Gather improvement data for this surface
    surface_improvements = [
        e.results.get("best_score", -np.inf) - project.best_score
        for e in project.experiments
        if e.status == "completed" and e.surface == exp.surface
        and isinstance(e.results.get("best_score"), (int, float))
    ]
    all_negligible = all(imp < 0.001 for imp in surface_improvements) if surface_improvements else False
    has_contradictions = len(project.contradictions) > 0 and any(
        c.get("surface") == exp.surface for c in project.contradictions
    )

    # KILL: surface exhausted with no improvement
    if surface_completed >= 3 and all_negligible and exp.expected_impact < 0.1:
        return "KILL"

    # ACT_NOW: high impact, affordable, surface not exhausted
    cost_by_surface: dict[str, float] = {}
    for s in project.surfaces:
        avg = np.mean([v.cost_estimate for v in s.vectors]) if s.vectors else 2.0
        cost_by_surface[s.id] = avg
    cost = cost_by_surface.get(exp.surface, 2.0)
    if exp.expected_impact > 0.2 and cost < 3.0 and surface_completed < 3:
        return "ACT_NOW"

    # RESEARCH_MORE: mixed results or contradictions on this surface
    if exp.expected_impact > 0.15 and (has_contradictions or (surface_improvements and not all_negligible)):
        return "RESEARCH_MORE"

    return "PARK"


def phase3_prioritize(
    experiments: list[Experiment],
    project: ResearchProject,
) -> list[Experiment]:
    """Phase 3: Triage + rank experiments.

    WI-2C: assigns triage category, filters KILL experiments, sorts by
    triage tier then priority_score within each tier.
    """
    if not experiments:
        return experiments

    # Count completed experiments per surface for diversity bonus
    surface_counts: dict[str, int] = {}
    for e in project.experiments:
        if e.status == "completed":
            surface_counts[e.surface] = surface_counts.get(e.surface, 0) + 1

    total_completed = sum(surface_counts.values())
    gap_to_threshold = max(0, project.stopping_threshold - project.best_score)

    # Cost lookup by surface
    cost_by_surface: dict[str, float] = {}
    for s in project.surfaces:
        avg_cost = np.mean([v.cost_estimate for v in s.vectors]) if s.vectors else 2.0
        cost_by_surface[s.id] = avg_cost

    for exp in experiments:
        impact = exp.expected_impact

        # Diversity bonus: surfaces with fewer completed experiments get boost
        surface_done = surface_counts.get(exp.surface, 0)
        if total_completed > 0:
            diversity_bonus = max(0, 0.3 * (1.0 - surface_done / max(1, total_completed)))
        else:
            diversity_bonus = 0.15  # small boost for first iteration

        # Gap bonus: larger gap to threshold = more urgency for high-impact experiments
        gap_bonus = min(0.3, gap_to_threshold * impact)

        cost = cost_by_surface.get(exp.surface, 2.0)
        cost = max(0.5, cost)  # avoid division by very small numbers

        exp.priority_score = round((impact + diversity_bonus + gap_bonus) / cost, 4)

        # WI-2C: assign triage
        exp.triage = triage_experiment(exp, project)

    # Filter KILL experiments (mark as skipped)
    killed = [e for e in experiments if e.triage == "KILL"]
    for e in killed:
        e.status = "skipped"
    active = [e for e in experiments if e.triage != "KILL"]

    # Sort: triage tier first (ACT_NOW > RESEARCH_MORE > PARK), then priority_score within
    active.sort(key=lambda e: (_TRIAGE_ORDER.get(e.triage, 2), -e.priority_score))

    if killed:
        log.info("Phase3: killed %d exhausted experiments", len(killed))

    log.info("Phase3: prioritized %d experiments (top: %s [%s] score=%.3f)",
             len(active),
             active[0].hypothesis[:50] if active else "none",
             active[0].triage if active else "?",
             active[0].priority_score if active else 0)
    return active + killed  # killed at the end for logging/reporting


# ===========================================================================
# Phase 4: Execute experiments with stopping criteria
# ===========================================================================
def phase4_execute(
    project: ResearchProject,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    experiments: list[Experiment],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: str | Path | None,
    anthropic_api_key: str | None,
) -> dict:
    """Phase 4: Execute experiments in priority order with stopping criteria.

    Returns dict with: trials, best_trial, stop_reason, experiments_completed.
    """
    from hooks.ml_experiment import (
        run_single_trial,
        build_extended_model_roster,
        build_voting_ensemble,
        build_stacking_ensemble,
        run_kfold_evaluation,
        run_bootstrap_ci,
        compute_shap_summary,
        compute_permutation_importance,
        extract_feature_importances,
        apply_feature_mutations,
        evaluate,
        evaluate_regression,
        evaluate_multiclass,
    )
    from sklearn.model_selection import train_test_split

    trials: list[dict] = []
    improvement_scores: list[float] = []
    start_time = time.time()
    trial_count = 0
    best_score = project.best_score
    best_trial_info: dict = {}
    best_model_obj: Any = None

    budget = project.budget
    max_trials = budget.get("max_trials", 30)
    max_seconds = budget.get("max_seconds", 600)

    task_type = project.task_type
    primary_metric = project.primary_metric
    is_classification = task_type in ("binary", "multiclass")

    # Collect all roster entries from existing trials for ensembles
    completed_entries: list[dict] = []

    for exp in experiments:
        # Check stopping criteria
        stop = _check_stopping(
            best_score, project.stopping_threshold, trial_count, max_trials,
            time.time() - start_time, max_seconds, improvement_scores,
            llm_budget_usd=budget.get("llm_budget_usd", 0.0),
        )
        if stop:
            exp.status = "skipped"
            project.stop_reason = stop
            log.info("Phase4: stopping — %s", stop)
            break

        exp.status = "running"
        log.info("Phase4: executing [%s/%s] %s", exp.surface, exp.vector, exp.hypothesis[:60])

        try:
            result = _execute_single_experiment(
                exp, project, X, y, feature_names,
                X_train, X_test, y_train, y_test,
                task_type, primary_metric, is_classification,
                completed_entries, out_dir,
            )

            exp.status = "completed"
            exp.results = result

            # Track trials from this experiment
            exp_trials = result.get("trials", [])
            trials.extend(exp_trials)
            trial_count += len(exp_trials)

            # Update best
            exp_best_score = result.get("best_score", -np.inf)
            if exp_best_score > best_score:
                improvement = exp_best_score - best_score
                improvement_scores.append(improvement)
                best_score = exp_best_score
                best_trial_info = result.get("best_trial", {})
                best_model_obj = result.get("best_model", None)
                project.best_score = best_score
                project.best_model = best_trial_info.get("model", "")
                log.info("Phase4: new best %.4f (+%.4f) from %s",
                         best_score, improvement, exp.surface)
            else:
                improvement_scores.append(0.0)

            # Track entry for ensembles
            if result.get("model_entry"):
                completed_entries.append(result["model_entry"])

        except Exception as exc:
            log.error("Phase4: experiment failed: %s\n%s", exc, traceback.format_exc())
            exp.status = "completed"
            exp.results = {"error": str(exc)}

    # Final stop check
    if not project.stop_reason:
        if trial_count >= max_trials:
            project.stop_reason = "budget_exhausted"
        elif all(e.status in ("completed", "skipped") for e in experiments):
            project.stop_reason = "all_experiments_done"
        else:
            project.stop_reason = "iteration_complete"

    return {
        "trials": trials,
        "best_trial": best_trial_info,
        "best_score": best_score,
        "best_model": best_model_obj,
        "stop_reason": project.stop_reason,
        "experiments_completed": sum(1 for e in experiments if e.status == "completed"),
        "total_time_s": round(time.time() - start_time, 2),
    }


def _check_stopping(
    best_score: float,
    threshold: float,
    trial_count: int,
    max_trials: int,
    elapsed_s: float,
    max_seconds: float,
    improvement_scores: list[float],
    llm_budget_usd: float = 0.0,
) -> str | None:
    """Check stopping criteria. Returns reason string or None."""
    if best_score >= threshold and threshold > 0:
        return "threshold_met"
    if trial_count >= max_trials:
        return "budget_exhausted"
    if elapsed_s >= max_seconds:
        return "budget_exhausted"
    # WI-3A: LLM token budget check
    if llm_budget_usd > 0:
        try:
            from hooks.llm_client import check_budget
            if not check_budget(llm_budget_usd):
                return "llm_budget_exhausted"
        except ImportError:
            pass
    # Plateau: need at least 5 experiments before checking, then last 4 must all
    # show < 0.001 improvement. This prevents early stopping before high-impact
    # experiments (Optuna, ensembles) get a chance to run.
    if len(improvement_scores) >= 5:
        last4 = improvement_scores[-4:]
        if all(imp < 0.001 for imp in last4):
            return "plateau_detected"
    return None


def _execute_single_experiment(
    exp: Experiment,
    project: ResearchProject,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    primary_metric: str,
    is_classification: bool,
    completed_entries: list[dict],
    out_dir: str | Path | None,
) -> dict:
    """Execute a single experiment based on its surface and design."""
    from hooks.ml_experiment import (
        run_single_trial,
        build_extended_model_roster,
        build_voting_ensemble,
        build_stacking_ensemble,
        run_kfold_evaluation,
        run_bootstrap_ci,
        compute_shap_summary,
        compute_permutation_importance,
        apply_feature_mutations,
        extract_feature_importances,
        _build_model_roster_binary,
        _build_model_roster_regression,
    )

    design = exp.design
    design_type = design.get("type", "")

    # ----- Models surface -----
    if exp.surface == "models" and design_type == "extended_roster":
        roster = build_extended_model_roster(task_type)
        trials = []
        best_score = -np.inf
        best_trial = {}
        best_entry = None
        for entry in roster:
            result = run_single_trial(
                entry, X_train, X_test, y_train, y_test,
                feature_names, task_type, primary_metric,
            )
            trials.append(result)
            if result["score"] > best_score:
                best_score = result["score"]
                best_trial = result
                best_entry = entry
        return {"trials": trials, "best_score": best_score,
                "best_trial": best_trial, "model_entry": best_entry}

    # ----- Features surface: interaction -----
    if exp.surface == "features" and design_type == "interaction":
        col_a = design.get("col_a", "")
        col_b = design.get("col_b", "")
        mutations = [{"type": "interaction", "col_a": col_a, "col_b": col_b}]
        return _run_with_mutations(
            mutations, X, y, feature_names, X_train, X_test, y_train, y_test,
            task_type, primary_metric, is_classification, project,
        )

    # ----- Features surface: ratio -----
    if exp.surface == "features" and design_type == "ratio":
        num = design.get("numerator", "")
        den = design.get("denominator", "")
        mutations = [{"type": "ratio", "numerator": num, "denominator": den}]
        return _run_with_mutations(
            mutations, X, y, feature_names, X_train, X_test, y_train, y_test,
            task_type, primary_metric, is_classification, project,
        )

    # ----- Features surface: log transforms -----
    if exp.surface == "features" and design_type in ("log", "log_top_features"):
        features = design.get("features", [])
        mutations = [{"type": "log", "col": f} for f in features[:5]]
        return _run_with_mutations(
            mutations, X, y, feature_names, X_train, X_test, y_train, y_test,
            task_type, primary_metric, is_classification, project,
        )

    # ----- Architectures surface: voting -----
    if exp.surface == "architectures" and design_type == "voting":
        if is_classification:
            base_roster = _build_model_roster_binary()
        else:
            base_roster = _build_model_roster_regression()
        # Use base roster entries (or completed entries if available)
        models_for_ensemble = completed_entries[:5] if len(completed_entries) >= 2 else base_roster[:3]
        voting_type = design.get("voting", "soft")
        entry = build_voting_ensemble(models_for_ensemble, task_type, voting_type)
        result = run_single_trial(
            entry, X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric,
        )
        return {"trials": [result], "best_score": result["score"],
                "best_trial": result, "model_entry": entry}

    # ----- Architectures surface: stacking -----
    if exp.surface == "architectures" and design_type == "stacking":
        if is_classification:
            base_roster = _build_model_roster_binary()
        else:
            base_roster = _build_model_roster_regression()
        base = completed_entries[:4] if len(completed_entries) >= 2 else base_roster[:3]
        meta = base_roster[0]  # LR/Ridge as meta-learner
        entry = build_stacking_ensemble(base, meta, task_type)
        result = run_single_trial(
            entry, X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric,
        )
        return {"trials": [result], "best_score": result["score"],
                "best_trial": result, "model_entry": entry}

    # ----- Data surface: SMOTE -----
    if exp.surface == "data" and design_type == "smote":
        return _run_smote_experiment(
            X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric, project,
        )

    # ----- Data surface: KNN impute -----
    if exp.surface == "data" and design_type == "knn_impute":
        return _run_knn_impute_experiment(
            X, y, feature_names, design.get("columns", []),
            task_type, primary_metric, is_classification, project,
        )

    # ----- Validation surface: K-fold -----
    if exp.surface == "validation" and design_type == "kfold":
        if is_classification:
            base_roster = _build_model_roster_binary()
        else:
            base_roster = _build_model_roster_regression()
        # Run kfold on best model from roster
        model_name = design.get("model", "")
        entry = next((e for e in base_roster if e["name"] == model_name), base_roster[0])
        n_splits = design.get("n_splits", 5)
        kfold_result = run_kfold_evaluation(entry, X, y, feature_names,
                                             task_type, primary_metric, n_splits)
        return {"trials": [], "best_score": kfold_result["mean_score"],
                "best_trial": {"model": kfold_result["model"],
                               "metrics": {primary_metric: kfold_result["mean_score"]}},
                "kfold": kfold_result}

    # ----- Validation surface: bootstrap -----
    if exp.surface == "validation" and design_type == "bootstrap":
        if is_classification:
            base_roster = _build_model_roster_binary()
        else:
            base_roster = _build_model_roster_regression()
        entry = base_roster[0]
        n_boot = design.get("n_bootstrap", 200)
        boot_result = run_bootstrap_ci(entry, X_train, X_test, y_train, y_test,
                                        feature_names, task_type, primary_metric, n_boot)
        return {"trials": [], "best_score": boot_result["point_estimate"],
                "best_trial": {"model": boot_result["model"],
                               "metrics": {primary_metric: boot_result["point_estimate"]}},
                "bootstrap": boot_result}

    # ----- Interpretation surface: permutation -----
    if exp.surface == "interpretation" and design_type == "permutation":
        if is_classification:
            base_roster = _build_model_roster_binary()
        else:
            base_roster = _build_model_roster_regression()
        # Fit best model first
        import copy
        entry = base_roster[0]
        model = copy.deepcopy(entry["model"])
        model.fit(X_train, y_train)
        perm_imp = compute_permutation_importance(
            model, X_test, y_test, feature_names, task_type,
        )
        return {"trials": [], "best_score": -np.inf,
                "best_trial": {}, "permutation_importances": perm_imp}

    # ----- Interpretation surface: SHAP -----
    if exp.surface == "interpretation" and design_type == "shap":
        import copy
        if is_classification:
            base_roster = _build_model_roster_binary()
        else:
            base_roster = _build_model_roster_regression()
        entry = base_roster[0]
        model = copy.deepcopy(entry["model"])
        model.fit(X_train, y_train)
        shap_result = compute_shap_summary(model, X_test, feature_names, out_dir)
        return {"trials": [], "best_score": -np.inf,
                "best_trial": {}, "shap": shap_result}

    # ----- Interpretation surface: threshold tuning -----
    if exp.surface == "interpretation" and design_type == "threshold_tuning":
        return _run_threshold_tuning(
            X_train, X_test, y_train, y_test, feature_names,
            task_type, primary_metric, project,
        )

    # ----- Parameters surface: Optuna -----
    if exp.surface == "parameters" and design_type == "optuna":
        return _run_optuna_experiment(
            X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric,
            design.get("model", ""), design.get("n_trials", 15),
        )

    # ----- Parameters surface: grid search -----
    if exp.surface == "parameters" and design_type == "grid_search":
        return _run_grid_search_experiment(
            X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric, project,
        )

    # ----- Generic LLM-designed experiment (feature mutations + model) -----
    # If it has mutation-like fields, treat as feature mutation + best model
    mutations = []
    if design_type == "ratio" and "numerator" in design:
        mutations = [{"type": "ratio", "numerator": design["numerator"],
                      "denominator": design["denominator"]}]
    elif design_type == "interaction" and "col_a" in design:
        mutations = [{"type": "interaction", "col_a": design["col_a"],
                      "col_b": design["col_b"]}]
    elif design_type == "bin" and "col" in design:
        mutations = [{"type": "bin", "col": design["col"],
                      "bins": design.get("bins", 5)}]
    elif design_type == "log" and "col" in design:
        mutations = [{"type": "log", "col": design["col"]}]
    elif design_type == "drop" and "cols" in design:
        mutations = [{"type": "drop", "cols": design["cols"]}]

    if mutations:
        return _run_with_mutations(
            mutations, X, y, feature_names, X_train, X_test, y_train, y_test,
            task_type, primary_metric, is_classification, project,
        )

    # Fallback: run best model from base roster as-is
    log.warning("Phase4: unrecognized experiment type '%s' — running base model", design_type)
    if is_classification:
        base_roster = _build_model_roster_binary()
    else:
        base_roster = _build_model_roster_regression()
    entry = base_roster[0]
    result = run_single_trial(
        entry, X_train, X_test, y_train, y_test,
        feature_names, task_type, primary_metric,
    )
    return {"trials": [result], "best_score": result["score"],
            "best_trial": result, "model_entry": entry}


def _find_best_model_entry(
    project: ResearchProject,
    task_type: str,
) -> dict:
    """Find the best model entry by checking base + extended rosters.

    Falls back to base roster[0] if no match found.
    """
    from hooks.ml_experiment import (
        _build_model_roster_binary,
        _build_model_roster_regression,
        build_extended_model_roster,
    )
    is_classification = task_type in ("binary", "multiclass")
    if is_classification:
        roster = _build_model_roster_binary()
    else:
        roster = _build_model_roster_regression()

    # Also include extended roster for matching
    try:
        extended = build_extended_model_roster(task_type)
        roster = roster + extended
    except Exception:
        pass

    best_name = project.best_model
    if not best_name:
        return roster[0]

    # Exact match
    entry = next((e for e in roster if e["name"] == best_name), None)
    if entry:
        return entry

    # Fuzzy match: strip version/trial suffixes like "_v2", "_t5", "Optuna_"
    import re
    clean_name = re.sub(r'(_v\d+|_t\d+|^Optuna_)', '', best_name)
    entry = next((e for e in roster if e["name"] == clean_name), None)
    if entry:
        return entry

    # Abbreviation map for common model name mismatches
    abbrev_map = {"LGB": "LightGBM", "XGB": "XGBoost", "RF": "RandomForest",
                  "CB": "CatBoost", "LR": "LogisticRegression", "GBM": "GradientBoosting"}
    expanded = abbrev_map.get(clean_name, clean_name)
    if expanded != clean_name:
        entry = next((e for e in roster if e["name"] == expanded
                      or e["name"].startswith(expanded)), None)
        if entry:
            return entry

    # Substring match
    entry = next((e for e in roster if clean_name in e["name"] or e["name"] in clean_name), None)
    if entry:
        return entry

    log.info("Model '%s' not found in roster, using %s", best_name, roster[0]["name"])
    return roster[0]


def _run_with_mutations(
    mutations: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    primary_metric: str,
    is_classification: bool,
    project: ResearchProject,
) -> dict:
    """Apply feature mutations and run best model on mutated features."""
    from hooks.ml_experiment import (
        apply_feature_mutations,
        run_single_trial,
    )
    from sklearn.model_selection import train_test_split

    try:
        X_mut, mut_names, mut_descs = apply_feature_mutations(X, y, mutations, feature_names)
    except Exception as exc:
        log.warning("Feature mutations failed: %s", exc)
        return {"trials": [], "best_score": -np.inf, "best_trial": {}}

    # Re-split with same random state
    if is_classification:
        Xt, Xv, yt, yv = train_test_split(X_mut, y, test_size=0.20, stratify=y, random_state=42)
    else:
        Xt, Xv, yt, yv = train_test_split(X_mut, y, test_size=0.20, random_state=42)

    # Find best model from combined base + extended roster
    entry = _find_best_model_entry(project, task_type)

    result = run_single_trial(entry, Xt, Xv, yt, yv, mut_names, task_type, primary_metric)
    result["description"] = f"{result.get('description', '')} + {', '.join(mut_descs[:3])}"
    return {"trials": [result], "best_score": result["score"],
            "best_trial": result, "model_entry": entry,
            "mutations": mut_descs}


def _run_smote_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    task_type: str,
    primary_metric: str,
    project: ResearchProject,
) -> dict:
    """Run SMOTE oversampling experiment."""
    from hooks.ml_experiment import run_single_trial

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log.info("imbalanced-learn not installed — skipping SMOTE")
        return {"trials": [], "best_score": -np.inf, "best_trial": {},
                "error": "imbalanced-learn not installed"}

    sm = SMOTE(random_state=42)
    try:
        X_res, y_res = sm.fit_resample(X_train, y_train)
    except Exception as exc:
        log.warning("SMOTE failed: %s", exc)
        return {"trials": [], "best_score": -np.inf, "best_trial": {}}

    entry = _find_best_model_entry(project, task_type)

    result = run_single_trial(entry, X_res, X_test, pd.Series(y_res), y_test,
                               feature_names, task_type, primary_metric)
    result["description"] = f"SMOTE + {result.get('description', '')}"
    return {"trials": [result], "best_score": result["score"],
            "best_trial": result, "model_entry": entry}


def _run_knn_impute_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    columns: list[str],
    task_type: str,
    primary_metric: str,
    is_classification: bool,
    project: ResearchProject,
) -> dict:
    """Run KNN imputation experiment on specified columns."""
    from hooks.ml_experiment import (
        run_single_trial, _build_model_roster_binary, _build_model_roster_regression,
    )
    from sklearn.model_selection import train_test_split

    try:
        from sklearn.impute import KNNImputer
    except ImportError:
        return {"trials": [], "best_score": -np.inf, "best_trial": {}}

    X_imp = X.copy()
    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = X_imp.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        X_imp[numeric_cols] = imputer.fit_transform(X_imp[numeric_cols])

    if is_classification:
        Xt, Xv, yt, yv = train_test_split(X_imp, y, test_size=0.20, stratify=y, random_state=42)
        roster = _build_model_roster_binary()
    else:
        Xt, Xv, yt, yv = train_test_split(X_imp, y, test_size=0.20, random_state=42)
        roster = _build_model_roster_regression()

    best_name = project.best_model
    entry = next((e for e in roster if e["name"] == best_name), roster[0])
    feat_names = X_imp.columns.tolist()

    result = run_single_trial(entry, Xt, Xv, yt, yv, feat_names, task_type, primary_metric)
    result["description"] = f"KNN-impute + {result.get('description', '')}"
    return {"trials": [result], "best_score": result["score"],
            "best_trial": result, "model_entry": entry}


def _run_threshold_tuning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    task_type: str,
    primary_metric: str,
    project: ResearchProject,
) -> dict:
    """Find optimal classification threshold."""
    if task_type != "binary":
        return {"trials": [], "best_score": -np.inf, "best_trial": {},
                "note": "threshold tuning only for binary classification"}

    import copy
    from hooks.ml_experiment import (
        evaluate, _build_model_roster_binary,
    )

    roster = _build_model_roster_binary()
    best_name = project.best_model
    entry = next((e for e in roster if e["name"] == best_name), roster[0])

    model = copy.deepcopy(entry["model"])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    best_thresh = 0.5
    best_f1 = -np.inf
    results_by_threshold: list[dict] = []
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        metrics = evaluate(f"{entry['name']}_t{thresh:.2f}", y_test.values, y_prob, y_pred)
        score = metrics.get(primary_metric, -np.inf)
        results_by_threshold.append({"threshold": round(float(thresh), 2),
                                      "score": round(score, 4)})
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return {"trials": [], "best_score": best_f1,
            "best_trial": {"model": entry["name"],
                           "metrics": {primary_metric: round(best_f1, 4)}},
            "optimal_threshold": round(float(best_thresh), 2),
            "threshold_results": results_by_threshold}


def _run_optuna_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    task_type: str,
    primary_metric: str,
    model_name: str,
    n_trials: int,
) -> dict:
    """Run Optuna Bayesian optimization with expanded search space."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.info("Optuna not installed — skipping Bayesian optimization")
        return {"trials": [], "best_score": -np.inf, "best_trial": {},
                "error": "optuna not installed"}

    from hooks.ml_experiment import run_single_trial

    is_classification = task_type in ("binary", "multiclass")
    trials: list[dict] = []
    best_score = -np.inf
    best_trial_info: dict = {}
    best_entry: dict = {}

    # Determine which model families are available
    has_lgb = _has_lightgbm()
    has_xgb = _has_xgboost()
    has_catboost = False
    try:
        import catboost  # noqa: F401
        has_catboost = True
    except ImportError:
        pass

    # Build model choice list based on availability + hint from model_name
    model_choices = []
    if has_lgb:
        model_choices.append("LightGBM")
    if has_xgb:
        model_choices.append("XGBoost")
    if has_catboost:
        model_choices.append("CatBoost")
    model_choices.append("RandomForest")  # always available

    # If model_name hints at a specific model, put it first
    import re
    clean = re.sub(r'(_v\d+|^Optuna_)', '', model_name)
    for mc in model_choices:
        if clean.lower() in mc.lower() or mc.lower() in clean.lower():
            model_choices.remove(mc)
            model_choices.insert(0, mc)
            break

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score, best_trial_info, best_entry

        # Common hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 100, 600)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

        # Model selection — Optuna explores across model families
        if len(model_choices) > 1:
            chosen_model = trial.suggest_categorical("model_type", model_choices)
        else:
            chosen_model = model_choices[0]

        try:
            if chosen_model == "LightGBM":
                import lightgbm as lgb
                num_leaves = trial.suggest_int("num_leaves", 15, 127)
                min_child = trial.suggest_int("min_child_samples", 5, 100)
                colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                params = dict(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth, num_leaves=num_leaves,
                    subsample=subsample, colsample_bytree=colsample,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    min_child_samples=min_child,
                    random_state=42, verbose=-1, n_jobs=-1,
                )
                if is_classification:
                    params["is_unbalance"] = True
                    model = lgb.LGBMClassifier(**params)
                else:
                    model = lgb.LGBMRegressor(**params)
                name = "Optuna_LGB"

            elif chosen_model == "XGBoost":
                import xgboost as xgb
                colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
                min_child_w = trial.suggest_float("min_child_weight", 1, 20, log=True)
                params = dict(
                    n_estimators=n_estimators, learning_rate=learning_rate,
                    max_depth=max_depth, subsample=subsample,
                    colsample_bytree=colsample,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    min_child_weight=min_child_w,
                    random_state=42, n_jobs=-1, verbosity=0,
                )
                if is_classification:
                    params["eval_metric"] = "aucpr"
                    params["use_label_encoder"] = False
                    model = xgb.XGBClassifier(**params)
                else:
                    model = xgb.XGBRegressor(**params)
                name = "Optuna_XGB"

            elif chosen_model == "CatBoost":
                import catboost as cb
                params = dict(
                    iterations=n_estimators, learning_rate=learning_rate,
                    depth=min(max_depth, 10),  # CatBoost max depth is 16 but 10 is practical
                    subsample=subsample,
                    l2_leaf_reg=reg_lambda,
                    random_seed=42, verbose=0,
                )
                if is_classification:
                    params["auto_class_weights"] = "Balanced"
                    model = cb.CatBoostClassifier(**params)
                else:
                    model = cb.CatBoostRegressor(**params)
                name = "Optuna_CB"

            else:  # RandomForest
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
                max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
                params = dict(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42, n_jobs=-1,
                )
                if is_classification:
                    params["class_weight"] = "balanced"
                    model = RandomForestClassifier(**params)
                else:
                    model = RandomForestRegressor(**params)
                name = "Optuna_RF"

        except Exception:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    random_state=42, n_jobs=-1,
                )
            name = "Optuna_RF"

        entry = {
            "name": f"{name}_t{trial.number}",
            "model": model,
            "description": (f"Optuna trial {trial.number}: {chosen_model} "
                            f"n={n_estimators} lr={learning_rate:.4f} depth={max_depth} "
                            f"subsample={subsample:.2f} reg_a={reg_alpha:.2e} reg_l={reg_lambda:.2e}"),
        }

        result = run_single_trial(
            entry, X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric,
        )
        trials.append(result)

        score = result["score"]
        if score > best_score:
            best_score = score
            best_trial_info = result
            best_entry = entry

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=min(n_trials, 20), show_progress_bar=False)

    return {"trials": trials, "best_score": best_score,
            "best_trial": best_trial_info, "model_entry": best_entry,
            "optuna_best_params": study.best_params if study.best_trial else {}}


def _run_grid_search_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    task_type: str,
    primary_metric: str,
    project: ResearchProject,
) -> dict:
    """Run targeted grid search on best model's hyperparameters."""
    from hooks.ml_experiment import run_single_trial

    is_classification = task_type in ("binary", "multiclass")
    trials: list[dict] = []
    best_score = -np.inf
    best_trial_info: dict = {}
    best_entry: dict = {}

    # Small grid of common hyperparams
    param_grid = [
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.03},
        {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.01},
        {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1},
    ]

    for i, params in enumerate(param_grid):
        try:
            if _has_lightgbm():
                import lightgbm as lgb
                if is_classification:
                    model = lgb.LGBMClassifier(
                        is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1,
                        **params,
                    )
                else:
                    model = lgb.LGBMRegressor(
                        random_state=42, verbose=-1, n_jobs=-1, **params,
                    )
                name = f"Grid_LGB_{i}"
            else:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                rf_params = {k: v for k, v in params.items() if k != "learning_rate"}
                if is_classification:
                    model = RandomForestClassifier(
                        class_weight="balanced", random_state=42, n_jobs=-1,
                        **rf_params,
                    )
                else:
                    model = RandomForestRegressor(
                        random_state=42, n_jobs=-1, **rf_params,
                    )
                name = f"Grid_RF_{i}"
        except Exception:
            continue

        desc = " ".join(f"{k}={v}" for k, v in params.items())
        entry = {"name": name, "model": model, "description": f"Grid search: {desc}"}
        result = run_single_trial(
            entry, X_train, X_test, y_train, y_test,
            feature_names, task_type, primary_metric,
        )
        trials.append(result)
        if result["score"] > best_score:
            best_score = result["score"]
            best_trial_info = result
            best_entry = entry

    return {"trials": trials, "best_score": best_score,
            "best_trial": best_trial_info, "model_entry": best_entry}


def _has_lightgbm() -> bool:
    try:
        import lightgbm
        return True
    except ImportError:
        return False


def _has_xgboost() -> bool:
    try:
        import xgboost
        return True
    except ImportError:
        return False


# ===========================================================================
# Stopping threshold computation
# ===========================================================================
def compute_stopping_threshold(
    baseline_metrics: dict,
    task_type: str,
    domain_ctx: dict | None,
    primary_metric: str,
    api_key: str | None,
) -> tuple[float, str]:
    """Compute stopping threshold as blend of LLM reasoning (60%) + heuristic (40%).

    Returns (threshold, rationale_string).
    """
    baseline_score = baseline_metrics.get(primary_metric, 0)

    # Heuristic threshold
    heuristic = _heuristic_threshold(baseline_score, task_type, primary_metric)
    rationale = f"Heuristic: baseline={baseline_score:.4f}, target={heuristic:.4f}"

    # Try LLM for informed threshold
    if api_key:
        llm_result = _llm_threshold(baseline_score, task_type, primary_metric,
                                     domain_ctx, api_key)
        if llm_result is not None:
            # Blend: 60% LLM, 40% heuristic
            blended = 0.6 * llm_result[0] + 0.4 * heuristic
            rationale = f"Blended (60% LLM + 40% heuristic): {blended:.4f}. LLM: {llm_result[1]}"
            return round(blended, 4), rationale

    return round(heuristic, 4), rationale


def _heuristic_threshold(baseline: float, task_type: str, metric: str) -> float:
    """Compute heuristic stopping threshold based on task type and baseline."""
    if metric == "PR_AUC":
        return min(0.85, baseline + max(0.10, (0.60 - baseline) * 0.4))
    elif metric == "ROC_AUC":
        return min(0.95, baseline + max(0.05, (0.90 - baseline) * 0.3))
    elif metric == "Macro_F1":
        return min(0.90, baseline + max(0.08, (0.80 - baseline) * 0.3))
    elif metric == "R2":
        return min(0.95, baseline + max(0.05, (0.85 - baseline) * 0.3))
    elif metric == "MCC":
        return min(0.80, baseline + max(0.10, (0.60 - baseline) * 0.4))
    else:
        # Generic: aim for 15% improvement over baseline
        return baseline + max(0.05, baseline * 0.15)


def _llm_threshold(
    baseline: float,
    task_type: str,
    metric: str,
    domain_ctx: dict | None,
    api_key: str,
) -> tuple[float, str] | None:
    """Ask LLM to reason about appropriate stopping threshold."""
    try:
        from hooks.llm_client import call_llm
    except ImportError:
        return None

    domain = domain_ctx.get("domain", "General") if domain_ctx else "General"
    prompt = f"""You are an ML research advisor. Given:
- Task: {task_type}
- Primary metric: {metric}
- Baseline score: {baseline:.4f}
- Domain: {domain}

What is a realistic and ambitious stopping threshold for this metric?
Consider:
1. Domain benchmarks (what's typical for {domain} {task_type} tasks)
2. Diminishing returns (when is further improvement not worth the compute)
3. The gap between baseline and theoretical maximum

Reply with ONLY a JSON object: {{"threshold": <float>, "reasoning": "<one sentence>"}}"""

    response = call_llm(prompt, max_tokens=200, expect_json=True)
    if not response:
        return None

    try:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        thresh = float(data["threshold"])
        reasoning = str(data.get("reasoning", "LLM-suggested"))
        # Sanity clamp
        thresh = max(baseline + 0.01, min(1.0, thresh))
        return thresh, reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def adapt_stopping_threshold(
    project: ResearchProject,
    improvement_history: list[dict],
) -> tuple[float, str]:
    """Adapt stopping threshold based on improvement trajectory.

    Raise threshold if rapid improvement; lower if plateauing.
    Returns (new_threshold, rationale).
    """
    current = project.stopping_threshold
    if not improvement_history:
        return current, "No history yet"

    recent_improvements = [h.get("improvement", 0) for h in improvement_history[-3:]]
    avg_improvement = np.mean(recent_improvements) if recent_improvements else 0

    if avg_improvement > 0.02:
        # Rapid improvement — raise threshold
        new_thresh = min(1.0, current + 0.03)
        rationale = f"Rapid improvement (avg +{avg_improvement:.4f}), raising threshold"
    elif avg_improvement < 0.002 and len(improvement_history) >= 2:
        # Plateauing — lower threshold slightly
        new_thresh = max(project.best_score + 0.005, current - 0.02)
        rationale = f"Plateauing (avg +{avg_improvement:.4f}), lowering threshold"
    else:
        new_thresh = current
        rationale = "Threshold unchanged"

    return round(new_thresh, 4), rationale


# ===========================================================================
# WI-2D: Strategy approval gate (OB1 pattern: life-engine self-improvement)
# ===========================================================================

def _propose_strategy_change(
    project: ResearchProject,
    improvement_history: list[dict],
    anthropic_api_key: str | None = None,
) -> dict | None:
    """Propose ONE specific strategy change based on improvement trajectory.

    Called every 2nd iteration. Returns a proposal dict or None if no change needed.

    Proposal types:
      - raise_budget: increase trial count or time budget
      - switch_surface: shift focus to a different research surface
      - add_custom_experiment: suggest a specific experiment not in the template set
      - lower_threshold: reduce stopping threshold if it seems unreachable
    """
    if len(improvement_history) < 2:
        return None

    # Analyze recent trajectory
    recent = improvement_history[-3:] if len(improvement_history) >= 3 else improvement_history
    improvements = [h.get("improvement", 0) for h in recent]
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    best_score = project.best_score
    gap = project.stopping_threshold - best_score

    # Count experiments per surface
    surface_scores: dict[str, list[float]] = {}
    for e in project.experiments:
        if e.status == "completed" and e.results.get("best_score") is not None:
            score = e.results["best_score"]
            if isinstance(score, (int, float)):
                surface_scores.setdefault(e.surface, []).append(score)

    # Heuristic proposals
    proposal = None

    # If stalling (avg improvement < 0.002) and gap is large
    if avg_improvement < 0.002 and gap > 0.1:
        # Find the surface with the best average score improvement
        best_surface = None
        best_avg = -np.inf
        for surface, scores in surface_scores.items():
            if len(scores) >= 2:
                avg = np.mean(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_surface = surface

        if best_surface and best_surface != "parameters":
            proposal = {
                "change_type": "switch_surface",
                "description": f"Focus next iteration on '{best_surface}' surface (avg score={best_avg:.4f})",
                "rationale": f"Recent improvement averaging {avg_improvement:.4f} per iteration. "
                             f"'{best_surface}' surface showed best average results.",
                "risk": "low",
            }
        else:
            # Suggest raising budget since we're stalling
            current_trials = project.budget.get("max_trials", 30)
            proposal = {
                "change_type": "raise_budget",
                "description": f"Increase max_trials from {current_trials} to {int(current_trials * 1.5)}",
                "rationale": f"Improvement stalled at {avg_improvement:.4f}/iter with gap={gap:.4f} to threshold. "
                             f"More trials may help Optuna find better hyperparameters.",
                "risk": "medium",
                "new_value": int(current_trials * 1.5),
            }

    # If threshold seems unreachable (gap > 3x best improvement seen)
    elif gap > 0 and avg_improvement > 0:
        iters_needed = gap / max(avg_improvement, 0.001)
        remaining = project.budget.get("max_trials", 30) - len(project.all_trials)
        if iters_needed > remaining * 2:
            proposal = {
                "change_type": "lower_threshold",
                "description": f"Lower stopping threshold from {project.stopping_threshold:.4f} to {best_score + gap * 0.5:.4f}",
                "rationale": f"At current rate ({avg_improvement:.4f}/iter), reaching threshold would take "
                             f"~{iters_needed:.0f} more experiment sets. Lowering threshold to a realistic target.",
                "risk": "low",
                "new_value": round(best_score + gap * 0.5, 4),
            }

    return proposal


def _apply_strategy_proposal(project: ResearchProject, proposal: dict) -> bool:
    """Apply a strategy proposal to the project. Returns True if applied."""
    change_type = proposal.get("change_type", "")

    if change_type == "raise_budget":
        new_val = proposal.get("new_value")
        if new_val and isinstance(new_val, int):
            project.budget["max_trials"] = new_val
            log.info("Strategy applied: max_trials raised to %d", new_val)
            return True

    elif change_type == "lower_threshold":
        new_val = proposal.get("new_value")
        if new_val and isinstance(new_val, (int, float)):
            project.stopping_threshold = float(new_val)
            project.threshold_rationale = f"Lowered by strategy proposal: {proposal['rationale'][:80]}"
            log.info("Strategy applied: threshold lowered to %.4f", new_val)
            return True

    elif change_type == "switch_surface":
        # This is informational — the next phase2_design will naturally use it
        log.info("Strategy proposal: %s (informational, no direct change)", proposal["description"])
        return True

    return False


# ===========================================================================
# Research project markdown
# ===========================================================================
def update_research_project_md(project: ResearchProject, out_dir: str | Path | None) -> str:
    """Generate/update the living research_project.md artifact.

    Returns path to the markdown file.
    """
    if not out_dir:
        return ""

    out_path = Path(out_dir) / "research_project.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append(f"# Research Project: {project.objective}")
    lines.append(f"\n_Last updated: {ts}_")
    lines.append(f"\n**Phase:** {project.phase} | **Iteration:** {project.iteration}")
    lines.append(f"**Task:** {project.task_type} | **Metric:** {project.primary_metric}")
    lines.append(f"**Domain:** {project.domain}")
    lines.append(f"**Best Score:** {project.best_score:.4f} ({project.best_model})")
    lines.append(f"**Stopping Threshold:** {project.stopping_threshold:.4f}")
    lines.append(f"**Stop Reason:** {project.stop_reason or 'running'}")

    # Phase 1: Understanding
    lines.append("\n## Phase 1: Understanding")
    lines.append(f"- Objective: {project.objective}")
    if project.baseline:
        lines.append(f"- Baseline: {project.baseline.get('incumbent_metric', 'N/A'):.4f}")
        lines.append(f"- Baseline model: {project.baseline.get('best_trial', {}).get('metrics', {}).get('Model', 'N/A')}")

    # Phase 2: Experiment Design
    lines.append("\n## Phase 2: Experiment Design")
    surface_counts: dict[str, int] = {}
    for e in project.experiments:
        surface_counts[e.surface] = surface_counts.get(e.surface, 0) + 1
    lines.append(f"- Total experiments designed: {len(project.experiments)}")
    lines.append(f"- Surfaces covered: {', '.join(f'{s}({c})' for s, c in sorted(surface_counts.items()))}")

    # Phase 3: Prioritized Queue
    lines.append("\n## Phase 3: Experiment Queue")
    lines.append("\n| # | Surface | Vector | Hypothesis | Priority | Status | Score |")
    lines.append("|---|---------|--------|-----------|----------|--------|-------|")
    for i, exp in enumerate(project.experiments[:20], 1):
        score_str = f"{exp.results.get('best_score', '')}" if exp.results.get("best_score") else ""
        hyp = exp.hypothesis[:50] + "..." if len(exp.hypothesis) > 50 else exp.hypothesis
        lines.append(
            f"| {i} | {exp.surface} | {exp.vector} | {hyp} | "
            f"{exp.priority_score:.3f} | {exp.status} | {score_str} |"
        )

    # Phase 4: Execution Log
    lines.append("\n## Phase 4: Execution")
    completed = [e for e in project.experiments if e.status == "completed"]
    lines.append(f"- Experiments completed: {len(completed)}/{len(project.experiments)}")
    lines.append(f"- Total trials: {len(project.all_trials)}")

    # Improvement history
    if project.improvement_history:
        lines.append("\n### Improvement Trajectory")
        for h in project.improvement_history:
            lines.append(f"- Iter {h.get('iteration', '?')}: "
                         f"{h.get('score', 0):.4f} (+{h.get('improvement', 0):.4f}) "
                         f"via {h.get('surface', '?')}")

    # Threshold history
    lines.append(f"\n### Stopping Threshold")
    lines.append(f"- Current: {project.stopping_threshold:.4f}")
    lines.append(f"- Rationale: {project.threshold_rationale}")

    # WI-2B: Contradictions
    if project.contradictions:
        lines.append("\n## Contradictions")
        lines.append("_Experiments with conflicting outcomes on the same surface/vector — both preserved._\n")
        for c in project.contradictions:
            lines.append(f"- **[{c.get('surface', '?')}]** {c.get('description', 'N/A')}")

    # WI-2D: Strategy proposals
    if project.strategy_proposals:
        lines.append("\n## Strategy Proposals")
        for i, p in enumerate(project.strategy_proposals, 1):
            status = "auto-approved" if p.get("auto_approved") else "queued for review"
            lines.append(f"- **[{p.get('change_type', '?')}]** {p.get('description', '')} ({status})")
            if p.get("rationale"):
                lines.append(f"  _Rationale: {p['rationale'][:120]}_")

    text = "\n".join(lines)
    text = text.encode("ascii", errors="replace").decode("ascii")
    out_path.write_text(text, encoding="utf-8")
    log.info("Research project MD updated: %s", out_path)
    return str(out_path)


# ===========================================================================
# Summary rendering (ASCII-safe)
# ===========================================================================
def render_research_summary(project: ResearchProject) -> str:
    """Render ASCII-safe summary of the research project."""
    lines: list[str] = []
    lines.append("## Research Project Summary")
    lines.append("")
    lines.append(f"**Objective:** {project.objective}")
    lines.append(f"**Result:** {project.primary_metric}={project.best_score:.4f} ({project.best_model})")
    lines.append(f"**Stop reason:** {project.stop_reason}")
    lines.append(f"**Iterations:** {project.iteration}")

    # Surface breakdown
    surface_results: dict[str, list[float]] = {}
    for exp in project.experiments:
        if exp.status == "completed" and exp.results.get("best_score", -np.inf) > -np.inf:
            surface_results.setdefault(exp.surface, []).append(exp.results["best_score"])

    if surface_results:
        lines.append("")
        lines.append("### Results by Surface")
        lines.append("| Surface | Experiments | Best Score | Avg Score |")
        lines.append("|---------|-------------|-----------|-----------|")
        for surface, scores in sorted(surface_results.items()):
            valid = [s for s in scores if s > -np.inf]
            best = max(valid) if valid else -np.inf
            avg = np.mean(valid) if valid else -np.inf
            lines.append(f"| {surface} | {len(scores)} | {best:.4f} | {avg:.4f} |")

    # Top experiments
    completed = [e for e in project.experiments if e.status == "completed"]
    completed.sort(key=lambda e: e.results.get("best_score", -np.inf), reverse=True)
    if completed:
        lines.append("")
        lines.append("### Top Experiments")
        for exp in completed[:5]:
            score = exp.results.get("best_score", -np.inf)
            lines.append(f"- [{exp.surface}] {exp.hypothesis[:60]} -> {score:.4f}")

    # Trials leaderboard
    if project.all_trials:
        lines.append("")
        lines.append(f"### All Trials ({len(project.all_trials)} total)")
        sorted_trials = sorted(project.all_trials,
                                key=lambda t: t.get("score", -np.inf), reverse=True)
        lines.append("| Model | Score | Status |")
        lines.append("|-------|-------|--------|")
        for t in sorted_trials[:10]:
            lines.append(f"| {t.get('model', '?')} | {t.get('score', 'N/A')} | {t.get('status', '?')} |")

    text = "\n".join(lines)
    text = text.encode("ascii", errors="replace").decode("ascii")
    return text


# ===========================================================================
# Main entry point
# ===========================================================================
def run_research_project(
    df: pd.DataFrame,
    col_classes: dict[str, list[str]],
    domain_ctx: dict[str, Any],
    target_col: str | None = None,
    objective: str | None = None,
    max_iterations: int = 5,
    budget_limit: dict | None = None,
    out_dir: str | Path | None = None,
    anthropic_api_key: str | None = None,
) -> dict:
    """Run a full 4-phase research project.

    Returns dict compatible with run_experiment_loop output (backward compat),
    plus additional research-specific keys.
    """
    from hooks.ml_experiment import (
        build_feature_matrix,
        run_experiment_loop,
        extract_feature_importances,
        render_experiment_summary,
    )

    start_time = time.time()

    # Initialize project
    project = ResearchProject(
        surfaces=_build_seed_surfaces(),
        budget=budget_limit or {"max_trials": 30, "max_seconds": 600},
    )

    # --- Quality report (optional) ---
    quality_df = None
    try:
        from hooks.data_quality import quality_report
        quality_df = quality_report(df)
    except Exception:
        pass

    # --- Phase 1: Understand ---
    project.phase = "understand"
    log.info("Research: Phase 1 — Understand")

    # Determine target
    if target_col is None:
        from hooks.ml_experiment import select_target
        target_info = select_target(df, col_classes, domain_ctx)
        if target_info is None:
            return {"error": "No suitable prediction target found",
                    "trials": [], "best_trial": {}, "incumbent_metric": -np.inf,
                    "primary_metric": "PR_AUC", "task_type": "binary",
                    "tsv_rows": [], "strategy_path": None,
                    "llm_proposals": [], "best_feature_set": [],
                    "best_importances": []}
        target_col = target_info["col"]

    understanding = phase1_understand(df, col_classes, domain_ctx,
                                       target_col, objective, quality_df)
    if "error" in understanding:
        return {"error": understanding["error"],
                "trials": [], "best_trial": {}, "incumbent_metric": -np.inf,
                "primary_metric": "PR_AUC", "task_type": "binary",
                "tsv_rows": [], "strategy_path": None,
                "llm_proposals": [], "best_feature_set": [],
                "best_importances": []}

    task_type = understanding["task_type"]
    primary_metric = _default_metric(task_type)
    project.objective = understanding["objective"]
    project.task_type = task_type
    project.primary_metric = primary_metric
    project.domain = understanding["domain"]

    # --- WI-2E: compute dataset hash for experiment ledger recall ---
    import hashlib
    _hash_input = f"{sorted(df.columns.tolist())}|{project.domain}|{target_col}"
    project._dataset_hash = hashlib.sha256(_hash_input.encode()).hexdigest()[:16]

    # --- Build feature matrix ---
    id_cols = col_classes.get("identifiers", [])
    try:
        X, y, feature_names = build_feature_matrix(df, target_col, id_cols)
    except Exception as exc:
        log.error("Research: feature matrix build failed: %s", exc)
        return {"error": str(exc),
                "trials": [], "best_trial": {}, "incumbent_metric": -np.inf,
                "primary_metric": primary_metric, "task_type": task_type,
                "tsv_rows": [], "strategy_path": None,
                "llm_proposals": [], "best_feature_set": [],
                "best_importances": []}

    # --- Phase 2 baseline: run existing experiment loop (round 0 = 5 models) ---
    log.info("Research: Baseline run (5-model roster)")
    baseline_result = run_experiment_loop(
        X, y, feature_names, task_type=task_type, max_trials=8,
        anthropic_api_key=None,  # No LLM for baseline — pure roster
        domain_ctx=domain_ctx,
        out_dir=out_dir,
        n_rounds=0,
    )

    project.baseline = baseline_result
    project.best_score = baseline_result["incumbent_metric"]
    best_trial = baseline_result.get("best_trial", {})
    project.best_model = best_trial.get("metrics", {}).get("Model", "")
    project.all_trials.extend(baseline_result.get("trials", []))

    # Extract baseline importances
    importances = baseline_result.get("best_importances", [])

    # --- Compute stopping threshold ---
    baseline_metrics = best_trial.get("metrics", {})
    threshold, threshold_rationale = compute_stopping_threshold(
        baseline_metrics, task_type, domain_ctx, primary_metric, anthropic_api_key,
    )
    project.stopping_threshold = threshold
    project.threshold_rationale = threshold_rationale
    log.info("Research: stopping threshold = %.4f (%s)", threshold, threshold_rationale[:80])

    # Check if baseline already meets threshold
    if project.best_score >= threshold:
        project.phase = "done"
        project.stop_reason = "threshold_met"
        log.info("Research: baseline already meets threshold!")
    else:
        # --- Split data for Phase 4 ---
        is_classification = task_type in ("binary", "multiclass")
        from sklearn.model_selection import train_test_split
        if is_classification:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=42,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42,
            )

        # --- Main research loop ---
        for iteration in range(max_iterations):
            project.iteration = iteration + 1
            log.info("Research: === Iteration %d/%d ===", project.iteration, max_iterations)

            # Phase 2: Design experiments
            # WI-2E: recall prior experiment findings for this domain
            prior_findings = None
            if hasattr(project, '_dataset_hash') and project._dataset_hash:
                try:
                    conn_str = os.environ.get("POSTGRES_URL")
                    if conn_str:
                        from hooks.db_store import get_connection
                        _conn = get_connection(conn_str)
                        prior_findings = recall_prior_experiments(
                            _conn, project._dataset_hash, project.domain, limit=15,
                        )
                        _conn.close()
                        if prior_findings:
                            log.info("Research: recalled %d prior experiment findings", len(prior_findings))
                except Exception as exc:
                    log.debug("Research: could not recall prior experiments: %s", exc)

            project.phase = "design"
            new_experiments = phase2_design(
                project, baseline_result, quality_df, importances, anthropic_api_key,
                prior_findings=prior_findings,
            )
            if not new_experiments:
                log.info("Research: no experiments designed, stopping")
                project.stop_reason = "no_experiments_designed"
                break

            # Phase 3: Prioritize
            new_experiments = phase3_prioritize(new_experiments, project)
            project.experiments.extend(new_experiments)

            # Phase 4: Execute
            project.phase = "execute"
            exec_result = phase4_execute(
                project, X, y, feature_names, new_experiments,
                X_train, X_test, y_train, y_test,
                out_dir, anthropic_api_key,
            )

            # Collect trials
            project.all_trials.extend(exec_result.get("trials", []))

            # Track improvement
            new_best = exec_result.get("best_score", -np.inf)
            improvement = max(0, new_best - project.best_score) if new_best > project.best_score else 0
            project.improvement_history.append({
                "iteration": project.iteration,
                "score": project.best_score,
                "improvement": improvement,
                "surface": "mixed",
                "experiments_run": exec_result.get("experiments_completed", 0),
            })

            if new_best > project.best_score:
                project.best_score = new_best
                best_trial = exec_result.get("best_trial", best_trial)
                project.best_model = best_trial.get("model", project.best_model)

            # Update importances from best model if available
            best_model_obj = exec_result.get("best_model")
            if best_model_obj:
                importances = extract_feature_importances(best_model_obj, feature_names)

            # WI-2B: detect contradictions across all completed experiments
            project.contradictions = _detect_contradictions(project.experiments)
            if project.contradictions:
                log.info("Research: %d contradictions detected", len(project.contradictions))

            # WI-2D: Strategy approval gate — every 2nd iteration, propose a change
            if project.iteration >= 2 and project.iteration % 2 == 0:
                proposal = _propose_strategy_change(project, project.improvement_history, anthropic_api_key)
                if proposal:
                    project.strategy_proposals.append(proposal)
                    risk = proposal.get("risk", "medium")
                    if risk == "low":
                        # Auto-approve low-risk changes
                        applied = _apply_strategy_proposal(project, proposal)
                        proposal["auto_approved"] = applied
                        log.info("Strategy: auto-approved '%s' — %s",
                                 proposal["change_type"], proposal["description"][:60])
                    else:
                        # Persist for human review via pending_confirmations
                        proposal["auto_approved"] = False
                        try:
                            conn_str = os.environ.get("POSTGRES_URL")
                            if conn_str:
                                from hooks.db_store import get_connection
                                from hooks.hitl_analyzer import (
                                    HITLRequest, analyze_all, aggregate_for_presentation,
                                    persist_presentation, init_pending_confirmations,
                                )
                                _conn = get_connection(conn_str)
                                init_pending_confirmations(_conn)
                                req = HITLRequest(
                                    request_id=f"strategy_{uuid.uuid4().hex[:8]}",
                                    agent_id="AG40",
                                    nature="rule_confirmation",
                                    entity_name=proposal["change_type"],
                                    entity_type="strategy_proposal",
                                    dataset=None,
                                    project=project.objective[:50],
                                    description=proposal["description"],
                                    priority="non_blocking",
                                    proposed_change=proposal,
                                )
                                analyzed = analyze_all([req])
                                pres = aggregate_for_presentation(analyzed)
                                persist_presentation(_conn, pres)
                                _conn.close()
                                log.info("Strategy: queued '%s' for human review — %s",
                                         proposal["change_type"], proposal["description"][:60])
                        except Exception as exc:
                            log.debug("Strategy: could not persist proposal: %s", exc)

            # Adapt threshold
            new_thresh, thresh_rationale = adapt_stopping_threshold(
                project, project.improvement_history,
            )
            project.stopping_threshold = new_thresh
            project.threshold_rationale = thresh_rationale

            # Update research project markdown
            update_research_project_md(project, out_dir)

            # Check if we should stop
            if project.stop_reason in ("threshold_met", "budget_exhausted", "plateau_detected"):
                log.info("Research: stopping after iteration %d — %s",
                         project.iteration, project.stop_reason)
                break

            # Reset stop_reason for next iteration if it was just "iteration_complete"
            if project.stop_reason == "iteration_complete":
                project.stop_reason = ""

    project.phase = "done"
    if not project.stop_reason:
        project.stop_reason = "max_iterations_reached"

    # --- Final outputs ---
    md_path = update_research_project_md(project, out_dir)
    summary = render_research_summary(project)

    # Self-improvement
    try:
        from hooks.self_improver import run_self_improvement
        run_self_improvement({
            "dataset_path": str(out_dir or ""),
            "query": f"ML research project: {project.objective}",
            "domain_detected": project.domain,
            "domain_method": domain_ctx.get("method", "unknown") if domain_ctx else "unknown",
            "domain_confidence": domain_ctx.get("confidence", "?") if domain_ctx else "?",
            "domain_correct": None,
            "domain_actual": None,
            "analysis_type": "ml_experiment",
            "workflows_run": ["E"],
            "row_count": len(df),
            "col_count": len(df.columns),
            "column_names": df.columns.tolist(),
            "issues": [],
            "findings_summary": (
                f"Research project: best={project.best_model}, "
                f"{primary_metric}={project.best_score:.4f}, "
                f"iterations={project.iteration}, stop={project.stop_reason}, "
                f"surfaces={list(set(e.surface for e in project.experiments))}"
            ),
        }, anthropic_api_key=anthropic_api_key)
    except Exception:
        pass

    # Build backward-compatible return dict
    # Merge baseline trials with research trials
    all_trials = project.all_trials

    # Find overall best trial
    overall_best = {}
    overall_best_score = -np.inf
    for t in all_trials:
        s = t.get("score", t.get("metrics", {}).get(primary_metric, -np.inf))
        if isinstance(s, (int, float)) and s > overall_best_score:
            overall_best_score = s
            overall_best = t

    return {
        # Backward-compatible keys from run_experiment_loop
        "trials": all_trials,
        "best_trial": overall_best,
        "incumbent_metric": project.best_score,
        "primary_metric": primary_metric,
        "task_type": task_type,
        "tsv_rows": baseline_result.get("tsv_rows", []),
        "strategy_path": baseline_result.get("strategy_path"),
        "llm_proposals": [],
        "best_feature_set": baseline_result.get("best_feature_set", []),
        "best_importances": importances,
        # Research-specific keys
        "project": asdict(project),
        "research_project_md_path": md_path,
        "research_summary": summary,
        "trace": {
            "phases": ["understand", "design", "prioritize", "execute"],
            "iterations": project.iteration,
            "stop_reason": project.stop_reason,
            "total_experiments": len(project.experiments),
            "total_trials": len(all_trials),
            "elapsed_s": round(time.time() - start_time, 2),
        },
    }


# ===========================================================================
# Agent entry point (for DESO SA3 → AG40 invocation)
# ===========================================================================
def execute_as_agent(
    dcp: dict,
    is_spec: dict,
    dataset_path: str,
    budget: dict | None = None,
    out_dir: str | None = None,
) -> dict:
    """Entry point for DESO SA3 invocation as AG40.

    Loads dataset, extracts col_classes from DCP, calls run_research_project(),
    returns standardized egress dict.
    """
    from pathlib import Path

    # Load dataset
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(dataset_path)
    elif dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)  # default to CSV

    # Extract col_classes from DCP or infer
    col_classes = dcp.get("col_classes", {})
    if not col_classes:
        # Basic inference
        col_classes = {
            "identifiers": [],
            "numerics": df.select_dtypes(include=["number"]).columns.tolist(),
            "categoricals": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        }

    # Extract domain_ctx from DCP
    domain_ctx = {
        "domain": dcp.get("domain", "General"),
        "confidence": dcp.get("confidence", 0.5),
        "summary": dcp.get("summary", ""),
    }

    # Extract target hint from IS if available
    target_col = is_spec.get("target_col")

    # Output directory
    if out_dir is None:
        out_dir = str(Path(dataset_path).parent / "research_output")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Run research project
    result = run_research_project(
        df=df,
        col_classes=col_classes,
        domain_ctx=domain_ctx,
        target_col=target_col,
        objective=is_spec.get("explicit_goal"),
        max_iterations=budget.get("max_iterations", 5) if budget else 5,
        budget_limit=budget or {"max_trials": 30, "max_seconds": 600},
        out_dir=out_dir,
        anthropic_api_key=None,  # Let it pick up from env
    )

    # Return standardized egress for DESO
    return {
        "research_result": {
            "best_score": result.get("incumbent_metric", -np.inf),
            "best_model": result.get("best_trial", {}).get("metrics", {}).get("Model", "N/A"),
            "primary_metric": result.get("primary_metric", ""),
            "task_type": result.get("task_type", ""),
            "total_trials": result.get("trace", {}).get("total_trials", 0),
            "total_experiments": result.get("trace", {}).get("total_experiments", 0),
            "iterations": result.get("trace", {}).get("iterations", 0),
        },
        "stop_reason": result.get("trace", {}).get("stop_reason", "unknown"),
        "research_project_md": result.get("research_project_md_path", ""),
        "research_summary": result.get("research_summary", ""),
        "full_result": result,
    }


# ===========================================================================
# L3 Experiment Ledger (PostgreSQL persistence)
# ===========================================================================
def init_experiment_ledger(conn) -> None:
    """Create the experiment_ledger table if not exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiment_ledger (
                id          SERIAL PRIMARY KEY,
                run_id      TEXT,
                dataset_hash TEXT,
                domain      TEXT,
                surface     TEXT,
                vector      TEXT,
                hypothesis  TEXT,
                config_json JSONB,
                metric_result FLOAT,
                expected_impact FLOAT,
                actual_impact FLOAT,
                lesson_learned TEXT,
                created_at  TIMESTAMPTZ DEFAULT now()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment_ledger_dataset
            ON experiment_ledger (dataset_hash, domain)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment_ledger_surface
            ON experiment_ledger (dataset_hash, surface)
        """)
        # WI-2A: structured lesson columns (idempotent migration)
        for col, col_type in [
            ("positive_finding", "TEXT"),
            ("negative_finding", "TEXT"),
            ("finding_type", "TEXT DEFAULT 'neutral'"),
        ]:
            try:
                cur.execute(f"ALTER TABLE experiment_ledger ADD COLUMN IF NOT EXISTS {col} {col_type}")
            except Exception:
                pass  # Column already exists or older PG without IF NOT EXISTS
    conn.commit()


def save_experiment_to_ledger(
    conn,
    run_id: str,
    experiment: dict,
    dataset_hash: str,
    domain: str,
    incumbent_score: float = -np.inf,
) -> None:
    """Persist a completed experiment to the ledger with structured findings.

    WI-2A: classifies each experiment as positive/negative/error and records
    both positive_finding and negative_finding for downstream learning.
    """
    # Handle both Experiment dataclass (as dict) and raw dict
    surface = experiment.get("surface", "")
    vector = experiment.get("vector", "")
    hypothesis = experiment.get("hypothesis", "")
    design = experiment.get("design", {})
    expected_impact = experiment.get("expected_impact", 0.0)
    results = experiment.get("results", {})

    metric_result = results.get("best_score", results.get("score", None))
    actual_impact = None
    error_msg = results.get("error", "")

    # Classify finding type and generate structured findings
    positive_finding = None
    negative_finding = None
    finding_type = "neutral"

    if error_msg:
        finding_type = "error"
        negative_finding = error_msg
    elif metric_result is not None and isinstance(metric_result, (int, float)):
        if metric_result > incumbent_score:
            delta = metric_result - incumbent_score
            actual_impact = delta
            finding_type = "positive"
            positive_finding = (
                f"Improved metric from {incumbent_score:.4f} to {metric_result:.4f} "
                f"(+{delta:.4f}) via {surface}/{vector}"
            )
        else:
            finding_type = "negative"
            negative_finding = (
                f"{surface}/{vector} did not improve metric "
                f"(got {metric_result:.4f} vs incumbent {incumbent_score:.4f})"
            )

    # Backward-compat: lesson_learned is a combined summary
    lesson = positive_finding or negative_finding or error_msg or ""

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO experiment_ledger
                (run_id, dataset_hash, domain, surface, vector, hypothesis,
                 config_json, metric_result, expected_impact, actual_impact,
                 lesson_learned, positive_finding, negative_finding, finding_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            run_id, dataset_hash, domain, surface, vector, hypothesis,
            json.dumps(design, default=str), metric_result,
            expected_impact, actual_impact, lesson,
            positive_finding, negative_finding, finding_type,
        ))


def recall_prior_experiments(
    conn,
    dataset_hash: str,
    domain: str,
    surface: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Retrieve prior experiments from the ledger for a dataset/domain.

    WI-2A: includes positive_finding, negative_finding, finding_type columns.
    """
    _COLUMNS = """run_id, dataset_hash, domain, surface, vector, hypothesis,
                  config_json, metric_result, expected_impact, actual_impact,
                  lesson_learned, created_at,
                  positive_finding, negative_finding, finding_type"""
    with conn.cursor() as cur:
        if surface:
            cur.execute(f"""
                SELECT {_COLUMNS}
                FROM experiment_ledger
                WHERE dataset_hash = %s AND domain = %s AND surface = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (dataset_hash, domain, surface, limit))
        else:
            cur.execute(f"""
                SELECT {_COLUMNS}
                FROM experiment_ledger
                WHERE dataset_hash = %s AND domain = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (dataset_hash, domain, limit))

        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(columns, row)) for row in rows]
