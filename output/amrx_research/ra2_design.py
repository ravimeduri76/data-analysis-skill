"""RA2: Experiment Designer + RA3: Priority Engine"""
import json

experiments = [
    # SURFACE 1: DATA
    {"id": "EXP-D1", "objective": "OBJ1", "surface": "Data", "vector": "Feature encoding recode",
     "hypothesis": "Recoding 1/2/3 to 1/0/NaN (treating 3=Unknown as missing) will improve MCC",
     "design": "Recode all 70 ternary features: 1->1, 2->0, 3->NaN. Impute with mode.",
     "expected_impact": 0.01, "cost": "low"},
    {"id": "EXP-D2", "objective": "OBJ1", "surface": "Data", "vector": "Dataset size surface",
     "hypothesis": "Total_data.csv (7866 rows) vs matching_data (4325) generalization test",
     "design": "Train on Total_data with same feature set. 5-fold CV.",
     "expected_impact": 0.02, "cost": "low"},
    {"id": "EXP-D3", "objective": "OBJ2a", "surface": "Data", "vector": "SMOTE rebalancing",
     "hypothesis": "SMOTE oversampling of non-Entero (n=344) improves MCC from 0.39",
     "design": "SMOTE(sampling_strategy=0.5) on training folds only.",
     "expected_impact": 0.08, "cost": "low"},
    {"id": "EXP-D4", "objective": "OBJ2b", "surface": "Data", "vector": "Aggressive class grouping",
     "hypothesis": "Reducing to top-5 organisms + Other (6 classes) improves MCC",
     "design": "Group all except codes 14,15,20,12,11 into Other.",
     "expected_impact": 0.06, "cost": "low"},
    # SURFACE 2: FEATURES
    {"id": "EXP-F1", "objective": "OBJ1", "surface": "Features", "vector": "Symptom interactions",
     "hypothesis": "UTI symptom co-occurrence patterns carry more signal than individuals",
     "design": "Top-10 pairwise interaction features from symptom group.",
     "expected_impact": 0.015, "cost": "low"},
    {"id": "EXP-F2", "objective": "OBJ1", "surface": "Features", "vector": "Symptom burden score",
     "hypothesis": "Composite symptom count captures UTI severity better",
     "design": "symptom_burden = count of symptoms where value=1. Also comorbidity_burden.",
     "expected_impact": 0.01, "cost": "low"},
    {"id": "EXP-F3", "objective": "OBJ2a", "surface": "Features", "vector": "Feature selection top-k",
     "hypothesis": "Top-20 features by importance reduces noise for small sample",
     "design": "RF importances from baseline. Retrain with top-20.",
     "expected_impact": 0.05, "cost": "low"},
    {"id": "EXP-F4", "objective": "OBJ2a", "surface": "Features", "vector": "BloodCulture one-hot",
     "hypothesis": "One-hot BloodCulture (44/45/46) captures non-linear relationship",
     "design": "One-hot encode into 3 binary columns.",
     "expected_impact": 0.03, "cost": "low"},
    {"id": "EXP-F5", "objective": "OBJ2b", "surface": "Features", "vector": "Urological PCA",
     "hypothesis": "PCA on 28 urological conditions creates latent risk factors",
     "design": "PCA top-5 components from urological columns.",
     "expected_impact": 0.04, "cost": "medium"},
    # SURFACE 3: MODELS
    {"id": "EXP-M1", "objective": "OBJ1", "surface": "Models", "vector": "CatBoost native categorical",
     "hypothesis": "CatBoost handles 1/2/3 as categorical without recoding",
     "design": "CatBoost with cat_features for all 70 ternary columns.",
     "expected_impact": 0.02, "cost": "medium"},
    {"id": "EXP-M2", "objective": "OBJ2a", "surface": "Models", "vector": "SVM-RBF",
     "hypothesis": "SVM-RBF captures non-linear Entero/non-Entero boundaries",
     "design": "SVM(kernel=rbf, class_weight=balanced) + StandardScaler.",
     "expected_impact": 0.06, "cost": "medium"},
    {"id": "EXP-M3", "objective": "OBJ2a", "surface": "Models", "vector": "Stacking ensemble",
     "hypothesis": "Stacking combines diverse model strengths for small sample",
     "design": "StackingClassifier(LR+RF+XGB base, LR meta).",
     "expected_impact": 0.07, "cost": "high"},
    {"id": "EXP-M4", "objective": "OBJ2b", "surface": "Models", "vector": "Hierarchical classifier",
     "hypothesis": "Two-stage (Entero vs non-Entero then species) outperforms flat multi-class",
     "design": "Stage 1: binary. Stage 2: sub-model within each group.",
     "expected_impact": 0.08, "cost": "high"},
    # SURFACE 4: PARAMETERS
    {"id": "EXP-P1", "objective": "OBJ1", "surface": "Parameters", "vector": "Optuna LightGBM",
     "hypothesis": "Bayesian optimization pushes LightGBM MCC above 0.90",
     "design": "Optuna 50 trials, MCC objective, 5-fold CV.",
     "expected_impact": 0.015, "cost": "medium"},
    {"id": "EXP-P2", "objective": "OBJ2a", "surface": "Parameters", "vector": "Optuna GBM + threshold",
     "hypothesis": "Tuning GBM params + threshold sweep improves bacteria MCC",
     "design": "Optuna 50 trials + threshold sweep 0.3-0.8.",
     "expected_impact": 0.06, "cost": "medium"},
    # SURFACE 5: VALIDATION
    {"id": "EXP-V1", "objective": "OBJ1", "surface": "Validation", "vector": "10-fold repeated CV",
     "hypothesis": "More folds and repeats provide tighter MCC confidence interval",
     "design": "RepeatedStratifiedKFold(10, 3). Report mean, std, 95% CI.",
     "expected_impact": 0.0, "cost": "medium"},
    # SURFACE 6: INTERPRETATION
    {"id": "EXP-I1", "objective": "OBJ1", "surface": "Interpretation", "vector": "SHAP analysis",
     "hypothesis": "SHAP reveals clinical feature interactions driving UTI prediction",
     "design": "SHAP on best model. Summary + dependence plots.",
     "expected_impact": 0.0, "cost": "medium"},
    {"id": "EXP-I2", "objective": "OBJ1", "surface": "Interpretation", "vector": "Clinical threshold opt",
     "hypothesis": "Threshold for sensitivity>=95% produces clinical screening tool",
     "design": "Sweep thresholds 0.1-0.9. Plot sensitivity vs specificity.",
     "expected_impact": 0.0, "cost": "low"},
]

# RA3: Prioritize
for exp in experiments:
    impact = exp["expected_impact"]
    cost_map = {"low": 1.0, "medium": 0.7, "high": 0.5}
    cost_factor = cost_map.get(exp["cost"], 0.5)
    obj_bonus = 1.0 if exp["objective"] == "OBJ1" else 1.5 if exp["objective"] == "OBJ2a" else 1.3
    exp["priority_score"] = round(impact * cost_factor * obj_bonus * 100, 2)

experiments.sort(key=lambda x: x["priority_score"], reverse=True)

print("=" * 80)
print("EXPERIMENT DESIGNS (RA2) -- Prioritized by RA3")
print("=" * 80)
print(f"{'ID':10s} {'Obj':6s} {'Surface':15s} {'Vector':35s} {'Impact':8s} {'Cost':6s} {'Score':6s}")
print("-" * 80)
for exp in experiments:
    print(f"{exp['id']:10s} {exp['objective']:6s} {exp['surface']:15s} {exp['vector'][:35]:35s} {exp['expected_impact']:8.3f} {exp['cost']:6s} {exp['priority_score']:6.2f}")

print(f"\nTotal: {len(experiments)} experiments")
print(f"  OBJ1:  {sum(1 for e in experiments if e['objective']=='OBJ1')}")
print(f"  OBJ2a: {sum(1 for e in experiments if e['objective']=='OBJ2a')}")
print(f"  OBJ2b: {sum(1 for e in experiments if e['objective']=='OBJ2b')}")

with open(r"C:\Users\Ravi\research\data-analysis\output\amrx_research\experiment_designs.json", "w") as f:
    json.dump(experiments, f, indent=2)
print("\nSaved: output/amrx_research/experiment_designs.json")
