"""
automl_mortality_v6.py
6-model leaderboard on mortality_ml_ready_v5.csv (211 features).

New v5 features vs v4:
  - rc_* pivot (61 reason-code count columns)
  - time_since_admission_days
  - eff_start_*/eff_end_*/adm_* weekday counts (21 columns)

Leakage guard:
  - rc_client_expired     (corr=0.9967) — directly encodes the death event -> DROPPED
  - rc_pending_closure_death (corr=0.067) — pending discharge for death  -> DROPPED

Models (no SMOTE, imbalance handled via class weights / scale_pos_weight):
  1. LightGBM
  2. RandomForest
  3. LogisticRegression
  4. MLP
  5. XGBoost (logloss)
  6. XGBoost (aucpr)  <-- primary target metric
"""
import sys, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

OUT     = r"C:\Users\Ravi\Elara_Probus\Probus_Data"
OUTIMG  = r"C:\Users\Ravi\research\data-analysis\output"
import os; os.makedirs(OUTIMG, exist_ok=True)

LEAKAGE_COLS = [
    "rc_client_expired",        # corr=0.997 — encodes mortality directly
    "rc_pending_closure_death", # corr=0.067 — pending discharge for death
]

# ── 1. Load ────────────────────────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready_v5.csv", low_memory=False)
print(f"Loaded: {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"Target: Mortality_Flag=1: {ml['Mortality_Flag'].sum():,} "
      f"({ml['Mortality_Flag'].mean()*100:.1f}%)")

dropped = [c for c in LEAKAGE_COLS if c in ml.columns]
if dropped:
    ml.drop(columns=dropped, inplace=True)
    print(f"Dropped leakage cols: {dropped}")

# ── 2. Features / target ───────────────────────────────────────────────────────
DROP_COLS = {"ClientId", "Mortality_Flag"}
feature_cols = [c for c in ml.columns if c not in DROP_COLS]

X = ml[feature_cols].copy()
y = ml["Mortality_Flag"].astype(int)

# Impute NaNs for sklearn models (LightGBM / XGB handle natively)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

print(f"\nFeature matrix: {X.shape[1]} features  (after leakage drop)")
print(f"Missing before impute: {X.isnull().any(axis=1).sum():,} rows")

# ── 3. Train / test split ──────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

X_tr, X_te, y_tr, y_te = train_test_split(
    X_imp, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_tr.shape[0]:,}  Test: {X_te.shape[0]:,}  "
      f"Pos rate train: {y_tr.mean()*100:.1f}%")

neg_pos_ratio = int((y_tr == 0).sum() / (y_tr == 1).sum())
print(f"neg_pos_ratio (scale_pos_weight): {neg_pos_ratio}")

# ── 4. Metrics helper ──────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, matthews_corrcoef,
    roc_curve, precision_recall_curve
)

def evaluate(name, y_true, y_prob, y_pred=None):
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    roc   = roc_auc_score(y_true, y_prob)
    pr    = average_precision_score(y_true, y_prob)
    f1p   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec   = (y_pred[y_true == 1] == 1).mean()
    mcc   = matthews_corrcoef(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm.ravel()
    spec  = tn / (tn + fp) if (tn+fp) > 0 else 0
    return dict(Model=name, ROC_AUC=round(roc,4), PR_AUC=round(pr,4),
                F1_pos=round(f1p,4), Recall=round(rec,4),
                Specificity=round(spec,4), MCC=round(mcc,4),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))

results = []
curves  = {}   # {name: (fpr,tpr) or (prec,rec)}

# ── 5a. LightGBM ──────────────────────────────────────────────────────────────
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=42, verbose=-1
)
lgb_model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
lgb_prob = lgb_model.predict_proba(X_te)[:, 1]
lgb_pred = lgb_model.predict(X_te)
r = evaluate("LightGBM", y_te, lgb_prob, lgb_pred)
results.append(r)
curves["LightGBM"] = (
    roc_curve(y_te, lgb_prob)[:2],
    precision_recall_curve(y_te, lgb_prob)[:2]
)
print(f"\n[LightGBM]  ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  "
      f"Recall={r['Recall']}  MCC={r['MCC']}")

# ── 5b. RandomForest ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=5,
    class_weight="balanced", n_jobs=-1, random_state=42
)
rf_model.fit(X_tr, y_tr)
rf_prob = rf_model.predict_proba(X_te)[:, 1]
rf_pred = rf_model.predict(X_te)
r = evaluate("RandomForest", y_te, rf_prob, rf_pred)
results.append(r)
curves["RandomForest"] = (
    roc_curve(y_te, rf_prob)[:2],
    precision_recall_curve(y_te, rf_prob)[:2]
)
print(f"[RandomForest] ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  "
      f"Recall={r['Recall']}  MCC={r['MCC']}")

# ── 5c. LogisticRegression ────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lr_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("lr",     LogisticRegression(C=0.1, class_weight="balanced",
                                  max_iter=1000, random_state=42))
])
lr_pipe.fit(X_tr, y_tr)
lr_prob = lr_pipe.predict_proba(X_te)[:, 1]
lr_pred = lr_pipe.predict(X_te)
r = evaluate("LogisticRegression", y_te, lr_prob, lr_pred)
results.append(r)
curves["LogisticRegression"] = (
    roc_curve(y_te, lr_prob)[:2],
    precision_recall_curve(y_te, lr_prob)[:2]
)
print(f"[LogReg]    ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  "
      f"Recall={r['Recall']}  MCC={r['MCC']}")

# ── 5d. MLP ───────────────────────────────────────────────────────────────────
from sklearn.neural_network import MLPClassifier

mlp_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("mlp",    MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu",
        learning_rate_init=0.001, max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=42
    ))
])
# MLPClassifier doesn't support class_weight — use sample_weight
sw = np.where(y_tr == 1, neg_pos_ratio, 1).astype(float)
mlp_pipe.fit(X_tr, y_tr, mlp__sample_weight=sw)
mlp_prob = mlp_pipe.predict_proba(X_te)[:, 1]
mlp_pred = mlp_pipe.predict(X_te)
r = evaluate("MLP", y_te, mlp_prob, mlp_pred)
results.append(r)
curves["MLP"] = (
    roc_curve(y_te, mlp_prob)[:2],
    precision_recall_curve(y_te, mlp_prob)[:2]
)
print(f"[MLP]       ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  "
      f"Recall={r['Recall']}  MCC={r['MCC']}")

# ── 5e-f. XGBoost (logloss + aucpr) ──────────────────────────────────────────
import xgboost as xgb

XGB_BASE = dict(
    n_estimators=1000, learning_rate=0.03,
    max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
    scale_pos_weight=neg_pos_ratio,
    use_label_encoder=False, random_state=42, n_jobs=-1
)

# Split training set for early stopping
X_tr_fit, X_val_es, y_tr_fit, y_val_es = train_test_split(
    X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr
)

for metric, label in [("logloss", "XGB (logloss)"), ("aucpr", "XGB (aucpr)")]:
    xgb_m = xgb.XGBClassifier(
        **XGB_BASE,
        eval_metric=metric,
        early_stopping_rounds=40,
        verbosity=0
    )
    xgb_m.fit(X_tr_fit, y_tr_fit,
              eval_set=[(X_val_es, y_val_es)],
              verbose=False)
    xgb_prob = xgb_m.predict_proba(X_te)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    r = evaluate(label, y_te, xgb_prob, xgb_pred)
    r["best_iter"] = xgb_m.best_iteration
    results.append(r)
    curves[label] = (
        roc_curve(y_te, xgb_prob)[:2],
        precision_recall_curve(y_te, xgb_prob)[:2]
    )
    print(f"[{label}]  ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  "
          f"Recall={r['Recall']}  MCC={r['MCC']}  best_iter={xgb_m.best_iteration}")

# ── 6. Leaderboard ────────────────────────────────────────────────────────────
lb = pd.DataFrame(results).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
lb.index += 1
print("\n=== LEADERBOARD (v6) ===")
print(lb[["Model","ROC_AUC","PR_AUC","F1_pos","Recall","Specificity","MCC"]].to_string())
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_csv(lb, f"{OUT}\\mortality_model_comparison_v6_final.csv")

# ── 7. Feature importance (LightGBM) ──────────────────────────────────────────
fi = pd.DataFrame({
    "feature":    feature_cols,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)
fi.index += 1
print("\nTop 30 LightGBM feature importances (v6):")
print(fi.head(30).to_string())

# ── 8. 9-panel chart ───────────────────────────────────────────────────────────
COLORS = {
    "LightGBM":         "#1565C0",
    "RandomForest":     "#2E7D32",
    "LogisticRegression":"#F57F17",
    "MLP":              "#6A1B9A",
    "XGB (logloss)":    "#AD1457",
    "XGB (aucpr)":      "#00838F",
}

fig = plt.figure(figsize=(22, 22))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: ROC + PR + PR-AUC bar
ax_roc = fig.add_subplot(gs[0, 0])
ax_pr  = fig.add_subplot(gs[0, 1])
ax_bar = fig.add_subplot(gs[0, 2])

# Baseline
ax_pr.axhline(y_te.mean(), color="gray", linestyle="--", lw=1,
              label=f"Baseline ({y_te.mean():.3f})")

for name, ((fpr, tpr), (prec, rec)) in curves.items():
    c = COLORS.get(name, "black")
    row = lb[lb["Model"] == name].iloc[0]
    ax_roc.plot(fpr, tpr, color=c, lw=1.5,
                label=f"{name} ({row['ROC_AUC']:.3f})")
    ax_pr.plot(rec, prec, color=c, lw=1.5,
               label=f"{name} ({row['PR_AUC']:.3f})")

ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
ax_roc.set(title="ROC Curves", xlabel="FPR", ylabel="TPR")
ax_roc.legend(fontsize=7)

ax_pr.set(title="Precision-Recall Curves", xlabel="Recall", ylabel="Precision")
ax_pr.legend(fontsize=7)

# PR-AUC bar
bar_models = lb["Model"].tolist()
bar_vals   = lb["PR_AUC"].tolist()
bar_cols   = [COLORS.get(m, "gray") for m in bar_models]
bars = ax_bar.bar(range(len(bar_models)), bar_vals, color=bar_cols)
ax_bar.set_xticks(range(len(bar_models)))
ax_bar.set_xticklabels([m.replace(" ", "\n") for m in bar_models], fontsize=8)
ax_bar.axhline(y_te.mean(), color="gray", linestyle="--", lw=1, label="Baseline")
ax_bar.set_title("PR-AUC Comparison")
ax_bar.set_ylabel("PR-AUC")
for bar, val in zip(bars, bar_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
ax_bar.legend(fontsize=8)

# Row 1: MCC/Recall scatter + 4 confusion matrices (top 4 by PR-AUC)
ax_sc = fig.add_subplot(gs[1, 0])
for _, row in lb.iterrows():
    c = COLORS.get(row["Model"], "black")
    ax_sc.scatter(row["Recall"], row["MCC"], color=c, s=100, zorder=5)
    ax_sc.annotate(row["Model"].replace(" ", "\n"),
                   (row["Recall"], row["MCC"]), fontsize=6,
                   xytext=(4, 2), textcoords="offset points")
ax_sc.set(title="MCC vs Recall", xlabel="Recall", ylabel="MCC")
ax_sc.grid(True, alpha=0.3)

top4 = lb.head(4)["Model"].tolist()
cm_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
           fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
for ax_cm, mname in zip(cm_axes, top4):
    row = lb[lb["Model"] == mname].iloc[0]
    cm_vals = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
    im = ax_cm.imshow(cm_vals, cmap="Blues")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred 0", "Pred 1"])
    ax_cm.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm_vals[i, j]), ha="center", va="center",
                       fontsize=11, color="white" if cm_vals[i, j] > cm_vals.max()/2 else "black")
    ax_cm.set_title(f"{mname}\nPR-AUC={row['PR_AUC']:.3f} Recall={row['Recall']:.3f}", fontsize=9)

# Row 2: Feature importance + PR-AUC evolution (v1-v6)
ax_fi = fig.add_subplot(gs[2, 2])
top20 = fi.head(20)
ax_fi.barh(top20["feature"][::-1], top20["importance"][::-1], color="#1565C0")
ax_fi.set_title("LightGBM Top-20 Features (v6)", fontsize=10)
ax_fi.set_xlabel("Importance")
ax_fi.tick_params(axis="y", labelsize=7)

fig.suptitle("Mortality Prediction — AutoML v6 (211 features, 6 models)", fontsize=14, y=0.98)
chart_path = f"{OUTIMG}\\mortality_model_comparison_v6_{ts}.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nChart saved: {chart_path}")

# ── 9. PR-AUC evolution across dataset versions ────────────────────────────────
evolution = pd.DataFrame([
    {"version": "v1\n(baseline)",  "best_prauc": 0.205},
    {"version": "v2\n(+ICD/Q)",   "best_prauc": 0.220},
    {"version": "v3\n(+CCI/interactions)", "best_prauc": 0.275},
    {"version": "v4\n(+events basic)",    "best_prauc": 0.374},
    {"version": "v5\n(+rc/DOW/tsa)",      "best_prauc": lb["PR_AUC"].max()},
])
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(evolution["version"], evolution["best_prauc"], "o-",
         color="#00838F", lw=2.5, markersize=9)
ax2.axhline(0.061, color="gray", linestyle="--", lw=1, label="Baseline (prevalence=0.061)")
for _, row in evolution.iterrows():
    ax2.annotate(f"{row['best_prauc']:.3f}",
                 (row["version"], row["best_prauc"]),
                 xytext=(0, 10), textcoords="offset points",
                 ha="center", fontsize=10, fontweight="bold", color="#00838F")
ax2.set_title("PR-AUC Evolution Across Dataset Versions", fontsize=13)
ax2.set_ylabel("Best PR-AUC")
ax2.set_ylim(0, 0.6)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
evo_path = f"{OUTIMG}\\prauc_evolution_v6_{ts}.png"
fig2.savefig(evo_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Evolution chart saved: {evo_path}")

print("\n=== DONE ===")
