"""
AutoML v4 — Adds XGBoost (eval_metric='aucpr') to the v3 feature set.
Combines all 5 models into a single final leaderboard.
Target: Mortality_Flag (binary, ~6.1% positive)
No SMOTE — class_weight / scale_pos_weight throughout.
"""
import sys, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.metrics          import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, average_precision_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.calibration      import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

OUT = r"C:\Users\Ravi\Elara_Probus\Probus_Data"

# ── 1. Load v3 dataset ────────────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready_v3.csv", low_memory=False)
ml["gender_enc"] = ml["gender_enc"].fillna(ml["gender_enc"].median())
print(f"Dataset: {ml.shape[0]:,} rows x {ml.shape[1]} cols")

TARGET = "Mortality_Flag"
DROP   = ["ClientId"]
X = ml.drop(columns=[TARGET] + DROP)
y = ml[TARGET]

pos_rate      = y.mean()
neg_pos_ratio = (y == 0).sum() / (y == 1).sum()
print(f"Target: {y.sum():,} positives ({pos_rate*100:.1f}%)  neg:pos={neg_pos_ratio:.1f}:1")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Pos in test: {y_test.sum():,}")

# ── 2. Models ─────────────────────────────────────────────────────────────────
# XGBoost — two variants:
#   XGB_logloss : standard binary:logistic + scale_pos_weight (baseline XGB)
#   XGB_aucpr   : optimises directly for PR-AUC via eval_metric='aucpr'
#                 uses early stopping on a 10% internal validation split

MODELS = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04, num_leaves=63,
        max_depth=-1, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        random_state=42, verbose=-1, n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=600, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "LogisticRegression": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1, class_weight="balanced",
            solver="lbfgs", max_iter=2000,
            random_state=42, n_jobs=-1,
        )),
    ]),
    "MLP (Neural Net)": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu", solver="adam", alpha=5e-4,
            learning_rate="adaptive", learning_rate_init=5e-4,
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=25,
            random_state=42,
        )),
    ]),
    # XGBoost standard (logloss) — reference
    "XGB (logloss)": xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    # XGBoost PR-AUC optimised — primary new model
    "XGB (aucpr)": xgb.XGBClassifier(
        n_estimators=1000,           # higher ceiling; early stopping will cut this
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,                   # min loss reduction per split — reduces FP splits
        scale_pos_weight=neg_pos_ratio,
        objective="binary:logistic",
        eval_metric="aucpr",         # optimise directly for PR-AUC
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=40,    # stop if no PR-AUC gain for 40 rounds
    ),
}

# ── 3. Train + evaluate ───────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Split a small internal val set for XGB early stopping (train only)
X_tr_fit, X_val_es, y_tr_fit, y_val_es = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

for name, model in MODELS.items():
    print(f"\nTraining {name}...", end=" ", flush=True)
    t0 = time.time()

    # XGB aucpr needs eval_set for early stopping
    if name == "XGB (aucpr)":
        model.fit(X_tr_fit, y_tr_fit,
                  eval_set=[(X_val_es, y_val_es)],
                  verbose=False)
    elif name == "XGB (logloss)":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s", end="")

    if name == "XGB (aucpr)":
        best_iter = model.best_iteration
        print(f"  [best_iteration={best_iter}]", end="")
    print()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    f1_pos   = f1_score(y_test, y_pred, pos_label=1)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    prec     = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec      = recall_score(y_test, y_pred, pos_label=1)
    acc      = accuracy_score(y_test, y_pred)
    mcc      = matthews_corrcoef(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec     = tn / (tn + fp) if (tn + fp) > 0 else 0

    # CV ROC-AUC (skip for XGB aucpr to save time — report test only)
    if "XGB" not in name:
        cv_auc = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    else:
        cv_auc = float("nan")

    results[name] = {
        "ROC-AUC (test)": round(roc_auc,4), "ROC-AUC (5-CV)": round(cv_auc,4) if not np.isnan(cv_auc) else "—",
        "PR-AUC": round(pr_auc,4),
        "F1 (mortality)": round(f1_pos,4), "F1 (macro)": round(f1_macro,4),
        "Precision": round(prec,4), "Recall": round(rec,4),
        "Specificity": round(spec,4), "Accuracy": round(acc,4),
        "MCC": round(mcc,4),
        "TP":tp,"FP":fp,"TN":tn,"FN":fn,
        "Train time (s)": round(elapsed,1),
        "_model":model, "_y_prob":y_prob, "_y_pred":y_pred, "_cm":cm,
    }
    print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1+={f1_pos:.4f}  "
          f"Recall={rec:.4f}  Prec={prec:.4f}  MCC={mcc:.4f}")
    print(f"  Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(classification_report(y_test, y_pred,
          target_names=["No Mortality","Mortality"], digits=4))

# ── 4. Summary leaderboard ────────────────────────────────────────────────────
DISP = ["ROC-AUC (test)","ROC-AUC (5-CV)","PR-AUC","F1 (mortality)",
        "F1 (macro)","Precision","Recall","Specificity","Accuracy","MCC","Train time (s)"]
summary = pd.DataFrame({k:{c:v[c] for c in DISP} for k,v in results.items()}).T
summary.index.name = "Model"

# Sort by PR-AUC (primary goal)
summary_sorted = summary.copy()
summary_sorted["PR-AUC_num"] = pd.to_numeric(summary_sorted["PR-AUC"], errors="coerce")
summary_sorted = summary_sorted.sort_values("PR-AUC_num", ascending=False).drop(columns="PR-AUC_num")

print("\n" + "=" * 110)
print("FINAL LEADERBOARD — All 6 models | Sorted by PR-AUC")
print("=" * 110)
print(summary_sorted[DISP].to_string())

best_roc  = summary["ROC-AUC (test)"].astype(float).idxmax()
best_pr   = summary["PR-AUC"].astype(float).idxmax()
best_mcc  = summary["MCC"].astype(float).idxmax()
best_rec  = summary["Recall"].astype(float).idxmax()
print(f"\nBest ROC-AUC : {best_roc}  ({summary.loc[best_roc,'ROC-AUC (test)']})")
print(f"Best PR-AUC  : {best_pr}  ({summary.loc[best_pr,'PR-AUC']})")
print(f"Best MCC     : {best_mcc}  ({summary.loc[best_mcc,'MCC']})")
print(f"Best Recall  : {best_rec}  ({summary.loc[best_rec,'Recall']})")

summary_sorted.reset_index().to_csv(f"{OUT}\\mortality_model_comparison_v4_final.csv", index=False)
print(f"\nLeaderboard saved: {OUT}\\mortality_model_comparison_v4_final.csv")

# ── 5. XGB feature importance ─────────────────────────────────────────────────
xgb_aucpr_model = results["XGB (aucpr)"]["_model"]
xgb_imp = pd.Series(xgb_aucpr_model.feature_importances_, index=X.columns)\
            .sort_values(ascending=False)
print(f"\nXGB (aucpr) Top 30 features:")
print(xgb_imp.head(30).to_string())

# ── 6. Final comprehensive plot ───────────────────────────────────────────────
COLORS = {
    "LightGBM":        "#E53935",
    "RandomForest":    "#43A047",
    "LogisticRegression": "#1E88E5",
    "MLP (Neural Net)":"#FB8C00",
    "XGB (logloss)":   "#7B1FA2",
    "XGB (aucpr)":     "#00ACC1",
}

fig = plt.figure(figsize=(26, 22))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# (a) ROC curves
ax_roc = fig.add_subplot(gs[0, 0])
for n, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["_y_prob"])
    lw = 2.5 if "XGB" in n else 1.5
    ax_roc.plot(fpr, tpr, label=f"{n} ({res['ROC-AUC (test)']:.3f})",
                color=COLORS[n], lw=lw,
                linestyle="--" if n == "XGB (logloss)" else "-")
ax_roc.plot([0,1],[0,1],"k--",lw=1)
ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
ax_roc.legend(fontsize=6.5); ax_roc.grid(alpha=0.3)

# (b) Precision-Recall curves
ax_pr = fig.add_subplot(gs[0, 1])
for n, res in results.items():
    p, r, _ = precision_recall_curve(y_test, res["_y_prob"])
    lw = 2.5 if "XGB" in n else 1.5
    ax_pr.plot(r, p, label=f"{n} (PR={res['PR-AUC']:.3f})",
               color=COLORS[n], lw=lw,
               linestyle="--" if n == "XGB (logloss)" else "-")
ax_pr.axhline(pos_rate, color="k", ls=":", lw=1, label=f"Baseline ({pos_rate:.3f})")
ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
ax_pr.legend(fontsize=6.5); ax_pr.grid(alpha=0.3)

# (c) PR-AUC bar — highlight XGB aucpr
ax_prbar = fig.add_subplot(gs[0, 2])
pr_vals = {n: float(res["PR-AUC"]) for n, res in results.items()}
pr_sorted = dict(sorted(pr_vals.items(), key=lambda x: x[1]))
edge_colors = ["#00ACC1" if n == "XGB (aucpr)" else "none" for n in pr_sorted]
bars = ax_prbar.barh(list(pr_sorted.keys()), list(pr_sorted.values()),
                     color=[COLORS[n] for n in pr_sorted], alpha=0.85,
                     edgecolor=edge_colors, linewidth=2)
ax_prbar.axvline(pos_rate, color="k", ls="--", lw=1, label=f"Baseline={pos_rate:.3f}")
ax_prbar.set(xlabel="PR-AUC", title="PR-AUC Ranking")
ax_prbar.legend(fontsize=8); ax_prbar.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, pr_sorted.values()):
    ax_prbar.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                  f"{val:.4f}", va="center", fontsize=8)

# (d) MCC vs Recall scatter
ax_sc = fig.add_subplot(gs[0, 3])
for n, res in results.items():
    ax_sc.scatter(float(res["Recall"]), float(res["MCC"]),
                  color=COLORS[n], s=120, zorder=5,
                  marker="*" if "XGB" in n else "o")
    ax_sc.annotate(n.replace(" (Neural Net)","").replace(" (aucpr)","*").replace(" (logloss)",""),
                   (float(res["Recall"]), float(res["MCC"])),
                   textcoords="offset points", xytext=(5,3), fontsize=7)
ax_sc.set(xlabel="Recall", ylabel="MCC", title="MCC vs Recall trade-off")
ax_sc.grid(alpha=0.3)
ax_sc.axhline(0, color="k", lw=0.5)

# (e-h) Confusion matrices — all 6 models (2 rows x 3 cols across rows 1+2)
cm_positions = [(1,0),(1,1),(1,2),(1,3),(2,0),(2,1)]
for idx, (name, res) in enumerate(results.items()):
    r, c = cm_positions[idx]
    ax_cm = fig.add_subplot(gs[r, c])
    disp = ConfusionMatrixDisplay(confusion_matrix=res["_cm"],
                                   display_labels=["No Mort.","Mortality"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"{name}\nROC={res['ROC-AUC (test)']:.3f}  PR={res['PR-AUC']:.3f}"
        f"\nRecall={res['Recall']:.3f}  MCC={res['MCC']:.3f}",
        fontsize=8, color="#00ACC1" if "aucpr" in name else "black")
    ax_cm.tick_params(labelsize=8)

# (i) XGB aucpr feature importance (top 25)
ax_xfi = fig.add_subplot(gs[2, 2])
xgb_imp.head(25).sort_values().plot(kind="barh", ax=ax_xfi, color="#00ACC1", alpha=0.85)
ax_xfi.set(title="XGB (aucpr) Top-25 Features")
ax_xfi.tick_params(labelsize=6.5); ax_xfi.grid(axis="x", alpha=0.3)

# (j) All-version PR-AUC evolution (v1 → v4 per model)
ax_evo = fig.add_subplot(gs[2, 3])
try:
    v1 = pd.read_csv(f"{OUT}\\mortality_model_comparison.csv",    index_col="Model")
    v3 = pd.read_csv(f"{OUT}\\mortality_model_comparison_v3.csv", index_col="Model")
    shared = ["LightGBM","RandomForest","LogisticRegression","MLP (Neural Net)"]
    for m in shared:
        pr_v1 = float(v1.loc[m,"PR-AUC"]) if m in v1.index else None
        pr_v3 = float(v3.loc[m,"PR-AUC"]) if m in v3.index else None
        pr_v4 = float(results[m]["PR-AUC"])
        pts = [x for x in [pr_v1, pr_v3, pr_v4] if x is not None]
        xs  = list(range(len(pts)))
        ax_evo.plot(xs, pts, "o-", color=COLORS[m], lw=1.5, label=m, markersize=5)
    # add XGB aucpr as a point at v4
    ax_evo.scatter([2], [float(results["XGB (aucpr)"]["PR-AUC"])],
                   color=COLORS["XGB (aucpr)"], s=120, zorder=6, marker="*",
                   label="XGB (aucpr)")
    ax_evo.set_xticks([0,1,2]); ax_evo.set_xticklabels(["v1","v3","v4"])
    ax_evo.set(xlabel="Version", ylabel="PR-AUC", title="PR-AUC Evolution v1 -> v4")
    ax_evo.legend(fontsize=6.5); ax_evo.grid(alpha=0.3)
    ax_evo.axhline(pos_rate, color="k", ls=":", lw=1)
except Exception as e:
    ax_evo.text(0.5,0.5,f"Evolution chart\nunavailable:\n{e}",
                transform=ax_evo.transAxes, ha="center", fontsize=8)

plt.suptitle(
    "Mortality Prediction — Final Model Leaderboard (v4)\n"
    "Dataset: 20,888 clients | 6.1% mortality | Features: CCI + ICD + Q interactions (115 cols)",
    fontsize=12, fontweight="bold", y=1.01
)
chart_path = f"{OUT}\\mortality_model_comparison_v4_final.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFinal chart saved: {chart_path}")
print("=== ALL DONE ===")
