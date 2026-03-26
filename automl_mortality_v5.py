"""
AutoML v5 — v4 dataset (adds client-event features from Client_Events.xlsx)
6 models: LightGBM, RandomForest, LogisticRegression, MLP, XGB(logloss), XGB(aucpr)
No SMOTE — class_weight / scale_pos_weight throughout.
days_active_before_expiry is intentionally sparse (NaN = never expired);
tree models handle NaN natively; pipelines use SimpleImputer(median).
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
import lightgbm as lgb
import xgboost as xgb

OUT = r"C:\Users\Ravi\Elara_Probus\Probus_Data"

# ── 1. Load v4 dataset ────────────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready_v4.csv", low_memory=False)
ml["gender_enc"] = ml["gender_enc"].fillna(ml["gender_enc"].median())
print(f"Dataset: {ml.shape[0]:,} rows x {ml.shape[1]} cols")

TARGET = "Mortality_Flag"
DROP   = ["ClientId"]
X = ml.drop(columns=[TARGET] + DROP)
y = ml[TARGET]

pos_rate      = y.mean()
neg_pos_ratio = (y == 0).sum() / (y == 1).sum()
print(f"Target: {y.sum():,} positives ({pos_rate*100:.1f}%)  neg:pos={neg_pos_ratio:.1f}:1")
print(f"NaN in days_active_before_expiry: {X['days_active_before_expiry'].isna().sum():,} "
      f"(filled by imputer in pipelines; LightGBM/XGB handle natively)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Pos in test: {y_test.sum():,}")

# ── 2. Models ─────────────────────────────────────────────────────────────────
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
    "XGB (logloss)": xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.04, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False,
        random_state=42, n_jobs=-1, verbosity=0,
    ),
    "XGB (aucpr)": xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, scale_pos_weight=neg_pos_ratio,
        objective="binary:logistic", eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=40,
    ),
}

# ── 3. Train + evaluate ───────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_tr_fit, X_val_es, y_tr_fit, y_val_es = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

for name, model in MODELS.items():
    print(f"\nTraining {name}...", end=" ", flush=True)
    t0 = time.time()

    if name == "XGB (aucpr)":
        model.fit(X_tr_fit, y_tr_fit,
                  eval_set=[(X_val_es, y_val_es)], verbose=False)
    else:
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    extra = f"  [best_iter={model.best_iteration}]" if name == "XGB (aucpr)" else ""
    print(f"done in {elapsed:.1f}s{extra}")

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

    if "XGB" not in name:
        cv_auc = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    else:
        cv_auc = float("nan")

    results[name] = dict(
        ROC_AUC=round(roc_auc,4), CV_AUC=round(cv_auc,4) if not np.isnan(cv_auc) else None,
        PR_AUC=round(pr_auc,4), F1_pos=round(f1_pos,4), F1_macro=round(f1_macro,4),
        Precision=round(prec,4), Recall=round(rec,4),
        Specificity=round(spec,4), Accuracy=round(acc,4), MCC=round(mcc,4),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        TrainTime=round(elapsed,1),
        _model=model, _y_prob=y_prob, _y_pred=y_pred, _cm=cm,
    )

    print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1+={f1_pos:.4f}  "
          f"Recall={rec:.4f}  Prec={prec:.4f}  MCC={mcc:.4f}")
    print(f"  Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(classification_report(y_test, y_pred,
          target_names=["No Mortality","Mortality"], digits=4))

# ── 4. Summary leaderboard ────────────────────────────────────────────────────
disp_map = {
    "ROC-AUC":    "ROC_AUC",
    "CV-AUC":     "CV_AUC",
    "PR-AUC":     "PR_AUC",
    "F1+":        "F1_pos",
    "F1-macro":   "F1_macro",
    "Precision":  "Precision",
    "Recall":     "Recall",
    "Specificity":"Specificity",
    "Accuracy":   "Accuracy",
    "MCC":        "MCC",
    "Train(s)":   "TrainTime",
}
rows = {}
for name, r in results.items():
    rows[name] = {k: r[v] for k, v in disp_map.items()}

summary = pd.DataFrame(rows).T
summary.index.name = "Model"
summary_sorted = summary.copy()
summary_sorted["_pr"] = pd.to_numeric(summary_sorted["PR-AUC"], errors="coerce")
summary_sorted = summary_sorted.sort_values("_pr", ascending=False).drop(columns="_pr")

print("\n" + "=" * 115)
print("FINAL LEADERBOARD v5 — Client Events + CCI + ICD + Q interactions | Sorted by PR-AUC")
print("=" * 115)
print(summary_sorted.to_string())

best_roc = summary["ROC-AUC"].astype(float).idxmax()
best_pr  = summary["PR-AUC"].astype(float).idxmax()
best_mcc = summary["MCC"].astype(float).idxmax()
best_rec = summary["Recall"].astype(float).idxmax()
print(f"\nBest ROC-AUC : {best_roc}  ({summary.loc[best_roc,'ROC-AUC']})")
print(f"Best PR-AUC  : {best_pr}  ({summary.loc[best_pr,'PR-AUC']})")
print(f"Best MCC     : {best_mcc}  ({summary.loc[best_mcc,'MCC']})")
print(f"Best Recall  : {best_rec}  ({summary.loc[best_rec,'Recall']})")

# Save leaderboard
save_rows = {n: {k: r[v] for k, v in disp_map.items()} for n, r in results.items()}
pd.DataFrame(save_rows).T.reset_index().rename(columns={"index":"Model"})\
  .to_csv(f"{OUT}\\mortality_model_comparison_v5_final.csv", index=False)

# ── 5. Feature importance ─────────────────────────────────────────────────────
# LightGBM
lgb_imp = pd.Series(results["LightGBM"]["_model"].feature_importances_,
                    index=X.columns).sort_values(ascending=False)
print(f"\nLightGBM Top 30 (v5):")
print(lgb_imp.head(30).to_string())

# XGB aucpr
xgb_imp = pd.Series(results["XGB (aucpr)"]["_model"].feature_importances_,
                    index=X.columns).sort_values(ascending=False)
print(f"\nXGB (aucpr) Top 30 (v5):")
print(xgb_imp.head(30).to_string())

# Highlight event feature rankings
event_cols = ["count_pending","count_active","count_discharges","count_discharge_hosp_snf",
              "sum_pending_days","sum_active_days","sum_onhold_days","sum_onhold_hosp_snf_days",
              "sum_waitlist_days","days_active_before_expiry",
              "onhold_ratio","hosp_hold_ratio","hosp_discharge_rate"]
print(f"\nEvent feature rankings in LightGBM:")
lgb_rank = lgb_imp.rank(ascending=False).astype(int)
for c in sorted(event_cols, key=lambda x: lgb_imp.get(x,0), reverse=True):
    if c in lgb_imp.index:
        print(f"  Rank {lgb_rank[c]:>3}/{len(lgb_imp)}: {c:<35} imp={lgb_imp[c]:.0f}")

# ── 6. Delta vs v4 (XGB aucpr baseline) ──────────────────────────────────────
try:
    v4 = pd.read_csv(f"{OUT}\\mortality_model_comparison_v4_final.csv", index_col="Model")
    delta_cols_map = {"PR-AUC":"PR-AUC", "ROC-AUC":"ROC-AUC (test)",
                      "Recall":"Recall", "MCC":"MCC"}
    print(f"\nDelta v5 vs v4 (positive = improvement):")
    for m in results:
        if m in v4.index:
            parts = []
            for k5, k4 in delta_cols_map.items():
                try:
                    d = float(results[m][k5.replace("-","_")]) - float(v4.loc[m, k4])
                    parts.append(f"{k5}:{d:+.4f}")
                except Exception:
                    pass
            print(f"  {m:<22}: {' | '.join(parts)}")
except Exception as e:
    print(f"[delta vs v4 skipped: {e}]")

# ── 7. Charts ─────────────────────────────────────────────────────────────────
COLORS = {
    "LightGBM":         "#E53935",
    "RandomForest":     "#43A047",
    "LogisticRegression":"#1E88E5",
    "MLP (Neural Net)": "#FB8C00",
    "XGB (logloss)":    "#7B1FA2",
    "XGB (aucpr)":      "#00ACC1",
}

fig = plt.figure(figsize=(26, 22))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# (a) ROC
ax_roc = fig.add_subplot(gs[0, 0])
for n, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["_y_prob"])
    ax_roc.plot(fpr, tpr, label=f"{n} ({res['ROC_AUC']:.3f})",
                color=COLORS[n], lw=2.5 if "XGB" in n else 1.5,
                linestyle="--" if n=="XGB (logloss)" else "-")
ax_roc.plot([0,1],[0,1],"k--",lw=1)
ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves (v5)")
ax_roc.legend(fontsize=6); ax_roc.grid(alpha=0.3)

# (b) PR curves
ax_pr = fig.add_subplot(gs[0, 1])
for n, res in results.items():
    p, r, _ = precision_recall_curve(y_test, res["_y_prob"])
    ax_pr.plot(r, p, label=f"{n} (PR={res['PR_AUC']:.3f})",
               color=COLORS[n], lw=2.5 if "XGB" in n else 1.5,
               linestyle="--" if n=="XGB (logloss)" else "-")
ax_pr.axhline(pos_rate, color="k", ls=":", lw=1, label=f"Baseline ({pos_rate:.3f})")
ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves (v5)")
ax_pr.legend(fontsize=6); ax_pr.grid(alpha=0.3)

# (c) PR-AUC bar chart
ax_prbar = fig.add_subplot(gs[0, 2])
pr_items = {n: float(res["PR_AUC"]) for n, res in results.items()}
pr_sorted = dict(sorted(pr_items.items(), key=lambda x: x[1]))
bars = ax_prbar.barh(list(pr_sorted.keys()), list(pr_sorted.values()),
                     color=[COLORS[n] for n in pr_sorted], alpha=0.85,
                     edgecolor=["#00ACC1" if n=="XGB (aucpr)" else "none" for n in pr_sorted],
                     linewidth=2)
ax_prbar.axvline(pos_rate, color="k", ls="--", lw=1, label=f"Baseline={pos_rate:.3f}")
ax_prbar.set(xlabel="PR-AUC", title="PR-AUC Ranking (v5)")
ax_prbar.legend(fontsize=8); ax_prbar.grid(axis="x", alpha=0.3)
for bar, val in zip(bars, pr_sorted.values()):
    ax_prbar.text(val+0.002, bar.get_y()+bar.get_height()/2,
                  f"{val:.4f}", va="center", fontsize=8)

# (d) MCC vs Recall
ax_sc = fig.add_subplot(gs[0, 3])
for n, res in results.items():
    ax_sc.scatter(float(res["Recall"]), float(res["MCC"]),
                  color=COLORS[n], s=150, zorder=5,
                  marker="*" if "XGB" in n else "o")
    ax_sc.annotate(n.replace(" (Neural Net)","").replace(" (aucpr)","*").replace(" (logloss)",""),
                   (float(res["Recall"]), float(res["MCC"])),
                   textcoords="offset points", xytext=(5,3), fontsize=7)
ax_sc.set(xlabel="Recall", ylabel="MCC", title="MCC vs Recall (v5)")
ax_sc.grid(alpha=0.3); ax_sc.axhline(0, color="k", lw=0.5)

# (e-j) Confusion matrices
cm_pos = [(1,0),(1,1),(1,2),(1,3),(2,0),(2,1)]
for idx, (name, res) in enumerate(results.items()):
    r, c = cm_pos[idx]
    ax_cm = fig.add_subplot(gs[r, c])
    disp = ConfusionMatrixDisplay(confusion_matrix=res["_cm"],
                                   display_labels=["No Mort.","Mortality"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"{name}\nROC={res['ROC_AUC']:.3f}  PR={res['PR_AUC']:.3f}"
        f"\nRecall={res['Recall']:.3f}  MCC={res['MCC']:.3f}",
        fontsize=8,
        color="#00ACC1" if "aucpr" in name else "black")
    ax_cm.tick_params(labelsize=8)

# (k) LightGBM top 25 feature importance (v5)
ax_fi = fig.add_subplot(gs[2, 2])
lgb_imp.head(25).sort_values().plot(kind="barh", ax=ax_fi,
    color=["#FF5722" if c in event_cols else "#E53935" for c in lgb_imp.head(25).sort_values().index],
    alpha=0.85)
ax_fi.set(title="LightGBM Top-25 Features (v5)\nOrange = new event features")
ax_fi.tick_params(labelsize=6.5); ax_fi.grid(axis="x", alpha=0.3)

# (l) PR-AUC evolution v1 -> v5
ax_evo = fig.add_subplot(gs[2, 3])
try:
    v1_df = pd.read_csv(f"{OUT}\\mortality_model_comparison.csv",    index_col="Model")
    v3_df = pd.read_csv(f"{OUT}\\mortality_model_comparison_v3.csv", index_col="Model")
    v4_df = pd.read_csv(f"{OUT}\\mortality_model_comparison_v4_final.csv", index_col="Model")
    shared = ["LightGBM","RandomForest","LogisticRegression","MLP (Neural Net)"]
    for m in shared:
        pts, xs = [], []
        for vi, df_v in enumerate([v1_df, v3_df, v4_df]):
            if "F1 (positive cls)" in df_v.columns:
                df_v = df_v.rename(columns={"F1 (positive cls)":"F1 (mortality)"})
            col = "PR-AUC" if "PR-AUC" in df_v.columns else None
            if col and m in df_v.index:
                pts.append(float(df_v.loc[m, col]))
                xs.append(vi)
        # v5 value
        pts.append(float(results[m]["PR_AUC"])); xs.append(3)
        ax_evo.plot(xs, pts, "o-", color=COLORS[m], lw=1.5, label=m, markersize=5)
    ax_evo.scatter([3],[float(results["XGB (aucpr)"]["PR_AUC"])],
                   color=COLORS["XGB (aucpr)"], s=150, zorder=6, marker="*",
                   label="XGB (aucpr)")
    ax_evo.set_xticks([0,1,2,3]); ax_evo.set_xticklabels(["v1","v3","v4","v5"])
    ax_evo.set(xlabel="Version", ylabel="PR-AUC", title="PR-AUC Evolution v1 -> v5")
    ax_evo.legend(fontsize=6.5); ax_evo.grid(alpha=0.3)
    ax_evo.axhline(pos_rate, color="k", ls=":", lw=1, label=f"Baseline")
except Exception as e:
    ax_evo.text(0.5,0.5,f"Evolution unavailable:\n{e}",
                transform=ax_evo.transAxes, ha="center", fontsize=8)

plt.suptitle(
    "Mortality Prediction v5 — Client Events + CCI + ICD + Q Interactions (128 features)\n"
    "20,888 clients | 6.1% mortality | 6 models | No SMOTE",
    fontsize=12, fontweight="bold", y=1.01
)
chart_path = f"{OUT}\\mortality_model_comparison_v5_final.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nChart saved: {chart_path}")
print("=== ALL DONE ===")
