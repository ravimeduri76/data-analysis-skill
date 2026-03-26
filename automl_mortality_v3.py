"""
AutoML v3 — CCI + ICD interactions + Questionnaire interactions
Target: Mortality_Flag  (binary, ~6.1% positive)
No SMOTE — class_weight='balanced' / scale_pos_weight for all models
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

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.impute            import SimpleImputer
from sklearn.pipeline          import Pipeline
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, average_precision_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay
)
import lightgbm as lgb

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

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Pos in test: {y_test.sum():,}")

# ── 2. Define models (class_weight / scale_pos_weight — NO SMOTE) ─────────────
MODELS = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,   # handles imbalance
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    "MLP (Neural Net)": Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=5e-4,
            learning_rate="adaptive",
            learning_rate_init=5e-4,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=42,
        )),
    ]),
}

# ── 3. Train + evaluate ───────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in MODELS.items():
    print(f"\nTraining {name}...", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc   = roc_auc_score(y_test, y_prob)
    pr_auc    = average_precision_score(y_test, y_prob)
    f1_pos    = f1_score(y_test, y_pred, pos_label=1)
    f1_macro  = f1_score(y_test, y_pred, average="macro")
    prec      = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec       = recall_score(y_test, y_pred, pos_label=1)
    acc       = accuracy_score(y_test, y_pred)
    mcc       = matthews_corrcoef(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)
    tn,fp,fn,tp = cm.ravel()
    spec      = tn / (tn + fp) if (tn + fp) > 0 else 0

    cv_auc = cross_val_score(model, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    results[name] = {
        "ROC-AUC (test)": round(roc_auc,4), "ROC-AUC (5-CV)": round(cv_auc,4),
        "PR-AUC": round(pr_auc,4),
        "F1 (mortality)": round(f1_pos,4),  "F1 (macro)": round(f1_macro,4),
        "Precision": round(prec,4),          "Recall": round(rec,4),
        "Specificity": round(spec,4),         "Accuracy": round(acc,4),
        "MCC": round(mcc,4),
        "TP":tp,"FP":fp,"TN":tn,"FN":fn,
        "Train time (s)": round(elapsed,1),
        "_model":model, "_y_prob":y_prob, "_y_pred":y_pred, "_cm":cm,
    }
    print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1+={f1_pos:.4f}  "
          f"Recall={rec:.4f}  Precision={prec:.4f}  MCC={mcc:.4f}")
    print(f"  Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(classification_report(y_test, y_pred,
          target_names=["No Mortality","Mortality"], digits=4))

# ── 4. Summary table ──────────────────────────────────────────────────────────
DISP = ["ROC-AUC (test)","ROC-AUC (5-CV)","PR-AUC","F1 (mortality)",
        "F1 (macro)","Precision","Recall","Specificity","Accuracy","MCC","Train time (s)"]
summary = pd.DataFrame({k:{c:v[c] for c in DISP} for k,v in results.items()}).T
summary.index.name = "Model"

print("\n" + "=" * 100)
print("SUMMARY — AutoML v3  (CCI + ICD interactions + Q interactions | class_weight balanced)")
print("=" * 100)
print(summary.to_string())
best = summary["ROC-AUC (test)"].idxmax()
print(f"\nBest by ROC-AUC: {best}  ({summary.loc[best,'ROC-AUC (test)']:.4f})")
best_pr = summary["PR-AUC"].idxmax()
print(f"Best by PR-AUC : {best_pr}  ({summary.loc[best_pr,'PR-AUC']:.4f})")

summary.reset_index().to_csv(f"{OUT}\\mortality_model_comparison_v3.csv", index=False)

# ── 5. Three-way comparison: v1 vs v2 vs v3 ──────────────────────────────────
try:
    v1 = pd.read_csv(f"{OUT}\\mortality_model_comparison.csv",    index_col="Model")
    v2 = pd.read_csv(f"{OUT}\\mortality_model_comparison_v2.csv", index_col="Model")
    if "F1 (positive cls)" in v1.columns:
        v1 = v1.rename(columns={"F1 (positive cls)": "F1 (mortality)"})
    if "F1 (positive cls)" in v2.columns:
        v2 = v2.rename(columns={"F1 (positive cls)": "F1 (mortality)"})

    delta_cols = ["ROC-AUC (test)", "PR-AUC", "F1 (mortality)", "Recall", "MCC"]
    print("\n" + "=" * 100)
    print("THREE-WAY COMPARISON  (v3 - v1 delta | v3 - v2 delta)")
    print("=" * 100)
    for col in delta_cols:
        if col in v1.columns and col in v2.columns:
            d1 = (summary[col] - v1[col]).round(4)
            d2 = (summary[col] - v2[col]).round(4)
            print(f"\n{col}:")
            for m in summary.index:
                print(f"  {m:<22} v3={float(summary.loc[m,col]):.4f}  "
                      f"Δv1={float(d1.get(m,0)):+.4f}  Δv2={float(d2.get(m,0)):+.4f}")
except Exception as e:
    print(f"[3-way compare skipped: {e}]")

# ── 6. Feature importance (LightGBM top 30) ──────────────────────────────────
lgb_model = results["LightGBM"]["_model"]
feat_imp = pd.Series(lgb_model.feature_importances_, index=X.columns)\
             .sort_values(ascending=False)

print(f"\nLightGBM Top 30 features:")
print(feat_imp.head(30).to_string())

# Check where new features rank
new_features = [c for c in X.columns if any(c.startswith(p) for p in
    ["ix_","qx_","cci_","CHF","Diabetes","Renal","Liver","Cancer","chf_","diabetes_"])]
print(f"\nNew features (CCI/interactions) importance ranking:")
ranked = feat_imp.rank(ascending=False).astype(int)
for c in sorted(new_features, key=lambda x: feat_imp.get(x, 0), reverse=True):
    if c in feat_imp.index:
        print(f"  Rank {ranked[c]:>3}: {c:<30} importance={feat_imp[c]:.0f}")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
COLORS = {"LightGBM":"#E53935","RandomForest":"#43A047",
          "LogisticRegression":"#1E88E5","MLP (Neural Net)":"#FB8C00"}

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: ROC | PR | Metrics bar | Feature importance
ax_roc = fig.add_subplot(gs[0, 0])
for n, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["_y_prob"])
    ax_roc.plot(fpr, tpr, label=f"{n} ({res['ROC-AUC (test)']:.3f})",
                color=COLORS[n], lw=2)
ax_roc.plot([0,1],[0,1],"k--",lw=1)
ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves (v3)")
ax_roc.legend(fontsize=7); ax_roc.grid(alpha=0.3)

ax_pr = fig.add_subplot(gs[0, 1])
for n, res in results.items():
    p, r, _ = precision_recall_curve(y_test, res["_y_prob"])
    ax_pr.plot(r, p, label=f"{n} (PR={res['PR-AUC']:.3f})",
               color=COLORS[n], lw=2)
ax_pr.axhline(pos_rate, color="k", ls="--", lw=1, label=f"Baseline ({pos_rate:.3f})")
ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves (v3)")
ax_pr.legend(fontsize=7); ax_pr.grid(alpha=0.3)

ax_bar = fig.add_subplot(gs[0, 2])
metrics_bar = ["ROC-AUC (test)","PR-AUC","F1 (mortality)","Recall","Precision","MCC"]
x = np.arange(len(metrics_bar)); w = 0.18
for i, n in enumerate(results):
    vals = [results[n][m] for m in metrics_bar]
    ax_bar.bar(x + i*w, vals, width=w, label=n, color=COLORS[n], alpha=0.85)
ax_bar.set_xticks(x + w*1.5)
ax_bar.set_xticklabels(["AUC","PR-AUC","F1+","Recall","Prec","MCC"], fontsize=8, rotation=20)
ax_bar.set(title="Key Metrics (v3)"); ax_bar.legend(fontsize=7)
ax_bar.set_ylim(0, 1); ax_bar.grid(axis="y", alpha=0.3)

# LightGBM top 25 feature importance
ax_fi = fig.add_subplot(gs[0, 3])
feat_imp.head(25).sort_values().plot(kind="barh", ax=ax_fi, color="#E53935", alpha=0.85)
ax_fi.set(title="LightGBM Top-25 Features (v3)")
ax_fi.tick_params(labelsize=6); ax_fi.grid(axis="x", alpha=0.3)

# Row 1: Confusion matrices
for idx, (name, res) in enumerate(results.items()):
    ax_cm = fig.add_subplot(gs[1, idx])
    disp = ConfusionMatrixDisplay(confusion_matrix=res["_cm"],
                                   display_labels=["No Mort.","Mortality"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"{name}\nROC={res['ROC-AUC (test)']:.3f}  PR={res['PR-AUC']:.3f}"
        f"\nF1={res['F1 (mortality)']:.3f}  MCC={res['MCC']:.3f}", fontsize=8)
    ax_cm.tick_params(labelsize=8)

# Row 2: v1→v3 delta bar | New feature importance highlight
ax_delta = fig.add_subplot(gs[2, 0:2])
try:
    v1_reload = pd.read_csv(f"{OUT}\\mortality_model_comparison.csv", index_col="Model")
    if "F1 (positive cls)" in v1_reload.columns:
        v1_reload = v1_reload.rename(columns={"F1 (positive cls)": "F1 (mortality)"})
    delta_metrics = ["ROC-AUC (test)","PR-AUC","F1 (mortality)","Recall","MCC"]
    x2 = np.arange(len(delta_metrics)); w2 = 0.18
    for i, mname in enumerate(results):
        deltas = [results[mname][m] - float(v1_reload.loc[mname, m])
                  for m in delta_metrics if m in v1_reload.columns]
        ax_delta.bar(x2 + i*w2, deltas, width=w2, label=mname,
                     color=COLORS[mname], alpha=0.85)
    ax_delta.axhline(0, color="k", lw=0.8)
    ax_delta.set_xticks(x2 + w2*1.5)
    ax_delta.set_xticklabels(delta_metrics, fontsize=8, rotation=15)
    ax_delta.set(title="Delta v1 -> v3 (positive = improvement)")
    ax_delta.legend(fontsize=7); ax_delta.grid(axis="y", alpha=0.3)
except Exception as e:
    ax_delta.text(0.5, 0.5, f"Delta unavailable:\n{e}",
                  transform=ax_delta.transAxes, ha="center")

# New feature importance
ax_new = fig.add_subplot(gs[2, 2:4])
new_feats_in_model = [c for c in new_features if c in feat_imp.index]
if new_feats_in_model:
    new_imp = feat_imp[new_feats_in_model].sort_values()
    colors_new = ["#9C27B0" if c.startswith("qx_") else
                  "#FF9800" if c.startswith("ix_") else "#009688"
                  for c in new_imp.index]
    new_imp.plot(kind="barh", ax=ax_new, color=colors_new, alpha=0.85)
    ax_new.set(title="New Feature Importances (v3)\nPurple=Q interactions | Orange=ICD interactions | Teal=CCI")
    ax_new.tick_params(labelsize=8); ax_new.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax_new.legend(handles=[
        Patch(color="#9C27B0", label="Q interactions"),
        Patch(color="#FF9800", label="ICD interactions"),
        Patch(color="#009688", label="CCI features"),
    ], fontsize=8)

plt.suptitle("Mortality AutoML v3 — CCI + ICD × ICD + Questionnaire Interactions | No SMOTE",
             fontsize=12, fontweight="bold", y=1.01)
chart_path = f"{OUT}\\mortality_model_comparison_v3.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nChart saved: {chart_path}")
print("=== ALL DONE ===")
