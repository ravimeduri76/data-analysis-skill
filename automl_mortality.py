"""
AutoML Classification — mortality_ml_ready.csv
Target: Mortality_Flag  (binary, ~6.1% positive)
Models: LightGBM, RandomForest, LogisticRegression, MLPClassifier
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
from sklearn.impute             import SimpleImputer
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.neural_network     import MLPClassifier
from sklearn.metrics            import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, average_precision_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay
)
import lightgbm as lgb

OUT = r"C:\Users\Ravi\Elara_Probus\Probus_Data"

# ── 1. Load & prepare ────────────────────────────────────────────────────────
df = pd.read_csv(f"{OUT}\\mortality_ml_ready.csv", low_memory=False)
print(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")

TARGET = "Mortality_Flag"
DROP   = ["ClientId"]
X = df.drop(columns=[TARGET] + DROP)
y = df[TARGET]

pos_rate = y.mean()
neg_pos_ratio = (y == 0).sum() / (y == 1).sum()
print(f"Target: {y.sum():,} positives ({pos_rate*100:.1f}%)  |  neg:pos ratio = {neg_pos_ratio:.1f}:1")

# Impute remaining gender_enc null with median (0.7%)
X["gender_enc"] = X["gender_enc"].fillna(X["gender_enc"].median())

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  Test: {len(X_test):,}  Pos in test: {y_test.sum():,}")

# ── 2. Define models ─────────────────────────────────────────────────────────
MODELS = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio,    # handles imbalance
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "LogisticRegression": Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scl",   StandardScaler()),
        ("clf",   LogisticRegression(
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
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,          # L2 regularisation
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )),
    ]),
}

# ── 3. Train & evaluate ───────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in MODELS.items():
    print(f"\nTraining {name}...", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    # Predictions
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc    = roc_auc_score(y_test, y_prob)
    pr_auc     = average_precision_score(y_test, y_prob)
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    f1_pos     = f1_score(y_test, y_pred, pos_label=1)
    precision  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall     = recall_score(y_test, y_pred, pos_label=1)
    accuracy   = accuracy_score(y_test, y_pred)
    mcc        = matthews_corrcoef(y_test, y_pred)
    cm         = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 5-fold CV ROC-AUC on training set
    cv_auc = cross_val_score(model, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    results[name] = {
        "ROC-AUC (test)":   round(roc_auc,  4),
        "ROC-AUC (5-CV)":   round(cv_auc,   4),
        "PR-AUC":           round(pr_auc,   4),
        "F1 (positive cls)":round(f1_pos,   4),
        "F1 (macro)":       round(f1_macro, 4),
        "Precision":        round(precision,4),
        "Recall":           round(recall,   4),
        "Specificity":      round(specificity,4),
        "Accuracy":         round(accuracy, 4),
        "MCC":              round(mcc,      4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Train time (s)":   round(elapsed,  1),
        "_model":           model,
        "_y_prob":          y_prob,
        "_y_pred":          y_pred,
        "_cm":              cm,
    }
    print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1+={f1_pos:.4f}  "
          f"Recall={recall:.4f}  MCC={mcc:.4f}")
    print(f"  Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Mortality","Mortality"],
                                 digits=4))

# ── 4. Summary table ──────────────────────────────────────────────────────────
DISPLAY_COLS = ["ROC-AUC (test)","ROC-AUC (5-CV)","PR-AUC",
                "F1 (positive cls)","F1 (macro)","Precision","Recall",
                "Specificity","Accuracy","MCC","Train time (s)"]
summary = pd.DataFrame({k: {col: v[col] for col in DISPLAY_COLS}
                         for k, v in results.items()}).T
summary.index.name = "Model"
print("\n" + "=" * 90)
print("SUMMARY COMPARISON")
print("=" * 90)
print(summary.to_string())

# Best model
best_model_name = summary["ROC-AUC (test)"].idxmax()
print(f"\nBest model by ROC-AUC: {best_model_name}  ({summary.loc[best_model_name,'ROC-AUC (test)']:.4f})")

# Save summary CSV
summary.reset_index().to_csv(f"{OUT}\\mortality_model_comparison.csv", index=False)
print(f"Summary saved: {OUT}\\mortality_model_comparison.csv")

# ── 5. Plots ──────────────────────────────────────────────────────────────────
COLORS = {"LightGBM":"#E53935","RandomForest":"#43A047",
          "LogisticRegression":"#1E88E5","MLP (Neural Net)":"#FB8C00"}

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# (a) ROC curves
ax1 = fig.add_subplot(gs[0, 0])
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["_y_prob"])
    ax1.plot(fpr, tpr, label=f"{name} ({res['ROC-AUC (test)']:.3f})",
             color=COLORS[name], lw=2)
ax1.plot([0,1],[0,1],"k--",lw=1)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

# (b) Precision-Recall curves
ax2 = fig.add_subplot(gs[0, 1])
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["_y_prob"])
    ax2.plot(rec, prec, label=f"{name} (PR-AUC={res['PR-AUC']:.3f})",
             color=COLORS[name], lw=2)
ax2.axhline(pos_rate, color="k", ls="--", lw=1, label=f"Baseline ({pos_rate:.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# (c) Metrics bar chart
ax3 = fig.add_subplot(gs[0, 2])
metrics_to_plot = ["ROC-AUC (test)","PR-AUC","F1 (positive cls)","Recall","Precision","MCC"]
model_names = list(results.keys())
x = np.arange(len(metrics_to_plot))
w = 0.18
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics_to_plot]
    ax3.bar(x + i*w, vals, width=w, label=name, color=COLORS[name], alpha=0.85)
ax3.set_xticks(x + w*1.5)
ax3.set_xticklabels(["ROC-AUC","PR-AUC","F1+","Recall","Precision","MCC"],
                     fontsize=8, rotation=20)
ax3.set_title("Key Metrics Comparison"); ax3.legend(fontsize=7); ax3.set_ylim(0,1)
ax3.grid(axis="y", alpha=0.3)

# (d-g) Confusion matrices
cm_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
# we have 4 models — add one more subplot
ax_extra = fig.add_subplot(gs[1, 2])  # overwrite last; use inset
for idx, (name, res) in enumerate(results.items()):
    if idx < 3:
        ax_cm = cm_axes[idx]
    else:
        # 4th model: inset inside ax3 area
        ax_cm = fig.add_axes([0.68, 0.08, 0.28, 0.28])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res["_cm"],
        display_labels=["No Mort.","Mortality"]
    )
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(f"{name}\nROC={res['ROC-AUC (test)']:.3f}", fontsize=9)
    ax_cm.tick_params(labelsize=8)

plt.suptitle("Mortality Prediction — AutoML Model Comparison", fontsize=14, fontweight="bold", y=1.01)
chart_path = f"{OUT}\\mortality_model_comparison.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved: {chart_path}")

# ── 6. Feature importance (LightGBM) ─────────────────────────────────────────
lgb_model = results["LightGBM"]["_model"]
feat_imp = pd.Series(lgb_model.feature_importances_, index=X.columns)\
             .sort_values(ascending=False).head(25)

fig2, ax = plt.subplots(figsize=(10, 7))
feat_imp.sort_values().plot(kind="barh", ax=ax, color="#E53935", alpha=0.85)
ax.set_title("LightGBM — Top 25 Feature Importances")
ax.set_xlabel("Importance (gain)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
imp_path = f"{OUT}\\mortality_feature_importance.png"
fig2.savefig(imp_path, dpi=150)
plt.close(fig2)
print(f"Feature importance saved: {imp_path}")

print("\n=== ALL DONE ===")
