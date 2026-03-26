"""
AutoML v2 — SMOTE + RobustScaler + Adverse Event Recency features
Target: Mortality_Flag  (binary, ~6.1% positive)
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

from sklearn.model_selection    import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing      import RobustScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.neural_network     import MLPClassifier
from sklearn.metrics            import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, matthews_corrcoef, average_precision_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling     import SMOTE
from imblearn.pipeline          import Pipeline as ImbPipeline
import lightgbm as lgb

OUT  = r"C:\Users\Ravi\Elara_Probus\Probus_Data"
ANALYSIS_DATE = pd.Timestamp("2026-03-26")
NEVER = 9999   # sentinel for "event never occurred / never assessed"

# ── 1. Load base ML dataset ───────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready.csv", low_memory=False)
ml["gender_enc"] = ml["gender_enc"].fillna(ml["gender_enc"].median())

# ── 2. Merge recency features ─────────────────────────────────────────────────
rec = pd.read_csv(f"{OUT}\\_recency_features.csv")
ml  = ml.merge(rec, on="ClientId", how="left")

# Clients never assessed → sensible fill values
never_assessed_mask = ml["total_assessments"].isna()
ml["total_assessments"]        = ml["total_assessments"].fillna(0).astype(int)
ml["pct_forms_with_adverse"]   = ml["pct_forms_with_adverse"].fillna(0)
ml["last_assessment_days_ago"] = ml["last_assessment_days_ago"].fillna(NEVER)
ml["avg_days_between_forms"]   = ml["avg_days_between_forms"].fillna(NEVER)
# days_since_X: never had that event → NEVER
dsince_cols = [c for c in ml.columns if c.startswith("days_since_")]
ml[dsince_cols] = ml[dsince_cols].fillna(NEVER)

print(f"After recency merge: {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"Clients with questionnaire data: {(~never_assessed_mask).sum():,} / {len(ml):,}")

# ── 3. Engineer rate features from Q scores + care duration ──────────────────
Q_EVENT_COLS = ["q_hospital_admit","q_urgent_care_er","q_fall","q_pain_swelling",
                "q_flu_symptoms","q_skin_breakdown","q_mood_changes","q_missed_meds",
                "q_bathroom_issues","q_breathing","q_other_concerns"]

safe_dur = ml["care_duration_days"].clip(lower=1)

# Events per 30 days of care (annualised rate)
ml["adverse_event_rate_per30d"] = (ml["q_score_total"] / safe_dur * 30).round(4)
ml["fall_rate_per30d"]          = (ml["q_fall"]          / safe_dur * 30).round(4)
ml["hospital_rate_per30d"]      = (ml["q_hospital_admit"] / safe_dur * 30).round(4)
ml["breathing_rate_per30d"]     = (ml["q_breathing"]      / safe_dur * 30).round(4)
ml["pain_rate_per30d"]          = (ml["q_pain_swelling"]  / safe_dur * 30).round(4)

# Recency-weighted event burden: recent events penalised more
# weight = 1 / (days_since_last_adverse_event / 30 + 1)
# → client with event 7 days ago weights ~4x more than client with event 90 days ago
ml["recency_weight"] = 1 / (ml["days_since_last_adverse_event"].clip(upper=NEVER-1) / 30 + 1)
ml["recency_weighted_burden"]   = (ml["q_score_total"] * ml["recency_weight"]).round(4)

# Escalation signal: increasing adverse events over time
# Proxy: events per assessment — captures density even when total is low
ml["events_per_assessment"] = np.where(
    ml["total_assessments"] > 0,
    ml["q_score_total"] / ml["total_assessments"],
    0
).round(4)

new_rate_cols = ["adverse_event_rate_per30d","fall_rate_per30d","hospital_rate_per30d",
                 "breathing_rate_per30d","pain_rate_per30d",
                 "recency_weight","recency_weighted_burden","events_per_assessment"]
print(f"\nNew rate/recency features: {new_rate_cols}")
print(f"Total features now: {ml.shape[1]}")

# ── 4. Train / test split ─────────────────────────────────────────────────────
TARGET  = "Mortality_Flag"
DROP    = ["ClientId","recency_weight"]      # recency_weight is an intermediate; keep burden
X = ml.drop(columns=[TARGET] + DROP)
y = ml[TARGET]

neg_pos = (y==0).sum() / (y==1).sum()
print(f"\nTarget: {y.sum():,} positive ({y.mean()*100:.1f}%)  neg:pos={neg_pos:.1f}:1")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train {len(X_train):,}  Test {len(X_test):,}  Pos in test: {y_test.sum():,}")

# ── 5. Scale: RobustScaler (median/IQR — handles extreme outliers well) ───────
scaler   = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. SMOTE on training set only ─────────────────────────────────────────────
smote = SMOTE(sampling_strategy=0.3,   # minority→30% of majority
              k_neighbors=5, random_state=42)
X_tr_sm, y_tr_sm = smote.fit_resample(X_train_scaled, y_train)
print(f"\nAfter SMOTE: {len(X_tr_sm):,} train rows | "
      f"Pos: {y_tr_sm.sum():,} ({y_tr_sm.mean()*100:.1f}%)")

# ── 7. Models (all receive pre-scaled + SMOTE'd training data) ───────────────
MODELS = {
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.04, num_leaves=63,
        max_depth=-1, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=600, max_depth=20, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        C=0.5, solver="lbfgs", max_iter=2000,
        random_state=42, n_jobs=-1,
    ),
    "MLP (Neural Net)": MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu", solver="adam", alpha=5e-4,
        learning_rate="adaptive", learning_rate_init=5e-4,
        max_iter=500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25,
        random_state=42,
    ),
}

# ── 8. Train + evaluate ───────────────────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in MODELS.items():
    print(f"\nTraining {name}...", end=" ", flush=True)
    t0 = time.time()
    model.fit(X_tr_sm, y_tr_sm)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

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
    spec      = tn / (tn + fp) if (tn+fp) > 0 else 0

    # CV on SMOTE'd training (approximate generalisation)
    cv_auc = cross_val_score(model, X_tr_sm, y_tr_sm,
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

# ── 9. Summary table ──────────────────────────────────────────────────────────
DISP = ["ROC-AUC (test)","ROC-AUC (5-CV)","PR-AUC","F1 (mortality)",
        "F1 (macro)","Precision","Recall","Specificity","Accuracy","MCC","Train time (s)"]
summary = pd.DataFrame({k:{c:v[c] for c in DISP} for k,v in results.items()}).T
summary.index.name = "Model"

print("\n" + "=" * 100)
print("SUMMARY — AutoML v2  (SMOTE + RobustScaler + Recency Features)")
print("=" * 100)
print(summary.to_string())
best = summary["ROC-AUC (test)"].idxmax()
print(f"\nBest model by ROC-AUC: {best}  ({summary.loc[best,'ROC-AUC (test)']:.4f})")

summary.reset_index().to_csv(f"{OUT}\\mortality_model_comparison_v2.csv", index=False)

# ── 10. Comparison v1 vs v2 ───────────────────────────────────────────────────
try:
    v1 = pd.read_csv(f"{OUT}\\mortality_model_comparison.csv", index_col="Model")
    delta_cols = ["ROC-AUC (test)","PR-AUC","F1 (mortality)","Recall","MCC"]
    # rename v1 col if needed
    if "F1 (positive cls)" in v1.columns:
        v1 = v1.rename(columns={"F1 (positive cls)": "F1 (mortality)"})
    comp = pd.DataFrame()
    for col in delta_cols:
        if col in v1.columns and col in summary.columns:
            comp[f"{col} v1"] = v1[col]
            comp[f"{col} v2"] = summary[col]
            comp[f"{col} delta"] = (summary[col] - v1[col]).round(4)
    print("\n" + "=" * 100)
    print("v1 vs v2 DELTA (positive = improvement)")
    print("=" * 100)
    print(comp[[c for c in comp.columns if "delta" in c]].to_string())
except Exception as e:
    print(f"[v1 compare skipped: {e}]")

# ── 11. Plots ─────────────────────────────────────────────────────────────────
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
ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
ax_roc.legend(fontsize=7); ax_roc.grid(alpha=0.3)

ax_pr = fig.add_subplot(gs[0, 1])
pos_rate = y_test.mean()
for n, res in results.items():
    p, r, _ = precision_recall_curve(y_test, res["_y_prob"])
    ax_pr.plot(r, p, label=f"{n} (PR={res['PR-AUC']:.3f})",
               color=COLORS[n], lw=2)
ax_pr.axhline(pos_rate, color="k", ls="--", lw=1, label=f"Baseline ({pos_rate:.3f})")
ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
ax_pr.legend(fontsize=7); ax_pr.grid(alpha=0.3)

ax_bar = fig.add_subplot(gs[0, 2])
metrics_bar = ["ROC-AUC (test)","PR-AUC","F1 (mortality)","Recall","Precision","MCC"]
x = np.arange(len(metrics_bar)); w = 0.18
for i, n in enumerate(results):
    vals = [results[n][m] for m in metrics_bar]
    ax_bar.bar(x + i*w, vals, width=w, label=n, color=COLORS[n], alpha=0.85)
ax_bar.set_xticks(x + w*1.5)
ax_bar.set_xticklabels(["AUC","PR-AUC","F1+","Recall","Prec","MCC"], fontsize=8, rotation=20)
ax_bar.set(title="Key Metrics (v2)"); ax_bar.legend(fontsize=7)
ax_bar.set_ylim(0,1); ax_bar.grid(axis="y", alpha=0.3)

# Feature importance — LightGBM
ax_fi = fig.add_subplot(gs[0, 3])
lgb_model = results["LightGBM"]["_model"]
feat_imp = pd.Series(lgb_model.feature_importances_, index=X.columns)\
             .sort_values(ascending=False).head(20)
feat_imp.sort_values().plot(kind="barh", ax=ax_fi, color="#E53935", alpha=0.85)
ax_fi.set(title="LightGBM Top-20 Features"); ax_fi.tick_params(labelsize=7)
ax_fi.grid(axis="x", alpha=0.3)

# Row 1-2: Confusion matrices (2 per row)
cm_positions = [(1,0),(1,1),(1,2),(1,3)]
for idx, (name, res) in enumerate(results.items()):
    r, c = cm_positions[idx]
    ax_cm = fig.add_subplot(gs[r, c])
    disp = ConfusionMatrixDisplay(confusion_matrix=res["_cm"],
                                   display_labels=["No Mort.","Mortality"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"{name}\nROC={res['ROC-AUC (test)']:.3f}  F1={res['F1 (mortality)']:.3f}  "
        f"MCC={res['MCC']:.3f}", fontsize=8)
    ax_cm.tick_params(labelsize=8)

# Row 2: v1 vs v2 delta bar chart
ax_delta = fig.add_subplot(gs[2, 0:2])
try:
    delta_metrics = ["ROC-AUC (test)","PR-AUC","F1 (mortality)","Recall","MCC"]
    v1_vals = pd.read_csv(f"{OUT}\\mortality_model_comparison.csv", index_col="Model")
    if "F1 (positive cls)" in v1_vals.columns:
        v1_vals = v1_vals.rename(columns={"F1 (positive cls)": "F1 (mortality)"})
    x2 = np.arange(len(delta_metrics)); w2 = 0.18
    for i, mname in enumerate(results):
        deltas = [results[mname][m] - float(v1_vals.loc[mname, m])
                  for m in delta_metrics if m in v1_vals.columns]
        ax_delta.bar(x2 + i*w2, deltas, width=w2, label=mname,
                     color=COLORS[mname], alpha=0.85)
    ax_delta.axhline(0, color="k", lw=0.8)
    ax_delta.set_xticks(x2 + w2*1.5)
    ax_delta.set_xticklabels(delta_metrics, fontsize=8, rotation=15)
    ax_delta.set(title="Delta v1 -> v2 (positive = improvement)")
    ax_delta.legend(fontsize=7); ax_delta.grid(axis="y", alpha=0.3)
except Exception as e:
    ax_delta.text(0.5, 0.5, f"Delta chart unavailable:\n{e}",
                  transform=ax_delta.transAxes, ha="center")

# Calibration / score distribution
ax_dist = fig.add_subplot(gs[2, 2:4])
for n, res in results.items():
    pos_probs = res["_y_prob"][y_test == 1]
    neg_probs = res["_y_prob"][y_test == 0]
    ax_dist.hist(pos_probs, bins=40, alpha=0.4, color=COLORS[n],
                 label=f"{n} (mort)", density=True, histtype="step", lw=2)
ax_dist.set(xlabel="Predicted probability", ylabel="Density",
             title="Score Distribution — Mortality=1 class")
ax_dist.legend(fontsize=7); ax_dist.grid(alpha=0.3)

plt.suptitle("Mortality AutoML v2 — SMOTE + RobustScaler + Recency Features",
             fontsize=13, fontweight="bold", y=1.01)
chart_path = f"{OUT}\\mortality_model_comparison_v2.png"
fig.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nChart saved: {chart_path}")
print("=== ALL DONE ===")
