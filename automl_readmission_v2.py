"""
automl_readmission_v2_clean.py
Two-pass AutoML on readmission_ml_ready_v2.csv — leaky features excluded.

Pass 1 — Readmission_Flag (>=1 hosp/SNF event)   positive rate ~14.2%
Pass 2 — True_Readmission_Flag (>1 hosp/SNF event) positive rate ~2.8%

New features vs v1:
  days_since_last_hosp  (9999 = never hospitalised)
  last_hosp_los         (LOS in days of most recent event; 0 = never)
  mean_hosp_los         (mean LOS across all events; 0 = never)
  total_hosp_events     (count of qualifying events; 0 = never)
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

OUT    = r"C:\Users\Ravi\Elara_Probus\Probus_Data"
OUTIMG = r"C:\Users\Ravi\research\data-analysis\output"
import os; os.makedirs(OUTIMG, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\readmission_ml_ready_v2.csv", low_memory=False)
print(f"Loaded: {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"Readmission_Flag=1      : {ml['Readmission_Flag'].sum():,} ({ml['Readmission_Flag'].mean()*100:.1f}%)")
print(f"True_Readmission_Flag=1 : {ml['True_Readmission_Flag'].sum():,} ({ml['True_Readmission_Flag'].mean()*100:.1f}%)")

# ── Shared helpers ────────────────────────────────────────────────────────────
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, matthews_corrcoef,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

COLORS = {
    "LightGBM":          "#1565C0",
    "RandomForest":      "#2E7D32",
    "LogisticRegression":"#F57F17",
    "MLP":               "#6A1B9A",
    "XGB (logloss)":     "#AD1457",
    "XGB (aucpr)":       "#00838F",
}

def evaluate(name, y_true, y_prob, y_pred=None):
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    roc  = roc_auc_score(y_true, y_prob)
    pr   = average_precision_score(y_true, y_prob)
    f1p  = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = (y_pred[y_true == 1] == 1).mean()
    mcc  = matthews_corrcoef(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return dict(Model=name, ROC_AUC=round(roc,4), PR_AUC=round(pr,4),
                F1_pos=round(f1p,4), Recall=round(rec,4),
                Specificity=round(spec,4), MCC=round(mcc,4),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))


def run_automl(ml_df, target_col, pass_label, out_csv, out_img_prefix):
    print(f"\n{'='*70}")
    print(f"PASS: {pass_label}  |  target={target_col}")
    print(f"{'='*70}")

    DROP_ALWAYS = {"ClientId", "Readmission_Flag", "True_Readmission_Flag"}
    feat_cols = [c for c in ml_df.columns if c not in DROP_ALWAYS]
    const_cols = [c for c in feat_cols if ml_df[c].nunique() <= 1]
    if const_cols:
        print(f"Dropping {len(const_cols)} constant col(s): {const_cols}")
    feat_cols = [c for c in feat_cols if c not in const_cols]
    print(f"Feature matrix: {len(feat_cols)} features")

    X = ml_df[feat_cols].copy()
    y = ml_df[target_col].astype(int)

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )
    neg_pos = int((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"Train: {X_tr.shape[0]:,}  Test: {X_te.shape[0]:,}  "
          f"Pos%: {y_tr.mean()*100:.1f}%  neg_pos_ratio: {neg_pos}")

    results, curves = [], {}

    # LightGBM
    lgb_m = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, verbose=-1
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    p = lgb_m.predict_proba(X_te)[:, 1]
    r = evaluate("LightGBM", y_te, p, lgb_m.predict(X_te))
    results.append(r); curves["LightGBM"] = (roc_curve(y_te, p)[:2], precision_recall_curve(y_te, p)[:2])
    print(f"[LightGBM]     ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  Recall={r['Recall']}  MCC={r['MCC']}")

    # RandomForest
    rf_m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1, random_state=42)
    rf_m.fit(X_tr, y_tr)
    p = rf_m.predict_proba(X_te)[:, 1]
    r = evaluate("RandomForest", y_te, p, rf_m.predict(X_te))
    results.append(r); curves["RandomForest"] = (roc_curve(y_te, p)[:2], precision_recall_curve(y_te, p)[:2])
    print(f"[RandomForest] ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  Recall={r['Recall']}  MCC={r['MCC']}")

    # LogisticRegression
    lr_p = Pipeline([("sc", RobustScaler()),
                     ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                               max_iter=1000, random_state=42))])
    lr_p.fit(X_tr, y_tr)
    p = lr_p.predict_proba(X_te)[:, 1]
    r = evaluate("LogisticRegression", y_te, p, lr_p.predict(X_te))
    results.append(r); curves["LogisticRegression"] = (roc_curve(y_te, p)[:2], precision_recall_curve(y_te, p)[:2])
    print(f"[LogReg]       ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  Recall={r['Recall']}  MCC={r['MCC']}")

    # MLP
    sw = np.where(y_tr == 1, neg_pos, 1).astype(float)
    mlp_p = Pipeline([("sc", RobustScaler()),
                      ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                            learning_rate_init=0.001, max_iter=300,
                                            early_stopping=True, validation_fraction=0.1,
                                            random_state=42))])
    mlp_p.fit(X_tr, y_tr, mlp__sample_weight=sw)
    p = mlp_p.predict_proba(X_te)[:, 1]
    r = evaluate("MLP", y_te, p, mlp_p.predict(X_te))
    results.append(r); curves["MLP"] = (roc_curve(y_te, p)[:2], precision_recall_curve(y_te, p)[:2])
    print(f"[MLP]          ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  Recall={r['Recall']}  MCC={r['MCC']}")

    # XGBoost
    XGB_BASE = dict(n_estimators=1000, learning_rate=0.03, max_depth=6, min_child_weight=5,
                    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                    scale_pos_weight=neg_pos, use_label_encoder=False, random_state=42, n_jobs=-1)
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.1,
                                                    random_state=42, stratify=y_tr)
    for metric, label in [("logloss","XGB (logloss)"), ("aucpr","XGB (aucpr)")]:
        xm = xgb.XGBClassifier(**XGB_BASE, eval_metric=metric,
                                early_stopping_rounds=40, verbosity=0)
        xm.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
        p = xm.predict_proba(X_te)[:, 1]
        r = evaluate(label, y_te, p, (p >= 0.5).astype(int))
        r["best_iter"] = xm.best_iteration
        results.append(r); curves[label] = (roc_curve(y_te, p)[:2], precision_recall_curve(y_te, p)[:2])
        print(f"[{label}]  ROC={r['ROC_AUC']}  PR={r['PR_AUC']}  "
              f"Recall={r['Recall']}  MCC={r['MCC']}  best_iter={xm.best_iteration}")

    # Leaderboard
    lb = pd.DataFrame(results).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    lb.index += 1
    print(f"\n=== LEADERBOARD — {pass_label} ===")
    print(lb[["Model","ROC_AUC","PR_AUC","F1_pos","Recall","Specificity","MCC"]].to_string())
    save_csv(lb, out_csv)

    # Feature importance (LightGBM)
    fi = pd.DataFrame({"feature": feat_cols,
                        "importance": lgb_m.feature_importances_}
                      ).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.index += 1
    print(f"\nTop 20 features — {pass_label}:")
    print(fi.head(20).to_string())

    # 9-panel chart
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline = y_te.mean()
    fig = plt.figure(figsize=(22, 22))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_roc = fig.add_subplot(gs[0,0])
    ax_pr  = fig.add_subplot(gs[0,1])
    ax_bar = fig.add_subplot(gs[0,2])
    ax_pr.axhline(baseline, color="gray", ls="--", lw=1, label=f"Baseline ({baseline:.3f})")
    for name, ((fpr,tpr),(prec,rec)) in curves.items():
        c = COLORS.get(name,"black")
        row = lb[lb["Model"]==name].iloc[0]
        ax_roc.plot(fpr,tpr,color=c,lw=1.5,label=f"{name} ({row['ROC_AUC']:.3f})")
        ax_pr.plot(rec,prec,color=c,lw=1.5,label=f"{name} ({row['PR_AUC']:.3f})")
    ax_roc.plot([0,1],[0,1],"k--",lw=0.8)
    ax_roc.set(title=f"ROC — {pass_label}", xlabel="FPR", ylabel="TPR"); ax_roc.legend(fontsize=7)
    ax_pr.set(title=f"PR — {pass_label}", xlabel="Recall", ylabel="Precision"); ax_pr.legend(fontsize=7)
    bars = ax_bar.bar(range(len(lb)), lb["PR_AUC"].tolist(),
                      color=[COLORS.get(m,"gray") for m in lb["Model"].tolist()])
    ax_bar.set_xticks(range(len(lb)))
    ax_bar.set_xticklabels([m.replace(" ","\n") for m in lb["Model"].tolist()], fontsize=8)
    ax_bar.axhline(baseline, color="gray", ls="--", lw=1, label=f"Baseline ({baseline:.3f})")
    ax_bar.set_title(f"PR-AUC — {pass_label}"); ax_bar.set_ylabel("PR-AUC"); ax_bar.legend(fontsize=8)
    for bar, val in zip(bars, lb["PR_AUC"].tolist()):
        ax_bar.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax_sc = fig.add_subplot(gs[1,0])
    for _, row in lb.iterrows():
        c = COLORS.get(row["Model"],"black")
        ax_sc.scatter(row["Recall"], row["MCC"], color=c, s=100, zorder=5)
        ax_sc.annotate(row["Model"].replace(" ","\n"), (row["Recall"],row["MCC"]),
                       fontsize=6, xytext=(4,2), textcoords="offset points")
    ax_sc.set(title=f"MCC vs Recall — {pass_label}", xlabel="Recall", ylabel="MCC")
    ax_sc.grid(True, alpha=0.3)
    top4 = lb.head(4)["Model"].tolist()
    cm_axes = [fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2]),
               fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]
    for ax_cm, mname in zip(cm_axes, top4):
        row = lb[lb["Model"]==mname].iloc[0]
        cm_v = np.array([[row["TN"],row["FP"]],[row["FN"],row["TP"]]])
        ax_cm.imshow(cm_v, cmap="Blues")
        ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
        ax_cm.set_xticklabels(["Pred 0","Pred 1"]); ax_cm.set_yticklabels(["True 0","True 1"])
        for i in range(2):
            for j in range(2):
                ax_cm.text(j,i,str(cm_v[i,j]),ha="center",va="center",fontsize=11,
                           color="white" if cm_v[i,j]>cm_v.max()/2 else "black")
        ax_cm.set_title(f"{mname}\nPR={row['PR_AUC']:.3f} Rec={row['Recall']:.3f}", fontsize=9)
    ax_fi = fig.add_subplot(gs[2,2])
    top20 = fi.head(20)
    ax_fi.barh(top20["feature"][::-1], top20["importance"][::-1], color="#1565C0")
    ax_fi.set_title(f"LightGBM Top-20 Features — {pass_label}", fontsize=10)
    ax_fi.set_xlabel("Importance"); ax_fi.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"{pass_label}  |  Pos rate: {baseline*100:.1f}%  |  N={len(y_te):,} test rows",
                 fontsize=13, y=0.98)
    chart = f"{OUTIMG}\\{out_img_prefix}_{ts}.png"
    fig.savefig(chart, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Chart saved: {chart}")
    return lb, fi


# ── Pass 1: Readmission_Flag ──────────────────────────────────────────────────
lb1, fi1 = run_automl(
    ml, "Readmission_Flag",
    "Readmission v2 (>=1 event, 14.2%)",
    f"{OUT}\\readmission_model_comparison_v2.csv",
    "readmission_model_comparison_v2"
)

# ── Pass 2: True_Readmission_Flag ─────────────────────────────────────────────
lb2, fi2 = run_automl(
    ml, "True_Readmission_Flag",
    "True Readmission v1 (>1 event, 2.8%)",
    f"{OUT}\\true_readmission_model_comparison_v1.csv",
    "true_readmission_model_comparison_v1"
)

# ── Side-by-side comparison ───────────────────────────────────────────────────
print("\n" + "="*70)
print("PASS 1 vs PASS 2 — PR-AUC & MCC comparison")
print("="*70)
print(f"{'Model':<22}  {'Pass1 PR':>9}  {'Pass1 MCC':>10}  {'Pass2 PR':>9}  {'Pass2 MCC':>10}")
for m in ["LightGBM","RandomForest","LogisticRegression","MLP","XGB (logloss)","XGB (aucpr)"]:
    r1 = lb1[lb1["Model"]==m].iloc[0] if m in lb1["Model"].values else None
    r2 = lb2[lb2["Model"]==m].iloc[0] if m in lb2["Model"].values else None
    p1  = f"{r1['PR_AUC']:.4f}" if r1 is not None else "n/a"
    m1  = f"{r1['MCC']:.4f}"    if r1 is not None else "n/a"
    p2  = f"{r2['PR_AUC']:.4f}" if r2 is not None else "n/a"
    m2  = f"{r2['MCC']:.4f}"    if r2 is not None else "n/a"
    print(f"  {m:<20}  {p1:>9}  {m1:>10}  {p2:>9}  {m2:>10}")

print("\n=== DONE ===")
