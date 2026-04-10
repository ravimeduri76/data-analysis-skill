"""RA4: Iteration 2 -- Focus on OBJ1 (Optuna+CatBoost), OBJ2a (deeper), OBJ2b (binary reframe)."""
import sys, os, warnings, time, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\Ravi\research\data-analysis')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\Ravi\Data_Analysis\.env')
from loader import load_dataframe

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

mcc_scorer = make_scorer(matthews_corrcoef)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def manual_cv_mcc(model_fn, X, y, cv=cv5):
    """Manual CV for models that can't be cloned by sklearn (e.g. CatBoost with cat_features)."""
    scores = []
    is_df = isinstance(X, pd.DataFrame)
    for train_idx, test_idx in cv.split(X, y):
        m = model_fn()
        if is_df:
            m.fit(X.iloc[train_idx], y[train_idx])
            preds = m.predict(X.iloc[test_idx])
        else:
            m.fit(X[train_idx], y[train_idx])
            preds = m.predict(X[test_idx])
        scores.append(matthews_corrcoef(y[test_idx], preds))
    return np.array(scores)

# Load data
match_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\matching_data.csv')
entero_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\enterobacteriaceae.csv')
target = 'Combined_Result_Culture_Result'
EXCLUDE = {'UrineCulture', 'UrineCulture_Yes_No', 'UrineNOOID', 'PatientUniqueID'}
obj1_cols = [c for c in match_df.select_dtypes(include=[np.number]).columns
             if c not in EXCLUDE and c != target]

ternary_cols = []
for c in obj1_cols:
    uvals = set(match_df[c].dropna().unique())
    if uvals <= {1.0, 2.0, 3.0} and len(uvals) == 3:
        ternary_cols.append(c)

all_results = []
def record(exp_id, obj, mcc_mean, mcc_std, model_name, notes=""):
    r = {"exp_id": exp_id, "objective": obj, "mcc_mean": round(mcc_mean, 4),
         "mcc_std": round(mcc_std, 4), "model": model_name, "notes": notes}
    all_results.append(r)
    print(f"  [{exp_id}] {model_name:30s} MCC={mcc_mean:.4f}+/-{mcc_std:.4f}  {notes}")

X1_raw = match_df[obj1_cols].fillna(0).values
y1 = match_df[target].values

# ================================================================
# OBJ1: ITERATION 2
# ================================================================
print("=" * 70)
print("OBJ1 ITERATION 2: CatBoost + Optuna LightGBM")
print("=" * 70)

# EXP-M1: CatBoost (manual CV to avoid sklearn clone issue)
print("\nEXP-M1: CatBoost with native categorical handling")
# CatBoost needs DataFrame with int-typed categoricals (not float numpy array)
X1_cb_df = match_df[obj1_cols].fillna(0).copy()
for c in ternary_cols:
    if c in X1_cb_df.columns:
        X1_cb_df[c] = X1_cb_df[c].astype(int).astype(str)  # string type for CatBoost cat
cat_names = [c for c in ternary_cols if c in obj1_cols]
def make_cb1():
    return CatBoostClassifier(iterations=150, auto_class_weights='Balanced', random_seed=42,
                               cat_features=cat_names, verbose=0, thread_count=-1)
scores = manual_cv_mcc(make_cb1, X1_cb_df, y1)
record("EXP-M1", "OBJ1", scores.mean(), scores.std(), "CatBoost_native")

# EXP-P1: Optuna LightGBM
print("\nEXP-P1: Optuna LightGBM (50 trials)")
def obj1_optuna(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_validate(model, X1_raw, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    return scores['test_mcc'].mean()

study1 = optuna.create_study(direction='maximize')
study1.optimize(obj1_optuna, n_trials=20)
record("EXP-P1", "OBJ1", study1.best_value, 0.0, "LightGBM_optuna",
       f"best_trial={study1.best_trial.number}")

# Voting ensemble of best models (use LightGBM + GBM, skip CatBoost in voter to avoid clone issue)
print("\nEXP-M5: Soft voting ensemble (LightGBM + GBM + ExtraTrees)")
best_lgbm_params = study1.best_params.copy()
best_lgbm_params.update({'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
voter = VotingClassifier(estimators=[
    ('lgbm', LGBMClassifier(**best_lgbm_params)),
    ('gbm', GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
], voting='soft', n_jobs=1)
cv_r = cross_validate(voter, X1_raw, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=1)
record("EXP-M5", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "VotingEnsemble_3model")

# ================================================================
# OBJ2a: ITERATION 2
# ================================================================
print("\n" + "=" * 70)
print("OBJ2a ITERATION 2: Optuna + MLP + KNN + CatBoost")
print("=" * 70)

pos_df = match_df[match_df[target] == 1].copy()
entero_ids = set(entero_df['PatientUniqueID'])
pos_df['Enterobacteriaceae'] = pos_df['PatientUniqueID'].isin(entero_ids).astype(int)
obj2a_cols = [c for c in obj1_cols if c != 'BloodCulture']
X2a = pos_df[obj2a_cols].fillna(0).values
y2a = pos_df['Enterobacteriaceae'].values

# Optuna GBM for OBJ2a
print("\nEXP-P2: Optuna GBM (50 trials)")
def obj2a_optuna(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.3, 1.0),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_validate(model, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    return scores['test_mcc'].mean()

study2a = optuna.create_study(direction='maximize')
study2a.optimize(obj2a_optuna, n_trials=20)
record("EXP-P2", "OBJ2a", study2a.best_value, 0.0, "GBM_optuna",
       f"best_trial={study2a.best_trial.number}")

# Optuna LightGBM for OBJ2a
print("\nEXP-P3: Optuna LightGBM for OBJ2a (50 trials)")
def obj2a_lgbm_optuna(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 60),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_validate(model, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    return scores['test_mcc'].mean()

study2a_lgbm = optuna.create_study(direction='maximize')
study2a_lgbm.optimize(obj2a_lgbm_optuna, n_trials=20)
record("EXP-P3", "OBJ2a", study2a_lgbm.best_value, 0.0, "LightGBM_optuna",
       f"best_trial={study2a_lgbm.best_trial.number}")

# MLP
print("\nEXP-M6: MLP Neural Network")
pipe_mlp = Pipeline([('scaler', StandardScaler()),
    ('clf', MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True))])
cv_r = cross_validate(pipe_mlp, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-M6", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "MLP_128_64_32")

# KNN
print("\nEXP-M7: KNN")
pipe_knn = Pipeline([('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1))])
cv_r = cross_validate(pipe_knn, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-M7", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "KNN_15")

# CatBoost for OBJ2a (manual CV)
print("\nEXP-M8: CatBoost for bacteria prediction")
ternary_2a = [c for c in ternary_cols if c in obj2a_cols]
X2a_cb_df = pos_df[obj2a_cols].fillna(0).copy()
for c in ternary_2a:
    X2a_cb_df[c] = X2a_cb_df[c].astype(int).astype(str)
cat_names_2a = ternary_2a
def make_cb2a():
    return CatBoostClassifier(iterations=150, auto_class_weights='Balanced', random_seed=42,
                               cat_features=cat_names_2a, verbose=0, thread_count=-1)
scores = manual_cv_mcc(make_cb2a, X2a_cb_df, y2a)
record("EXP-M8", "OBJ2a", scores.mean(), scores.std(), "CatBoost_bacteria")

# Recoded features + best model
print("\nEXP-D1b: Recoded features for OBJ2a")
df_recoded_2a = pos_df[obj2a_cols].copy()
for c in ternary_2a:
    df_recoded_2a[c] = df_recoded_2a[c].map({1.0: 1, 2.0: 0, 3.0: np.nan})
for c in df_recoded_2a.columns:
    if df_recoded_2a[c].isna().any():
        mode_val = df_recoded_2a[c].mode()
        df_recoded_2a[c].fillna(mode_val[0] if len(mode_val) > 0 else 0, inplace=True)
X2a_recoded = df_recoded_2a.values

model = GradientBoostingClassifier(**study2a.best_params, random_state=42)
cv_r = cross_validate(model, X2a_recoded, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-D1b", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "GBM_optuna_recoded")

# ================================================================
# OBJ2b: ITERATION 2 -- Try Optuna on multi-class
# ================================================================
print("\n" + "=" * 70)
print("OBJ2b ITERATION 2: Optuna + CatBoost + top-5 classes")
print("=" * 70)

org_df = pos_df[pos_df['UrineNOOID'].notna()].copy()
org_counts = org_df['UrineNOOID'].value_counts()
top5_codes = org_counts.head(5).index.tolist()
org_df['org_top5'] = org_df['UrineNOOID'].apply(lambda x: int(x) if x in top5_codes else 0)
le5 = LabelEncoder()
y2b_5 = le5.fit_transform(org_df['org_top5'].values)
X2b = org_df[obj2a_cols].fillna(0).values
n_classes5 = len(le5.classes_)

# CatBoost multi-class (manual CV)
print("\nEXP-M9: CatBoost multi-class")
ternary_2b = [c for c in ternary_cols if c in obj2a_cols]
X2b_cb_df = org_df[obj2a_cols].fillna(0).copy()
for c in ternary_2b:
    X2b_cb_df[c] = X2b_cb_df[c].astype(int).astype(str)
cat_names_2b = ternary_2b
def make_cb2b():
    return CatBoostClassifier(iterations=150, random_seed=42, cat_features=cat_names_2b,
                               verbose=0, thread_count=-1, loss_function='MultiClass')
scores = manual_cv_mcc(make_cb2b, X2b_cb_df, y2b_5)
record("EXP-M9", "OBJ2b", scores.mean(), scores.std(), "CatBoost_multiclass")

# Optuna XGBoost multi-class
print("\nEXP-P4: Optuna XGBoost multi-class (30 trials)")
def obj2b_optuna(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'objective': 'multi:softprob', 'num_class': n_classes5,
        'random_state': 42, 'eval_metric': 'mlogloss', 'verbosity': 0,
    }
    model = XGBClassifier(**params)
    scores = cross_validate(model, X2b, y2b_5, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    return scores['test_mcc'].mean()

study2b = optuna.create_study(direction='maximize')
study2b.optimize(obj2b_optuna, n_trials=15)
record("EXP-P4", "OBJ2b", study2b.best_value, 0.0, "XGB_optuna_multiclass",
       f"best_trial={study2b.best_trial.number}")

# Binary: predict E.coli (code 14) specifically
print("\nEXP-D5: Binary E.coli (code 14) vs rest")
y_ecoli = (org_df['UrineNOOID'] == 14.0).astype(int).values
print(f"  E.coli: {y_ecoli.sum()}/{len(y_ecoli)} ({y_ecoli.mean():.3f})")
lgbm_ecoli = LGBMClassifier(n_estimators=300, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
cv_r = cross_validate(lgbm_ecoli, X2b, y_ecoli, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-D5a", "OBJ2b_ecoli", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "LightGBM_ecoli")

def make_cb_ecoli():
    return CatBoostClassifier(iterations=150, auto_class_weights='Balanced', random_seed=42,
                               cat_features=cat_names_2b, verbose=0, thread_count=-1)
scores = manual_cv_mcc(make_cb_ecoli, X2b_cb_df, y_ecoli)
record("EXP-D5b", "OBJ2b_ecoli", scores.mean(), scores.std(), "CatBoost_ecoli")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("ITERATION 2 COMPLETE -- ALL RESULTS")
print("=" * 70)

results_df = pd.DataFrame(all_results)
baselines = {'OBJ1': 0.8973, 'OBJ2a': 0.3900, 'OBJ2b': 0.1549, 'OBJ2b_ecoli': 0.0}

for obj in ['OBJ1', 'OBJ2a', 'OBJ2b', 'OBJ2b_ecoli']:
    obj_results = results_df[results_df['objective'] == obj].sort_values('mcc_mean', ascending=False)
    if len(obj_results) == 0:
        continue
    baseline = baselines.get(obj, 0)
    print(f"\n{obj} (baseline={baseline:.4f}):")
    for _, row in obj_results.iterrows():
        delta = row['mcc_mean'] - baseline
        print(f"  {row['exp_id']:10s} {row['model']:30s} MCC={row['mcc_mean']:.4f} ({delta:+.4f})  {row['notes']}")

results_df.to_csv(r'C:\Users\Ravi\research\data-analysis\output\amrx_research\experiment_results_iter2.csv', index=False)
print(f"\nSaved: output/amrx_research/experiment_results_iter2.csv")
