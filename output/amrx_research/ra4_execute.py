"""RA4: Lab Executor -- Execute top experiments for all 3 objectives."""
import sys, os, warnings, time, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\Ravi\research\data-analysis')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\Ravi\Data_Analysis\.env')
from loader import load_dataframe

from sklearn.model_selection import StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               StackingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.decomposition import PCA

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

mcc_scorer = make_scorer(matthews_corrcoef)

# ================================================================
# LOAD AND PREPARE DATA
# ================================================================
match_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\matching_data.csv')
total_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\Total_data.csv')
entero_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\enterobacteriaceae.csv')

target = 'Combined_Result_Culture_Result'
EXCLUDE = {'UrineCulture', 'UrineCulture_Yes_No', 'UrineNOOID', 'PatientUniqueID'}

obj1_cols = [c for c in match_df.select_dtypes(include=[np.number]).columns
             if c not in EXCLUDE and c != target]

# Identify ternary columns (1/2/3 encoding)
ternary_cols = []
for c in obj1_cols:
    uvals = set(match_df[c].dropna().unique())
    if uvals <= {1.0, 2.0, 3.0} and len(uvals) == 3:
        ternary_cols.append(c)

# Standard CV
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Results collector
all_results = []

def record(exp_id, obj, mcc_mean, mcc_std, model_name, notes=""):
    r = {"exp_id": exp_id, "objective": obj, "mcc_mean": round(mcc_mean, 4),
         "mcc_std": round(mcc_std, 4), "model": model_name, "notes": notes}
    all_results.append(r)
    print(f"  [{exp_id}] {model_name:25s} MCC={mcc_mean:.4f}+/-{mcc_std:.4f}  {notes}")

# ================================================================
# OBJ1 EXPERIMENTS
# ================================================================
print("=" * 70)
print("OBJ1: UTI PREDICTION EXPERIMENTS")
print("=" * 70)

X1_raw = match_df[obj1_cols].fillna(0).values
y1 = match_df[target].values

# --- EXP-D1: Recode 1/2/3 to 1/0/NaN ---
print("\nEXP-D1: Feature encoding recode (1/2/3 -> 1/0/NaN)")
df_recoded = match_df[obj1_cols].copy()
for c in ternary_cols:
    df_recoded[c] = df_recoded[c].map({1.0: 1, 2.0: 0, 3.0: np.nan})
# Impute NaN with mode
for c in df_recoded.columns:
    if df_recoded[c].isna().any():
        df_recoded[c].fillna(df_recoded[c].mode()[0] if len(df_recoded[c].mode()) > 0 else 0, inplace=True)
X1_recoded = df_recoded.values

for name, model in [
    ('LightGBM_recoded', LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)),
    ('GBM_recoded', GradientBoostingClassifier(n_estimators=200, random_state=42)),
]:
    cv_r = cross_validate(model, X1_recoded, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    record("EXP-D1", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), name)

# --- EXP-D2: Total_data.csv ---
print("\nEXP-D2: Dataset size surface (Total_data 7866 rows)")
total_cols = [c for c in obj1_cols if c in total_df.columns]
X1_total = total_df[total_cols].fillna(0).values
y1_total = total_df[target].values
for name, model in [
    ('LightGBM_total', LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)),
    ('GBM_total', GradientBoostingClassifier(n_estimators=200, random_state=42)),
]:
    cv_r = cross_validate(model, X1_total, y1_total, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    record("EXP-D2", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), name)

# --- EXP-F1: Symptom interactions ---
print("\nEXP-F1: Symptom interaction features")
symptoms = ['Dysuria', 'SuprapubicPain', 'Hematuria', 'FoulSmellingUrine', 'CloudyUrine',
            'HOFeverChills', 'HOGeneralizedWeaknessMalaise', 'HONauseaVomiting', 'HOFlankPain',
            'StorageSymptoms', 'VoidingSymptoms']
df_interact = match_df[obj1_cols].fillna(0).copy()
# Add top interaction pairs
pairs = [('CloudyUrine', 'FoulSmellingUrine'), ('StorageSymptoms', 'Dysuria'),
         ('HOFeverChills', 'CloudyUrine'), ('VoidingSymptoms', 'Hematuria'),
         ('Dysuria', 'HOFeverChills'), ('StorageSymptoms', 'HOFeverChills'),
         ('FoulSmellingUrine', 'HONauseaVomiting'), ('CloudyUrine', 'Dysuria'),
         ('StorageSymptoms', 'VoidingSymptoms'), ('HOFlankPain', 'HOFeverChills')]
for a, b in pairs:
    df_interact[f'{a}_x_{b}'] = df_interact[a] * df_interact[b]
X1_interact = df_interact.values
model = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
cv_r = cross_validate(model, X1_interact, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-F1", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "LightGBM_interactions")

# --- EXP-F2: Symptom burden ---
print("\nEXP-F2: Symptom burden score")
df_burden = match_df[obj1_cols].fillna(0).copy()
df_burden['symptom_burden'] = (match_df[symptoms] == 1).sum(axis=1)
comorbidity_cols = [c for c in ternary_cols if c not in symptoms]
df_burden['comorbidity_burden'] = (match_df[comorbidity_cols] == 1).sum(axis=1)
X1_burden = df_burden.values
model = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
cv_r = cross_validate(model, X1_burden, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-F2", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "LightGBM_burden")

# --- EXP-M1: CatBoost ---
if HAS_CATBOOST:
    print("\nEXP-M1: CatBoost with native categorical")
    cat_idx = [obj1_cols.index(c) for c in ternary_cols if c in obj1_cols]
    model = CatBoostClassifier(iterations=200, class_weights={0: 1, 1: 2}, random_seed=42,
                                cat_features=cat_idx, verbose=0)
    cv_r = cross_validate(model, X1_raw, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=1)
    record("EXP-M1", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "CatBoost_native")
else:
    print("\nEXP-M1: CatBoost not installed, skipping")

# --- EXP-P1: Optuna LightGBM ---
print("\nEXP-P1: Optuna LightGBM tuning")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        model = LGBMClassifier(**params)
        scores = cross_validate(model, X1_raw, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
        return scores['test_mcc'].mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    record("EXP-P1", "OBJ1", study.best_value, 0.0, "LightGBM_optuna",
           f"best_params: lr={study.best_params.get('learning_rate', '?'):.4f}")
except ImportError:
    print("  Optuna not installed, skipping")

# --- EXP-V1: Repeated CV ---
print("\nEXP-V1: 10-fold repeated CV (3 repeats)")
rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
model = LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
cv_r = cross_validate(model, X1_raw, y1, cv=rcv, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-V1", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "LightGBM_10fold_3rep",
       f"95%CI=[{np.percentile(cv_r['test_mcc'], 2.5):.4f}, {np.percentile(cv_r['test_mcc'], 97.5):.4f}]")

# ================================================================
# OBJ2a EXPERIMENTS
# ================================================================
print("\n" + "=" * 70)
print("OBJ2a: ENTEROBACTERIACEAE PREDICTION EXPERIMENTS")
print("=" * 70)

pos_df = match_df[match_df[target] == 1].copy()
entero_ids = set(entero_df['PatientUniqueID'])
pos_df['Enterobacteriaceae'] = pos_df['PatientUniqueID'].isin(entero_ids).astype(int)
obj2a_cols = [c for c in obj1_cols if c != 'BloodCulture']
X2a = pos_df[obj2a_cols].fillna(0).values
y2a = pos_df['Enterobacteriaceae'].values

# --- EXP-D3: SMOTE ---
print("\nEXP-D3: SMOTE rebalancing")
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    for name, base_model in [
        ('GBM_SMOTE', GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ('RF_SMOTE', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('LightGBM_SMOTE', LGBMClassifier(n_estimators=200, random_state=42, verbose=-1, n_jobs=-1)),
    ]:
        pipe = ImbPipeline([('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
                             ('clf', base_model)])
        cv_r = cross_validate(pipe, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
        record("EXP-D3", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), name)
except ImportError:
    print("  imblearn not installed. pip install imbalanced-learn")

# --- EXP-F3: Feature selection top-k ---
print("\nEXP-F3: Feature selection top-k")
rf_imp = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_imp.fit(X2a, y2a)
importances = pd.Series(rf_imp.feature_importances_, index=obj2a_cols).sort_values(ascending=False)
top20 = importances.head(20).index.tolist()
top30 = importances.head(30).index.tolist()
print(f"  Top-10 features: {importances.head(10).index.tolist()}")

for k, feat_list in [(20, top20), (30, top30)]:
    X2a_sel = pos_df[feat_list].fillna(0).values
    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    cv_r = cross_validate(model, X2a_sel, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
    record("EXP-F3", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
           f"GBM_top{k}", f"{k} features")

# --- EXP-F4: BloodCulture one-hot ---
print("\nEXP-F4: BloodCulture one-hot encoding")
bc_dummies = pd.get_dummies(pos_df['BloodCulture'], prefix='BC').astype(int)
df_f4 = pd.concat([pos_df[obj2a_cols].fillna(0).reset_index(drop=True),
                    bc_dummies.reset_index(drop=True)], axis=1)
X2a_bc = df_f4.values
model = GradientBoostingClassifier(n_estimators=200, random_state=42)
cv_r = cross_validate(model, X2a_bc, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-F4", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "GBM_BC_onehot")

# --- EXP-M2: SVM-RBF ---
print("\nEXP-M2: SVM-RBF")
pipe_svm = Pipeline([('scaler', StandardScaler()),
                      ('clf', SVC(kernel='rbf', class_weight='balanced', random_state=42))])
cv_r = cross_validate(pipe_svm, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-M2", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "SVM_RBF")

# --- EXP-M3: Stacking ---
print("\nEXP-M3: Stacking ensemble")
estimators = [
    ('lr', Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))])),
    ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
]
if HAS_XGB:
    scale_pos = (y2a == 0).sum() / max((y2a == 1).sum(), 1)
    estimators.append(('xgb', XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss', verbosity=0)))

stack = StackingClassifier(estimators=estimators,
                            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                            cv=3, n_jobs=-1)
cv_r = cross_validate(stack, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-M3", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "Stacking_LR+RF+XGB")

# --- EXP-P2: Optuna GBM ---
print("\nEXP-P2: Optuna GBM + threshold")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj2a_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        model = GradientBoostingClassifier(**params, random_state=42)
        scores = cross_validate(model, X2a, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
        return scores['test_mcc'].mean()

    study2a = optuna.create_study(direction='maximize')
    study2a.optimize(obj2a_objective, n_trials=50, show_progress_bar=False)
    record("EXP-P2", "OBJ2a", study2a.best_value, 0.0, "GBM_optuna",
           f"lr={study2a.best_params.get('learning_rate', '?'):.4f}")
except ImportError:
    print("  Optuna not installed")

# ================================================================
# OBJ2b EXPERIMENTS
# ================================================================
print("\n" + "=" * 70)
print("OBJ2b: ORGANISM IDENTIFICATION EXPERIMENTS")
print("=" * 70)

org_df = pos_df[pos_df['UrineNOOID'].notna()].copy()
org_counts = org_df['UrineNOOID'].value_counts()

# --- EXP-D4: Aggressive class grouping (top-5 + Other) ---
print("\nEXP-D4: Top-5 organisms + Other (6 classes)")
top5_codes = org_counts.head(5).index.tolist()  # 14, 15, 20, 12, 11
org_df['org_top5'] = org_df['UrineNOOID'].apply(lambda x: int(x) if x in top5_codes else 0)
le5 = LabelEncoder()
y2b_5 = le5.fit_transform(org_df['org_top5'].values)
X2b = org_df[obj2a_cols].fillna(0).values
n_classes5 = len(le5.classes_)
print(f"  Classes: {n_classes5}, distribution: {dict(zip(le5.classes_, np.bincount(y2b_5)))}")

for name, model in [
    ('LightGBM_top5', LGBMClassifier(n_estimators=300, class_weight='balanced', objective='multiclass',
                                       num_class=n_classes5, random_state=42, verbose=-1, n_jobs=-1)),
    ('GBM_top5', GradientBoostingClassifier(n_estimators=300, random_state=42)),
    ('RF_top5', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('ExtraTrees_top5', ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)),
]:
    cv_r = cross_validate(model, X2b, y2b_5, cv=cv5,
                          scoring={'mcc': mcc_scorer, 'f1': 'f1_weighted'}, n_jobs=-1)
    record("EXP-D4", "OBJ2b", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), name,
           f"F1w={cv_r['test_f1'].mean():.4f}")

# --- EXP-F5: PCA on urological conditions ---
print("\nEXP-F5: Urological PCA features")
uro_cols = [c for c in obj2a_cols if c in [
    'Urolithiasis','TumoursOfTheUrinaryTract','UretericStrictures','UPJO','urethralstricture',
    'NeurogenicBladder','RenalCysts','Posteriorurethralvalve','VesicoureteralReflux',
    'BladderDiverticula','Nephrocalcinosis','ProstaticHypertrophy','Diverticula',
    'PelvicalycealObstruction','CongenitalAbnormalities','IndwellingUrethralCatheter',
    'IntermittentCatheterization','UretericStent','NephrostomyTube','UrologicalProcedures',
    'IlealConduit','MedullarySpongeKidney','RenalFailure','RenalTransplant','SpinalAnomalies',
    'Cystocele','Gynaecologicalmalignancy']]
pca = PCA(n_components=5, random_state=42)
uro_pca = pca.fit_transform(org_df[uro_cols].fillna(0).values)
df_pca = pd.DataFrame(org_df[obj2a_cols].fillna(0).values, columns=obj2a_cols)
for i in range(5):
    df_pca[f'uro_pca_{i}'] = uro_pca[:, i]
X2b_pca = df_pca.values

model = LGBMClassifier(n_estimators=300, class_weight='balanced', objective='multiclass',
                         num_class=n_classes5, random_state=42, verbose=-1, n_jobs=-1)
cv_r = cross_validate(model, X2b_pca, y2b_5, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-F5", "OBJ2b", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "LightGBM_PCA")

# --- EXP-M4: Hierarchical classification ---
print("\nEXP-M4: Hierarchical classifier (2-stage)")
# Stage 1: Entero vs non-Entero (already have predictions)
# Stage 2: Within Entero, predict top organisms
# For CV we do a simplified version: predict org_top5 using a 2-stage approach
from sklearn.base import BaseEstimator, ClassifierMixin

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.stage1 = LGBMClassifier(n_estimators=200, class_weight='balanced',
                                      random_state=42, verbose=-1, n_jobs=-1)
        self.stage2_entero = LGBMClassifier(n_estimators=200, class_weight='balanced',
                                             random_state=42, verbose=-1, n_jobs=-1)
        self.stage2_other = None  # non-entero all map to "Other" (class 0)
        self.le_entero = LabelEncoder()
        self.entero_codes = set()

    def fit(self, X, y):
        # y is label-encoded org_top5: 0=Other, and entero codes are 14,15,20,12,11
        # Stage 1: is this Entero? (codes 11,12,14,15,20 vs Other=0)
        # In our encoding, class 0 = Other (includes non-Entero)
        # For simplicity, we use all entero top-5 codes
        y_binary = (y != 0).astype(int)  # 0=Other/non-entero, 1=entero species
        self.stage1.fit(X, y_binary)

        # Stage 2: within predicted Entero, which species?
        entero_mask = y != 0
        if entero_mask.sum() > 10:
            self.stage2_entero.fit(X[entero_mask], y[entero_mask])
        return self

    def predict(self, X):
        stage1_pred = self.stage1.predict(X)
        result = np.zeros(len(X), dtype=int)
        entero_mask = stage1_pred == 1
        if entero_mask.sum() > 0:
            result[entero_mask] = self.stage2_entero.predict(X[entero_mask])
        return result

hier = HierarchicalClassifier()
cv_r = cross_validate(hier, X2b, y2b_5, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=1)
record("EXP-M4", "OBJ2b", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(), "Hierarchical_2stage")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("EXPERIMENT EXECUTION SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(all_results)
for obj in ['OBJ1', 'OBJ2a', 'OBJ2b']:
    obj_results = results_df[results_df['objective'] == obj].sort_values('mcc_mean', ascending=False)
    baselines = {'OBJ1': 0.8973, 'OBJ2a': 0.3900, 'OBJ2b': 0.1549}
    thresholds = {'OBJ1': 0.92, 'OBJ2a': 0.55, 'OBJ2b': 0.30}
    baseline = baselines[obj]
    threshold = thresholds[obj]

    print(f"\n{obj} (baseline MCC={baseline:.4f}, threshold={threshold:.4f}):")
    print(f"{'Exp':10s} {'Model':30s} {'MCC':10s} {'Delta':8s} {'Notes'}")
    print("-" * 80)
    for _, row in obj_results.iterrows():
        delta = row['mcc_mean'] - baseline
        marker = " <<< NEW BEST" if delta > 0 else ""
        print(f"{row['exp_id']:10s} {row['model']:30s} {row['mcc_mean']:.4f}     {delta:+.4f}   {row.get('notes','')}{marker}")

    best_mcc = obj_results['mcc_mean'].max()
    if best_mcc >= threshold:
        print(f"  >>> THRESHOLD MET: {best_mcc:.4f} >= {threshold:.4f}")
    else:
        print(f"  --- Gap to threshold: {threshold - best_mcc:.4f}")

# Save results
results_df.to_csv(r'C:\Users\Ravi\research\data-analysis\output\amrx_research\experiment_results.csv', index=False)
print(f"\nResults saved to output/amrx_research/experiment_results.csv")
