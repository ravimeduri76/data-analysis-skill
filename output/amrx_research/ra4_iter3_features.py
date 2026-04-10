"""RA4: Iteration 3 -- Deep Feature Engineering.

Focus: domain-informed interactions, non-linear transforms, clinical composites.
Addresses the gap identified in iterations 1-2 where feature engineering was shallow.
"""
import sys, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\Ravi\research\data-analysis')
from dotenv import load_dotenv
load_dotenv(r'C:\Users\Ravi\Data_Analysis\.env')
from loader import load_dataframe, save_csv

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import matthews_corrcoef, make_scorer
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

mcc_scorer = make_scorer(matthews_corrcoef)
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load data
match_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\matching_data.csv')
entero_df = load_dataframe(r'C:\Users\Ravi\Downloads\AMRx\datasets\enterobacteriaceae.csv')
target = 'Combined_Result_Culture_Result'
EXCLUDE = {'UrineCulture', 'UrineCulture_Yes_No', 'UrineNOOID', 'PatientUniqueID'}
obj1_cols = [c for c in match_df.select_dtypes(include=[np.number]).columns
             if c not in EXCLUDE and c != target]

# Identify feature groups
symptoms = ['Dysuria', 'SuprapubicPain', 'Hematuria', 'FoulSmellingUrine', 'CloudyUrine',
            'HOFeverChills', 'HOGeneralizedWeaknessMalaise', 'HONauseaVomiting', 'HOFlankPain',
            'StorageSymptoms', 'VoidingSymptoms']
histories = ['HOPreviousUTI', 'HOCatheterization', 'HOPreviousAntibiotics',
             'HORecurrentUTI', 'HOUrinaryIncontinence']
vitals = ['Age', 'Temperature', 'PR', 'RR', 'Systolic_BP', 'Diastolic_BP']

# Recode ternary to binary for interactions (1=yes, 2=no, 3=unknown -> 1/0/0)
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
    print(f"  [{exp_id}] {model_name:35s} MCC={mcc_mean:.4f}+/-{mcc_std:.4f}  {notes}")

y1 = match_df[target].values

# Helper: build base feature DataFrame with binary-recoded ternary cols
def build_base_df(source_df, cols):
    df = source_df[cols].fillna(0).copy()
    for c in ternary_cols:
        if c in df.columns:
            df[c] = (df[c] == 1.0).astype(int)  # 1=yes -> 1, 2=no/3=unknown -> 0
    return df

print("=" * 70)
print("ITERATION 3: DEEP FEATURE ENGINEERING")
print("=" * 70)

# ================================================================
# BLOCK A: Non-linear transforms on continuous variables
# ================================================================
print("\n--- BLOCK A: Non-linear transforms on continuous variables ---")

base_df = build_base_df(match_df, obj1_cols)

# EXP-FE1: Age exponential scaling (risk accelerates after 50)
print("\nEXP-FE1: Age exponential scaling (1.1^(age-50) for age>50)")
df_a = base_df.copy()
df_a['Age_exp50'] = np.where(df_a['Age'] > 50, 1.1 ** (df_a['Age'] - 50), 1.0)
df_a['Age_sq'] = df_a['Age'] ** 2 / 10000  # quadratic, scaled
df_a['Age_over50'] = (df_a['Age'] > 50).astype(int)
df_a['Age_over65'] = (df_a['Age'] > 65).astype(int)
cv_r = cross_validate(LGBMClassifier(n_estimators=200, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_a.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE1", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_age_nonlinear", "Age_exp50 + Age_sq + Age_over50/65")

# EXP-FE2: Temperature deviation from normal (fever magnitude)
print("\nEXP-FE2: Temperature deviation and clinical thresholds")
df_b = base_df.copy()
df_b['Temp_deviation'] = (df_b['Temperature'] - 37.0).abs()  # distance from normal
df_b['Temp_fever'] = (df_b['Temperature'] > 38.0).astype(int)  # clinical fever
df_b['Temp_high_fever'] = (df_b['Temperature'] > 38.5).astype(int)
df_b['Temp_hypothermia'] = (df_b['Temperature'] < 36.0).astype(int)
cv_r = cross_validate(LGBMClassifier(n_estimators=200, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_b.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE2", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_temp_features", "Temp_deviation + fever flags")

# EXP-FE3: All vitals clinical thresholds (SIRS-like)
print("\nEXP-FE3: Clinical vital sign thresholds (SIRS-inspired)")
df_c = base_df.copy()
df_c['Temp_abnormal'] = ((df_c['Temperature'] > 38.0) | (df_c['Temperature'] < 36.0)).astype(int)
df_c['HR_tachy'] = (df_c['PR'] > 90).astype(int)
df_c['RR_tachy'] = (df_c['RR'] > 20).astype(int)
df_c['BP_hypo'] = (df_c['Systolic_BP'] < 90).astype(int)
df_c['SIRS_count'] = df_c['Temp_abnormal'] + df_c['HR_tachy'] + df_c['RR_tachy']
df_c['Sepsis_risk'] = (df_c['SIRS_count'] >= 2).astype(int)
cv_r = cross_validate(LGBMClassifier(n_estimators=200, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_c.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE3", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_SIRS", "SIRS_count + Sepsis_risk + vital thresholds")

# ================================================================
# BLOCK B: Domain-informed cross-group interactions
# ================================================================
print("\n--- BLOCK B: Domain-informed cross-group interactions ---")

# EXP-FE4: Symptom x History interactions (the key missing piece)
print("\nEXP-FE4: Symptom x History interactions")
df_d = base_df.copy()
# Key clinical interaction: active symptoms with relevant history
cross_pairs = [
    ('Dysuria', 'HOPreviousUTI'),           # active dysuria + UTI history
    ('Dysuria', 'HORecurrentUTI'),           # active dysuria + recurrent UTI
    ('HOFeverChills', 'HOCatheterization'),   # fever + catheter = high risk
    ('CloudyUrine', 'HOPreviousUTI'),        # cloudy urine + UTI history
    ('FoulSmellingUrine', 'HOPreviousUTI'),  # foul urine + UTI history
    ('StorageSymptoms', 'HORecurrentUTI'),   # storage symptoms + recurrent
    ('HOFlankPain', 'HOFeverChills'),        # flank pain + fever = pyelonephritis risk
    ('Hematuria', 'HOCatheterization'),      # blood + catheter
    ('VoidingSymptoms', 'HOUrinaryIncontinence'),  # voiding + incontinence
    ('HONauseaVomiting', 'HOFeverChills'),   # systemic symptoms co-occurrence
]
for a, b in cross_pairs:
    if a in df_d.columns and b in df_d.columns:
        df_d[f'{a}_x_{b}'] = df_d[a] * df_d[b]  # both binary now, so 1*1=1

cv_r = cross_validate(LGBMClassifier(n_estimators=200, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_d.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE4", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_symptom_x_history", "10 cross-group binary interactions")

# EXP-FE5: Symptom x Vitals interactions (age-modulated symptoms)
print("\nEXP-FE5: Symptom x Vitals interactions")
df_e = base_df.copy()
df_e['Age_over50'] = (df_e['Age'] > 50).astype(int)
df_e['Age_over65'] = (df_e['Age'] > 65).astype(int)
df_e['Temp_fever'] = (df_e['Temperature'] > 38.0).astype(int)

symptom_vital_pairs = [
    ('Dysuria', 'Age_over50'),             # older + dysuria
    ('HOPreviousUTI', 'Age_over65'),       # elderly + UTI history
    ('HOCatheterization', 'Age_over65'),   # elderly + catheter
    ('HOFeverChills', 'Temp_fever'),       # subjective + objective fever agreement
    ('CloudyUrine', 'Temp_fever'),         # urine signs + fever
    ('Dysuria', 'Temp_fever'),             # dysuria + fever = complicated UTI
]
for a, b in symptom_vital_pairs:
    if a in df_e.columns and b in df_e.columns:
        df_e[f'{a}_x_{b}'] = df_e[a] * df_e[b]

cv_r = cross_validate(LGBMClassifier(n_estimators=200, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_e.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE5", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_symptom_x_vitals", "6 symptom x age/temp interactions")

# ================================================================
# BLOCK C: Clinical composite risk scores
# ================================================================
print("\n--- BLOCK C: Clinical composite risk scores ---")

# EXP-FE6: Weighted symptom index (using SHAP-like importances)
print("\nEXP-FE6: Weighted symptom index + recurrent UTI risk")
df_f = base_df.copy()

# Symptom weights from LightGBM importance (top features from SHAP analysis)
symptom_weights = {
    'HOFeverChills': 333, 'Dysuria': 314, 'StorageSymptoms': 267,
    'CloudyUrine': 227, 'FoulSmellingUrine': 170, 'HOFlankPain': 144,
    'Hematuria': 100, 'SuprapubicPain': 178, 'VoidingSymptoms': 80,
    'HOGeneralizedWeaknessMalaise': 90, 'HONauseaVomiting': 70,
}
# Normalize weights to sum to 1
total_w = sum(symptom_weights.values())
for k in symptom_weights:
    symptom_weights[k] /= total_w

df_f['weighted_symptom_score'] = sum(
    df_f[s] * w for s, w in symptom_weights.items() if s in df_f.columns
)

# Recurrent UTI risk index
history_weights = {
    'HOPreviousUTI': 0.35, 'HORecurrentUTI': 0.30,
    'HOCatheterization': 0.20, 'HOPreviousAntibiotics': 0.10,
    'HOUrinaryIncontinence': 0.05,
}
df_f['uti_history_score'] = sum(
    df_f[h] * w for h, w in history_weights.items() if h in df_f.columns
)

# Combined risk = symptom score * (1 + history score) * age factor
df_f['Age_factor'] = np.where(df_f['Age'] > 50, 1.1 ** ((df_f['Age'] - 50) / 10), 1.0)
df_f['combined_risk'] = df_f['weighted_symptom_score'] * (1 + df_f['uti_history_score']) * df_f['Age_factor']

cv_r = cross_validate(LGBMClassifier(n_estimators=200, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_f.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE6", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_composite_risk", "weighted_symptom + uti_history + age_factor")

# ================================================================
# BLOCK D: KITCHEN SINK -- All transforms combined
# ================================================================
print("\n--- BLOCK D: All feature engineering combined ---")

# EXP-FE7: Combine ALL new features from blocks A-C
print("\nEXP-FE7: Kitchen sink (all new features)")
df_all = base_df.copy()

# Block A: Age transforms
df_all['Age_exp50'] = np.where(df_all['Age'] > 50, 1.1 ** (df_all['Age'] - 50), 1.0)
df_all['Age_sq'] = df_all['Age'] ** 2 / 10000
df_all['Age_over50'] = (df_all['Age'] > 50).astype(int)
df_all['Age_over65'] = (df_all['Age'] > 65).astype(int)

# Block A: Temp transforms
df_all['Temp_deviation'] = (df_all['Temperature'] - 37.0).abs()
df_all['Temp_fever'] = (df_all['Temperature'] > 38.0).astype(int)
df_all['Temp_high_fever'] = (df_all['Temperature'] > 38.5).astype(int)

# Block A: SIRS
df_all['Temp_abnormal'] = ((df_all['Temperature'] > 38.0) | (df_all['Temperature'] < 36.0)).astype(int)
df_all['HR_tachy'] = (df_all['PR'] > 90).astype(int)
df_all['RR_tachy'] = (df_all['RR'] > 20).astype(int)
df_all['BP_hypo'] = (df_all['Systolic_BP'] < 90).astype(int)
df_all['SIRS_count'] = df_all['Temp_abnormal'] + df_all['HR_tachy'] + df_all['RR_tachy']

# Block B: Symptom x History
for a, b in cross_pairs:
    if a in df_all.columns and b in df_all.columns:
        df_all[f'{a}_x_{b}'] = df_all[a] * df_all[b]

# Block B: Symptom x Vitals
for a, b in [('Dysuria', 'Age_over50'), ('HOPreviousUTI', 'Age_over65'),
             ('HOCatheterization', 'Age_over65'), ('HOFeverChills', 'Temp_fever'),
             ('CloudyUrine', 'Temp_fever'), ('Dysuria', 'Temp_fever')]:
    if a in df_all.columns and b in df_all.columns:
        df_all[f'{a}_x_{b}'] = df_all[a] * df_all[b]

# Block C: Composites
df_all['weighted_symptom_score'] = sum(
    df_all[s] * w for s, w in symptom_weights.items() if s in df_all.columns
)
df_all['uti_history_score'] = sum(
    df_all[h] * w for h, w in history_weights.items() if h in df_all.columns
)
df_all['Age_factor'] = np.where(df_all['Age'] > 50, 1.1 ** ((df_all['Age'] - 50) / 10), 1.0)
df_all['combined_risk'] = df_all['weighted_symptom_score'] * (1 + df_all['uti_history_score']) * df_all['Age_factor']

print(f"  Total features: {df_all.shape[1]} (base={len(obj1_cols)}, new={df_all.shape[1]-len(obj1_cols)})")

# Test with LightGBM
cv_r = cross_validate(LGBMClassifier(n_estimators=300, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_all.values, y1, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE7a", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_kitchen_sink", f"{df_all.shape[1]} features total")

# Test with GBM
cv_r = cross_validate(GradientBoostingClassifier(n_estimators=200, random_state=42),
    df_all.values, y1, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE7b", "OBJ1", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "GBM_kitchen_sink", f"{df_all.shape[1]} features total")

# ================================================================
# BLOCK E: Same feature engineering for OBJ2a (Enterobacteriaceae)
# ================================================================
print("\n--- BLOCK E: Feature engineering for OBJ2a ---")

pos_df = match_df[match_df[target] == 1].copy()
entero_ids = set(entero_df['PatientUniqueID'])
pos_df['Enterobacteriaceae'] = pos_df['PatientUniqueID'].isin(entero_ids).astype(int)
obj2a_cols = [c for c in obj1_cols if c != 'BloodCulture']
y2a = pos_df['Enterobacteriaceae'].values

# Build enriched feature set for OBJ2a
df_2a = build_base_df(pos_df, obj2a_cols)

# Add all transforms (same as kitchen sink but for OBJ2a subset)
df_2a['Age_exp50'] = np.where(df_2a['Age'] > 50, 1.1 ** (df_2a['Age'] - 50), 1.0)
df_2a['Age_over50'] = (df_2a['Age'] > 50).astype(int)
df_2a['Age_over65'] = (df_2a['Age'] > 65).astype(int)
df_2a['Temp_deviation'] = (df_2a['Temperature'] - 37.0).abs()
df_2a['Temp_fever'] = (df_2a['Temperature'] > 38.0).astype(int)
df_2a['Temp_abnormal'] = ((df_2a['Temperature'] > 38.0) | (df_2a['Temperature'] < 36.0)).astype(int)
df_2a['HR_tachy'] = (df_2a['PR'] > 90).astype(int)
df_2a['RR_tachy'] = (df_2a['RR'] > 20).astype(int)
df_2a['SIRS_count'] = df_2a['Temp_abnormal'] + df_2a['HR_tachy'] + df_2a['RR_tachy']

for a, b in cross_pairs:
    if a in df_2a.columns and b in df_2a.columns:
        df_2a[f'{a}_x_{b}'] = df_2a[a] * df_2a[b]

df_2a['weighted_symptom_score'] = sum(
    df_2a[s] * w for s, w in symptom_weights.items() if s in df_2a.columns
)
df_2a['uti_history_score'] = sum(
    df_2a[h] * w for h, w in history_weights.items() if h in df_2a.columns
)
df_2a['Age_factor'] = np.where(df_2a['Age'] > 50, 1.1 ** ((df_2a['Age'] - 50) / 10), 1.0)
df_2a['combined_risk'] = df_2a['weighted_symptom_score'] * (1 + df_2a['uti_history_score']) * df_2a['Age_factor']

print(f"  OBJ2a features: {df_2a.shape[1]} (base={len(obj2a_cols)}, new={df_2a.shape[1]-len(obj2a_cols)})")

# EXP-FE8: Enriched features for Enterobacteriaceae
cv_r = cross_validate(GradientBoostingClassifier(n_estimators=200, random_state=42),
    df_2a.values, y2a, cv=cv5, scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE8a", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "GBM_enriched_features", f"{df_2a.shape[1]} features total")

cv_r = cross_validate(LGBMClassifier(n_estimators=300, class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1), df_2a.values, y2a, cv=cv5,
    scoring={'mcc': mcc_scorer}, n_jobs=-1)
record("EXP-FE8b", "OBJ2a", cv_r['test_mcc'].mean(), cv_r['test_mcc'].std(),
       "LightGBM_enriched_features", f"{df_2a.shape[1]} features total")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("ITERATION 3 (FEATURE ENGINEERING) COMPLETE")
print("=" * 70)

results_df = pd.DataFrame(all_results)
baselines = {'OBJ1': 0.9025, 'OBJ2a': 0.4139}  # best from iter1+2

for obj in ['OBJ1', 'OBJ2a']:
    obj_results = results_df[results_df['objective'] == obj].sort_values('mcc_mean', ascending=False)
    if len(obj_results) == 0:
        continue
    baseline = baselines.get(obj, 0)
    print(f"\n{obj} (previous best={baseline:.4f}):")
    for _, row in obj_results.iterrows():
        delta = row['mcc_mean'] - baseline
        marker = " <<<" if delta > 0.005 else ""
        print(f"  {row['exp_id']:10s} {row['model']:35s} MCC={row['mcc_mean']:.4f} ({delta:+.4f}){marker}  {row['notes']}")

save_csv(results_df, os.path.join(r'C:\Users\Ravi\research\data-analysis\output\amrx_research',
    'experiment_results_iter3_features.csv'))
print(f"\nSaved: experiment_results_iter3_features.csv")
