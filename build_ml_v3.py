"""
build_ml_v3.py
Builds mortality_ml_ready_v3.csv from mortality_ml_ready.csv by adding:
  1. CCI score + 5 binary comorbidity flags (from CCI_Elixhauser_Output.csv, cols D-K)
  2. ICD chapter interaction terms (clinically meaningful pairwise products)
  3. Questionnaire interaction features (symptom co-occurrence signals)
  4. CCI x Q composite features (comorbidity burden x adverse event signal)
"""
import sys, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

OUT = r"C:\Users\Ravi\Elara_Probus\Probus_Data"

# ── 1. Load base ML dataset ───────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready.csv", low_memory=False)
ml["gender_enc"] = ml["gender_enc"].fillna(ml["gender_enc"].median())
print(f"[0] Base ML dataset: {ml.shape[0]:,} rows x {ml.shape[1]} cols")

# ── 2. Build CCI per-client table from CCI_Elixhauser_Output.csv ─────────────
cci_raw = pd.read_csv(f"{OUT}\\CCI_Elixhauser_Output.csv", low_memory=False)
cci_raw.rename(columns={"clientid": "ClientId"}, inplace=True)

# Charlson_Index is row-level (one row per diagnosis) and stored as string
# "UNKNOWN" -> 0, then take max per client
cci_raw["cci_numeric"] = pd.to_numeric(cci_raw["Charlson_Index"], errors="coerce").fillna(0)

# Binary comorbidity flags cols D-K: CHF, Cardiac_Arrhythmia, Pulmonary, Diabetes, Renal, Liver, Cancer
binary_cols = ["CHF", "Cardiac_Arrhythmia", "Pulmonary", "Diabetes", "Renal", "Liver", "Cancer"]

cci_per_client = (
    cci_raw.groupby("ClientId")
    .agg(
        cci_score=("cci_numeric", "max"),
        **{c: (c, "max") for c in binary_cols}
    )
    .reset_index()
)

# Drop columns that are entirely 0 (no predictive value)
non_zero = [c for c in binary_cols if cci_per_client[c].sum() > 0]
zero_cols = [c for c in binary_cols if c not in non_zero]
if zero_cols:
    print(f"[2] Dropping all-zero comorbidity flags: {zero_cols}")
    cci_per_client.drop(columns=zero_cols, inplace=True)
    binary_cols = non_zero

print(f"[2] CCI per-client: {cci_per_client.shape[0]:,} clients")
print(f"    cci_score distribution: {cci_per_client['cci_score'].value_counts().sort_index().to_dict()}")
for c in binary_cols:
    print(f"    {c}: {cci_per_client[c].sum():,} clients ({cci_per_client[c].mean()*100:.1f}%)")

# ── 3. Left join CCI to ML dataset ───────────────────────────────────────────
ml = ml.merge(cci_per_client, on="ClientId", how="left")
# Clients without CCI data → fill with 0 (no documented comorbidity score)
cci_new_cols = ["cci_score"] + binary_cols
ml[cci_new_cols] = ml[cci_new_cols].fillna(0)
ml["cci_score"] = ml["cci_score"].astype(int)
for c in binary_cols:
    ml[c] = ml[c].astype(int)

n_matched = (ml["cci_score"] > 0).sum() + ml[binary_cols].any(axis=1).sum()
print(f"[3] After CCI join: {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"    Clients with any CCI data: {ml[binary_cols + ['cci_score']].any(axis=1).sum():,} / {len(ml):,}")

# ── 4. ICD chapter interaction terms ─────────────────────────────────────────
# Use clinically meaningful pairwise products for the most prevalent chapters
# Focus on chapters with ≥1% prevalence: M,I,G,R,J,E,F,S,N,C
# Clinical pairs chosen for mortality relevance:
ICD_PAIRS = [
    ("icd_ch_I", "icd_ch_J"),   # Circulatory + Respiratory (cardiorespiratory)
    ("icd_ch_I", "icd_ch_G"),   # Circulatory + Neurological (stroke/CVD + neuro)
    ("icd_ch_I", "icd_ch_E"),   # Circulatory + Metabolic (cardiovascular + diabetes)
    ("icd_ch_I", "icd_ch_M"),   # Circulatory + Musculoskeletal (mobility + heart)
    ("icd_ch_I", "icd_ch_N"),   # Circulatory + Renal (cardiorenal syndrome)
    ("icd_ch_G", "icd_ch_M"),   # Neurological + Musculoskeletal (falls risk)
    ("icd_ch_G", "icd_ch_R"),   # Neurological + Symptoms/Signs (neuro deterioration)
    ("icd_ch_J", "icd_ch_R"),   # Respiratory + Symptoms (dyspnea, respiratory symptoms)
    ("icd_ch_E", "icd_ch_N"),   # Metabolic + Renal (diabetic nephropathy)
    ("icd_ch_M", "icd_ch_R"),   # Musculoskeletal + Symptoms (pain + symptoms)
    ("icd_ch_F", "icd_ch_G"),   # Mental + Neurological (dementia/psychiatric + neuro)
]

icd_interaction_cols = []
for a, b in ICD_PAIRS:
    if a in ml.columns and b in ml.columns:
        col_name = f"ix_{a[7:]}_{b[7:]}"   # e.g. ix_I_J
        ml[col_name] = (ml[a] * ml[b]).astype(int)
        icd_interaction_cols.append(col_name)

print(f"\n[4] ICD chapter interactions ({len(icd_interaction_cols)}):")
for c in icd_interaction_cols:
    n = ml[c].sum()
    print(f"    {c}: {n:,} clients ({n/len(ml)*100:.2f}%)")

# ── 5. Questionnaire interaction features ────────────────────────────────────
# Symptom co-occurrence signals: binarise first (event occurred ≥ 1 time), then multiply
# This captures co-occurrence of adverse event types — stronger mortality signal than totals

# Binary occurrence flags (1 if any event of this type was reported)
Q_EVENT_COLS = [
    "q_hospital_admit", "q_urgent_care_er", "q_fall", "q_pain_swelling",
    "q_flu_symptoms", "q_skin_breakdown", "q_mood_changes", "q_missed_meds",
    "q_bathroom_issues", "q_breathing", "q_other_concerns"
]
for c in Q_EVENT_COLS:
    ml[f"b_{c}"] = (ml[c] > 0).astype(int)

# Clinically meaningful co-occurrence pairs
Q_PAIRS = [
    ("b_q_fall",          "b_q_urgent_care_er",  "qx_fall_er"),        # fall → ER visit
    ("b_q_breathing",     "b_q_urgent_care_er",  "qx_breathing_er"),   # dyspnea → ER
    ("b_q_breathing",     "b_q_hospital_admit",  "qx_breathing_hosp"), # dyspnea → hospitalised
    ("b_q_fall",          "b_q_hospital_admit",  "qx_fall_hosp"),      # fall → hospitalised
    ("b_q_pain_swelling", "b_q_breathing",        "qx_pain_breathing"), # multi-system distress
    ("b_q_mood_changes",  "b_q_missed_meds",      "qx_mood_meds"),      # cognitive decline + non-adherence
    ("b_q_skin_breakdown","b_q_missed_meds",      "qx_wound_meds"),     # wound + non-adherence
    ("b_q_bathroom_issues","b_q_mood_changes",    "qx_adl_mood"),       # ADL deterioration combo
    ("b_q_breathing",     "b_q_missed_meds",      "qx_breathing_meds"), # respiratory + non-adherence
    ("b_q_fall",          "b_q_pain_swelling",    "qx_fall_pain"),      # fall injury combo
]

q_interaction_cols = []
for a, b, name in Q_PAIRS:
    ml[name] = (ml[a] * ml[b]).astype(int)
    q_interaction_cols.append(name)

print(f"\n[5] Questionnaire co-occurrence interactions ({len(q_interaction_cols)}):")
for c in q_interaction_cols:
    n = ml[c].sum()
    pct_mort = ml.loc[ml[c]==1, "Mortality_Flag"].mean()*100 if n > 0 else 0
    print(f"    {c}: {n:,} clients | mortality% when=1: {pct_mort:.1f}%")

# Drop the intermediate binary cols (already have raw counts + interactions)
b_cols = [f"b_{c}" for c in Q_EVENT_COLS]
ml.drop(columns=b_cols, inplace=True)

# ── 6. CCI x Q composite features ────────────────────────────────────────────
# cci_score × q_score_total: high comorbidity burden combined with high event rate
ml["cci_x_q_total"]    = (ml["cci_score"] * ml["q_score_total"]).astype(float)
ml["cci_x_breathing"]  = (ml["cci_score"] * ml["q_breathing"]).astype(float)
ml["cci_x_hosp"]       = (ml["cci_score"] * ml["q_hospital_admit"]).astype(float)
# CHF × breathing (congestive heart failure + respiratory events)
ml["chf_x_breathing"]  = (ml["CHF"] * ml["q_breathing"]).astype(int)
# Diabetes × renal (diabetic complications)
if "Renal" in ml.columns:
    ml["diabetes_x_renal"] = (ml["Diabetes"] * ml["Renal"]).astype(int)
    composite_cols = ["cci_x_q_total", "cci_x_breathing", "cci_x_hosp", "chf_x_breathing", "diabetes_x_renal"]
else:
    composite_cols = ["cci_x_q_total", "cci_x_breathing", "cci_x_hosp", "chf_x_breathing"]

print(f"\n[6] CCI x Q composite features ({len(composite_cols)}):")
for c in composite_cols:
    n_nonzero = (ml[c] > 0).sum()
    print(f"    {c}: {n_nonzero:,} clients with non-zero ({n_nonzero/len(ml)*100:.1f}%)")

# ── 7. Final checks ───────────────────────────────────────────────────────────
null_remaining = ml.isnull().sum()
null_remaining = null_remaining[null_remaining > 0]
new_col_count = ml.shape[1]

print(f"\n=== FINAL ML V3 DATASET ===")
print(f"Shape       : {ml.shape[0]:,} rows x {new_col_count} columns")
print(f"Target dist : Mortality_Flag=1: {ml['Mortality_Flag'].sum():,}  ({ml['Mortality_Flag'].mean()*100:.1f}%)")
print(f"New columns : {new_col_count - 83} added vs v2 base (83 cols)")
if len(null_remaining):
    print(f"Remaining nulls: {null_remaining.to_dict()}")
else:
    print("Remaining nulls: None")

# Drop constant columns (all zeros — no predictive value)
n_before = ml.shape[1]
feature_cols = [c for c in ml.columns if c not in ["ClientId", "Mortality_Flag"]]
constant_cols = [c for c in feature_cols if ml[c].nunique() <= 1]
if constant_cols:
    print(f"\nDropping {len(constant_cols)} constant columns: {constant_cols}")
    ml.drop(columns=constant_cols, inplace=True)
print(f"Final shape: {ml.shape[0]:,} rows x {ml.shape[1]} cols")

print("\nNew feature summary:")
all_new = icd_interaction_cols + q_interaction_cols + composite_cols + cci_new_cols
for c in all_new:
    if c in ml.columns:
        print(f"  {c}")

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_path = save_csv(ml, f"{OUT}\\mortality_ml_ready_v3.csv")
print(f"\nSaved: {out_path}")
print("=== DONE ===")
