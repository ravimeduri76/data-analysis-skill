import sys, pandas as pd, numpy as np
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

ANALYSIS_DATE = pd.Timestamp("2026-03-26")
SENTINEL_DATE = pd.Timestamp("2099-12-31")

# ── Load source files ────────────────────────────────────────────────────────
mort_raw = pd.read_csv(r"C:\Users\Ravi\Elara_Probus\Probus_Data\mortality_status.csv",
                       low_memory=False)
ml = pd.read_csv(r"C:\Users\Ravi\Elara_Probus\Probus_Data\mortality_ml_ready.csv",
                 low_memory=False)

print(f"ML base   : {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"Source raw: {mort_raw.shape[0]:,} rows x {mort_raw.shape[1]} cols")

# ── 1. Impute age and care_duration_days with median ─────────────────────────
age_median = ml["age"].median()
dur_median  = ml["care_duration_days"].median()
ml["age"]               = ml["age"].fillna(age_median).round(1)
ml["care_duration_days"]= ml["care_duration_days"].fillna(dur_median).round(0).astype(int)
print(f"\n[1] Imputed age median={age_median:.1f}  care_duration median={dur_median:.0f} days")

# ── 2. Compute condition_period_days per diagnosis row, then per client ───────
# Parse dates; treat 2099-12-31 sentinel as no end date
mort_raw["diag_start"] = pd.to_datetime(mort_raw["diagnosis_start_date"],
                                         format="mixed", errors="coerce")
mort_raw["diag_end"]   = pd.to_datetime(mort_raw["diagnosis_end_date"],
                                         format="mixed", errors="coerce")
# Replace sentinel with NaT
mort_raw.loc[mort_raw["diag_end"] >= SENTINEL_DATE, "diag_end"] = pd.NaT

# condition_period = end - start; if no end → analysis_date - start
mort_raw["end_for_calc"] = mort_raw["diag_end"].fillna(ANALYSIS_DATE)
mort_raw["condition_period_days"] = (
    (mort_raw["end_for_calc"] - mort_raw["diag_start"]).dt.days
    .clip(lower=0)                  # no negative periods
)

# Per client: max condition period across all diagnoses
cond_period = (mort_raw.groupby("ClientId")["condition_period_days"]
               .max().reset_index(name="condition_period_days"))
null_before = cond_period["condition_period_days"].isna().sum()
cond_period_median = cond_period["condition_period_days"].median()
cond_period["condition_period_days"] = (cond_period["condition_period_days"]
                                         .fillna(cond_period_median).round(0).astype(int))
ml = ml.merge(cond_period, on="ClientId", how="left")
ml["condition_period_days"] = ml["condition_period_days"].fillna(cond_period_median).round(0).astype(int)
print(f"[2] condition_period_days added (max per client); {null_before} clients had no diagnosis date -> imputed with median {cond_period_median:.0f} days")
print(f"    Range: {ml['condition_period_days'].min()} to {ml['condition_period_days'].max()} days")

# ── 3. Merge paired Q_ columns ───────────────────────────────────────────────
Q_PAIRS = [
    ("Q_1641","Q_6339","q_hospital_admit"),
    ("Q_1642","Q_6340","q_urgent_care_er"),
    ("Q_1643","Q_6341","q_fall"),
    ("Q_1644","Q_6342","q_pain_swelling"),
    ("Q_1645","Q_6343","q_flu_symptoms"),
    ("Q_1646","Q_6344","q_skin_breakdown"),
    ("Q_1647","Q_6345","q_mood_changes"),
    ("Q_1648","Q_6346","q_missed_meds"),
    ("Q_1649","Q_6347","q_bathroom_issues"),
    ("Q_1650","Q_6348","q_breathing"),
    ("Q_1652","Q_6350","q_other_concerns"),
]
orig_q_cols = []
for v1, v2, name in Q_PAIRS:
    ml[name] = ml[v1] + ml[v2]
    orig_q_cols += [v1, v2]

# Recompute total q score from merged columns
merged_q_names = [name for _, _, name in Q_PAIRS]
ml["q_score_total"] = ml[merged_q_names].sum(axis=1)
# Drop original split columns and old v1/v2 subtotals
ml = ml.drop(columns=orig_q_cols + ["q_score_v1","q_score_v2"], errors="ignore")
print(f"[3] Merged {len(Q_PAIRS)} Q_ pairs into: {merged_q_names}")

# ── 4. One-hot encode MedicaidState ─────────────────────────────────────────
# Get one value per client from source (first non-null)
med_state = (mort_raw.groupby("ClientId")["MedicaidState"]
             .first().reset_index())
ml = ml.merge(med_state, on="ClientId", how="left")
ml["MedicaidState"] = ml["MedicaidState"].fillna("Unknown")
state_dummies = pd.get_dummies(ml["MedicaidState"], prefix="medicaid", dtype=int)
ml = pd.concat([ml, state_dummies], axis=1)
ml = ml.drop(columns=["MedicaidState"])
print(f"[4] MedicaidState OHE columns: {state_dummies.columns.tolist()}")

# ── 5. WeeklyAuthHrs → 4 categories ─────────────────────────────────────────
# Get one value per client from source
wah = (mort_raw.groupby("ClientId")["WeeklyAuthHrs"]
       .first().reset_index())
ml = ml.merge(wah, on="ClientId", how="left")

# Compute thresholds on non-null values
wah_vals = wah["WeeklyAuthHrs"].dropna()
q25 = wah_vals.quantile(0.25)
q90 = wah_vals.quantile(0.90)
print(f"[5] WeeklyAuthHrs thresholds: Low=0-{q25:.0f}  Medium={q25:.0f}-{q90:.0f}  High>{q90:.0f}")

def wah_category(v):
    if pd.isna(v):
        return "Unknown"
    if v <= q25:
        return "Low"
    if v <= q90:
        return "Medium"
    return "High"

ml["weekly_auth_hrs_cat"] = ml["WeeklyAuthHrs"].apply(wah_category)
wah_dummies = pd.get_dummies(ml["weekly_auth_hrs_cat"], prefix="wah", dtype=int)
ml = pd.concat([ml, wah_dummies], axis=1)
ml = ml.drop(columns=["WeeklyAuthHrs","weekly_auth_hrs_cat"])
print(f"    Category counts:\n{ml[wah_dummies.columns].sum().to_string()}")

# ── 6. Final checks ──────────────────────────────────────────────────────────
null_remaining = ml.isnull().sum()
null_remaining = null_remaining[null_remaining > 0]
print(f"\n=== FINAL ML DATASET ===")
print(f"Shape       : {ml.shape[0]:,} rows x {ml.shape[1]} columns")
print(f"Target dist : Mortality_Flag=1: {ml['Mortality_Flag'].sum():,}  ({ml['Mortality_Flag'].mean()*100:.1f}%)")
if len(null_remaining):
    print(f"Remaining nulls: {null_remaining.to_dict()}")
else:
    print("Remaining nulls: None — dataset is complete")

print(f"\nAll columns:")
for i, c in enumerate(ml.columns):
    print(f"  {i+1:>3}. {c}")

out_path = save_csv(ml, r"C:\Users\Ravi\Elara_Probus\Probus_Data\mortality_ml_ready.csv")
print(f"\nSaved: {out_path}")
