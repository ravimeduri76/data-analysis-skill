"""
build_ml_v5.py
Adds 3 new feature groups from Client_Events.xlsx to mortality_ml_ready_v4.csv:

  A. Reason-code pivot (rc_*)
     One column per ClientStatusReasoncode value -> count of occurrences per ClientId.
     Codes with < 5 total occurrences across the whole dataset are grouped into rc_other.
     Column names: rc_<cleaned_label>  (text before " - " separator, lowercased, underscored)

  B. Time since admission
     time_since_admission_days = ANALYSIS_DATE - min(AdmissionDate) per client

  C. Weekday counts (7 columns each, Mon–Sun)
     eff_start_<DOW>  : count of EffectiveDateTZ   events falling on each weekday
     eff_end_<DOW>    : count of EffectiveDateToTZ  events falling on each weekday
     adm_<DOW>        : count of AdmissionDate      events falling on each weekday
"""
import sys, re, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

OUT           = r"C:\Users\Ravi\Elara_Probus\Probus_Data"
ANALYSIS_DATE = pd.Timestamp("2026-03-26")
RARE_THRESHOLD = 5           # codes with fewer total rows -> bucketed into rc_other
DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── 1. Load events ────────────────────────────────────────────────────────────
ev = pd.read_excel(f"{OUT}\\Client_Events.xlsx", engine="openpyxl")
ev.columns = ev.columns.str.strip()
ev.rename(columns={"ClientID":"ClientId"}, inplace=True)

ev["EffectiveDateTZ"]   = pd.to_datetime(ev["EffectiveDateTZ"],   errors="coerce")
ev["EffectiveDateToTZ"] = pd.to_datetime(ev["EffectiveDateToTZ"], errors="coerce")
ev["AdmissionDate"]     = pd.to_datetime(ev["AdmissionDate"],     errors="coerce")

print(f"[0] Events: {ev.shape[0]:,} rows | clients: {ev['ClientId'].nunique():,}")

# ── A. Reason-code pivot ──────────────────────────────────────────────────────
def clean_rc(label: str) -> str:
    """Take text before ' - ' separator, lower, replace non-alphanumeric with _, dedupe _."""
    if not isinstance(label, str):
        return ""
    part = label.split(" - ")[0].strip()
    part = re.sub(r"[^a-z0-9]+", "_", part.lower())
    part = re.sub(r"_+", "_", part).strip("_")
    return part[:45]   # cap length

ev["rc_clean"] = ev["ClientStatusReasoncode"].apply(clean_rc)

# Identify rare codes
rc_totals = ev["rc_clean"].value_counts()
rare_codes = set(rc_totals[rc_totals < RARE_THRESHOLD].index) | {""}   # empty = null rows
ev["rc_bucket"] = ev["rc_clean"].apply(
    lambda x: "rc_other" if x in rare_codes else f"rc_{x}"
)

print(f"\n[A] Reason code stats:")
print(f"    Unique raw codes (excl null): {ev['ClientStatusReasoncode'].nunique()}")
print(f"    After cleaning: {ev['rc_clean'].nunique()}")
print(f"    Rare (< {RARE_THRESHOLD} rows) -> rc_other: {len(rare_codes)-1}")
non_rare = sorted({f"rc_{x}" for x in rc_totals[rc_totals >= RARE_THRESHOLD].index if x})
print(f"    Non-rare buckets ({len(non_rare)}): {non_rare}")

# Pivot: count occurrences per client per bucket
# Drop rows where bucket is null-derived (rc_bucket == '' handled as rc_other already)
rc_pivot = (
    ev[ev["rc_bucket"] != ""]
    .groupby(["ClientId", "rc_bucket"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
rc_pivot.columns.name = None
# Ensure rc_other exists even if all rare codes sum to 0
if "rc_other" not in rc_pivot.columns:
    rc_pivot["rc_other"] = 0

# Drop columns that are entirely zero after pivot
rc_cols = [c for c in rc_pivot.columns if c.startswith("rc_")]
zero_cols = [c for c in rc_cols if rc_pivot[c].sum() == 0]
if zero_cols:
    rc_pivot.drop(columns=zero_cols, inplace=True)
    rc_cols = [c for c in rc_cols if c not in zero_cols]

print(f"    Final rc_ columns: {len(rc_cols)}")
print(f"    Sample pivot (top 5 rows, top 8 cols):")
print(rc_pivot[["ClientId"] + rc_cols[:8]].head(5).to_string())

# ── B. Time since admission ────────────────────────────────────────────────────
tsa = (ev.groupby("ClientId")["AdmissionDate"]
         .min()          # earliest admission date if multiple
         .reset_index())
tsa["time_since_admission_days"] = (
    (ANALYSIS_DATE - tsa["AdmissionDate"]).dt.days.clip(lower=0)
)
tsa = tsa[["ClientId","time_since_admission_days"]]
print(f"\n[B] time_since_admission_days: min={tsa['time_since_admission_days'].min()} "
      f"median={tsa['time_since_admission_days'].median():.0f} "
      f"max={tsa['time_since_admission_days'].max()}")

# ── C. Weekday counts ─────────────────────────────────────────────────────────
def weekday_counts(df_grp, date_col, prefix):
    """Return per-client weekday count columns (Mon-Sun)."""
    df_grp = df_grp.copy()
    df_grp["_dow"] = df_grp[date_col].dt.day_name()
    pivot = (
        df_grp.dropna(subset=[date_col])
        .groupby(["ClientId","_dow"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    # Ensure all 7 days present
    for day in DOW_ORDER:
        if day not in pivot.columns:
            pivot[day] = 0
    # Rename to prefix_DOW (first 3 chars)
    rename = {d: f"{prefix}_{d[:3]}" for d in DOW_ORDER}
    pivot = pivot.rename(columns=rename)
    return pivot[["ClientId"] + [f"{prefix}_{d[:3]}" for d in DOW_ORDER]]

dow_start = weekday_counts(ev, "EffectiveDateTZ",   "eff_start")
dow_end   = weekday_counts(ev, "EffectiveDateToTZ", "eff_end")
dow_adm   = weekday_counts(ev, "AdmissionDate",     "adm")

print(f"\n[C] Weekday count columns: {dow_start.shape[1]-1 + dow_end.shape[1]-1 + dow_adm.shape[1]-1} total")
print(f"    eff_start_ (EffectiveDateTZ)  : {dow_start.shape[0]:,} clients")
print(f"    eff_end_   (EffectiveDateToTZ): {dow_end.shape[0]:,} clients")
print(f"    adm_       (AdmissionDate)    : {dow_adm.shape[0]:,} clients")

# ── D. Merge all new features ─────────────────────────────────────────────────
all_new = (tsa
           .merge(rc_pivot, on="ClientId", how="outer")
           .merge(dow_start, on="ClientId", how="outer")
           .merge(dow_end,   on="ClientId", how="outer")
           .merge(dow_adm,   on="ClientId", how="outer"))
all_new.fillna(0, inplace=True)
# time_since_admission_days: keep as int
all_new["time_since_admission_days"] = all_new["time_since_admission_days"].astype(int)

print(f"\n[D] New feature table: {all_new.shape[0]:,} clients x {all_new.shape[1]} cols")

# ── E. Join to v4 ML dataset ──────────────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready_v4.csv", low_memory=False)
ml["gender_enc"] = ml["gender_enc"].fillna(ml["gender_enc"].median())
print(f"[E] ML v4 base: {ml.shape[0]:,} rows x {ml.shape[1]} cols")

ml = ml.merge(all_new, on="ClientId", how="left")

# Clients not in events -> fill counts with 0
new_feature_cols = [c for c in all_new.columns if c != "ClientId"]
ml[new_feature_cols] = ml[new_feature_cols].fillna(0)
ml["time_since_admission_days"] = ml["time_since_admission_days"].astype(int)

print(f"[E] After join: {ml.shape[0]:,} rows x {ml.shape[1]} cols (+{ml.shape[1]-128} vs v4)")

# ── F. Drop constant/duplicate columns ───────────────────────────────────────
feat_cols = [c for c in ml.columns if c not in ["ClientId","Mortality_Flag"]]
const_cols = [c for c in feat_cols if ml[c].nunique() <= 1]
if const_cols:
    print(f"[F] Dropping {len(const_cols)} constant columns: {const_cols}")
    ml.drop(columns=const_cols, inplace=True)

# ── G. Correlation with target ────────────────────────────────────────────────
print(f"\n=== FINAL ML V5 DATASET ===")
print(f"Shape  : {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"Target : Mortality_Flag=1: {ml['Mortality_Flag'].sum():,} ({ml['Mortality_Flag'].mean()*100:.1f}%)")
null_check = ml.isnull().sum(); null_check = null_check[null_check > 0]
if len(null_check):
    print(f"Nulls  : {null_check.to_dict()}")
else:
    print("Nulls  : None")

new_cols = [c for c in new_feature_cols if c in ml.columns]
corr = (ml[new_cols + ["Mortality_Flag"]]
        .corr()["Mortality_Flag"]
        .drop("Mortality_Flag")
        .sort_values(key=abs, ascending=False))
print(f"\nTop 30 correlations of new features with Mortality_Flag:")
print(corr.head(30).round(4).to_string())

# ── H. Save ───────────────────────────────────────────────────────────────────
out_path = save_csv(ml, f"{OUT}\\mortality_ml_ready_v5.csv")
print(f"\nSaved: {out_path}")
print("=== DONE ===")
