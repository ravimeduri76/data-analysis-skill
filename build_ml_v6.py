"""
build_ml_v6.py
Rebuilds ALL Client_Events.xlsx-derived features with per-client cutoff dates
to eliminate temporal leakage in the reason-code pivot and event counts.

Cutoff logic
------------
  Mortality_Flag = 1  -> cutoff = EffectiveDateTZ of first discharged+Expired event
                         Events STRICTLY BEFORE (<) this date are used.
                         Fallback: last 'active' event (inclusive) if no expired row found.

  Mortality_Flag = 0  -> cutoff = EffectiveDateTZ of last 'active' event (inclusive <=).
                         If no active event exists: all events for that client are used.

Features rebuilt on pre-cutoff events only
------------------------------------------
  A. V4 event counts/days:
       count_pending, count_active, count_discharges, count_discharge_hosp_snf,
       sum_pending_days, sum_active_days, sum_onhold_days, sum_onhold_hosp_snf_days,
       sum_waitlist_days
  B. days_active_before_expiry  (looks back from the expired event — uses full events
       for Mortality=1 clients so the look-back row is still available)
  C. Ratio features: onhold_ratio, hosp_hold_ratio, hosp_discharge_rate
  D. Reason-code pivot (rc_*)
  E. time_since_admission_days  (min AdmissionDate from pre-cutoff events)
  F. Weekday counts (eff_start_*, eff_end_*, adm_*)

Base  : mortality_ml_ready_v3.csv  (CCI + interactions, no event features)
Output: mortality_ml_ready_v6.csv
"""
import sys, re, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

OUT            = r"C:\Users\Ravi\Elara_Probus\Probus_Data"
ANALYSIS_DATE  = pd.Timestamp("2026-03-26")
RARE_THRESHOLD = 5
DOW_ORDER      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── 1. Load events ─────────────────────────────────────────────────────────────
ev = pd.read_excel(f"{OUT}\\Client_Events.xlsx", engine="openpyxl")
ev.columns = ev.columns.str.strip()
ev.rename(columns={"ClientID": "ClientId"}, inplace=True)
ev["status"] = ev["CLIENTSTATUS"].str.lower().str.strip()
ev["reason"] = ev["ClientStatusReasoncode"].fillna("").str.lower().str.strip()
ev["EffectiveDateTZ"]   = pd.to_datetime(ev["EffectiveDateTZ"],   errors="coerce")
ev["EffectiveDateToTZ"] = pd.to_datetime(ev["EffectiveDateToTZ"], errors="coerce")
ev["AdmissionDate"]     = pd.to_datetime(ev["AdmissionDate"],     errors="coerce")
print(f"[0] Events loaded: {ev.shape[0]:,} rows | {ev['ClientId'].nunique():,} clients")

# ── 2. Load ML base (v3) to get per-client Mortality_Flag ─────────────────────
ml_base = pd.read_csv(f"{OUT}\\mortality_ml_ready_v3.csv", low_memory=False)
ml_base["gender_enc"] = ml_base["gender_enc"].fillna(ml_base["gender_enc"].median())
print(f"[1] ML v3 base: {ml_base.shape[0]:,} rows x {ml_base.shape[1]} cols")
print(f"    Mortality_Flag=1: {ml_base['Mortality_Flag'].sum():,} "
      f"({ml_base['Mortality_Flag'].mean()*100:.1f}%)")

flag_map = ml_base.set_index("ClientId")["Mortality_Flag"]

# ── 3. Determine per-client cutoff dates ──────────────────────────────────────
# Expired discharge events
m_exp = ev["reason"].str.contains("expired", na=False) & (ev["status"] == "discharged")
exp_cutoff = ev[m_exp].groupby("ClientId")["EffectiveDateTZ"].min()   # first expired event

# Last active event per client
active_cutoff = (ev[ev["status"] == "active"]
                 .groupby("ClientId")["EffectiveDateTZ"].max())        # last active event

# Mortality=1 clients
mort1 = ml_base.loc[ml_base["Mortality_Flag"] == 1, "ClientId"]
m1_exp    = exp_cutoff.reindex(mort1)           # expired date if exists
m1_act    = active_cutoff.reindex(mort1)        # fallback: last active
mort1_cutoff  = m1_exp.fillna(m1_act)           # prefer expired, fall back to active
mort1_strict  = m1_exp.notna()                  # strict=True when expired event found

# Mortality=0 clients
mort0 = ml_base.loc[ml_base["Mortality_Flag"] == 0, "ClientId"]
mort0_cutoff  = active_cutoff.reindex(mort0)    # last active (may be NaT = no active event)
mort0_strict  = pd.Series(False, index=mort0)   # always inclusive

cutoff_df = pd.DataFrame({
    "cutoff": pd.concat([mort1_cutoff, mort0_cutoff]),
    "strict": pd.concat([mort1_strict, mort0_strict]),
})
cutoff_df.index.name = "ClientId"
cutoff_df = cutoff_df.reset_index()

print(f"\n[2] Cutoff summary:")
print(f"    Mort=1 clients with expired event cutoff : "
      f"{mort1_strict.sum():,} / {len(mort1):,}")
print(f"    Mort=1 clients fallback to last active   : "
      f"{(~mort1_strict & mort1_cutoff.notna()).sum():,}")
print(f"    Mort=1 clients with NO cutoff (all events): "
      f"{mort1_cutoff.isna().sum():,}")
print(f"    Mort=0 clients with active-event cutoff  : "
      f"{mort0_cutoff.notna().sum():,} / {len(mort0):,}")
print(f"    Mort=0 clients with NO cutoff (all events): "
      f"{mort0_cutoff.isna().sum():,}")

# ── 4. Filter events to pre-cutoff window ─────────────────────────────────────
ev2 = ev.merge(cutoff_df, on="ClientId", how="left")
no_cutoff   =  ev2["cutoff"].isna()
strict_mask =  ev2["strict"].fillna(False)

keep = (
    no_cutoff
    | (strict_mask  & (ev2["EffectiveDateTZ"] <  ev2["cutoff"]))
    | (~strict_mask & (ev2["EffectiveDateTZ"] <= ev2["cutoff"]))
)
ev_f = ev2[keep].copy()

print(f"\n[3] Pre-cutoff events: {len(ev_f):,} / {len(ev):,} "
      f"({len(ev_f)/len(ev)*100:.1f}% retained)")
print(f"    Clients with at least 1 pre-cutoff event: {ev_f['ClientId'].nunique():,}")

# ── A. V4 event counts/days ───────────────────────────────────────────────────
ev_f["end_dt"]        = ev_f["EffectiveDateToTZ"].fillna(ANALYSIS_DATE)
ev_f["days_in_status"] = (ev_f["end_dt"] - ev_f["EffectiveDateTZ"]).dt.days.clip(lower=0)

m_pending    = ev_f["status"] == "pending"
m_active     = ev_f["status"] == "active"
m_discharged = ev_f["status"] == "discharged"
m_onhold     = ev_f["status"] == "on_hold"
m_waitlist   = ev_f["status"] == "waiting_list"
m_hosp_snf   = ev_f["reason"].str.contains("hospital|snf|nursing home", regex=True, na=False)

def sum_days(mask):
    return ev_f[mask].groupby("ClientId")["days_in_status"].sum()

def count_rows(mask):
    return ev_f[mask].groupby("ClientId").size()

agg = pd.DataFrame({
    "count_pending":            count_rows(m_pending),
    "count_active":             count_rows(m_active),
    "count_discharges":         count_rows(m_discharged),
    "count_discharge_hosp_snf": count_rows(m_discharged & m_hosp_snf),
    "sum_pending_days":         sum_days(m_pending),
    "sum_active_days":          sum_days(m_active),
    "sum_onhold_days":          sum_days(m_onhold),
    "sum_onhold_hosp_snf_days": sum_days(m_onhold & m_hosp_snf),
    "sum_waitlist_days":        sum_days(m_waitlist),
}).fillna(0).astype(int).reset_index()

print(f"\n[A] V4 event counts/days: {agg.shape}")
for c in agg.columns[1:]:
    print(f"    {c:<30} mean={agg[c].mean():.1f}  max={agg[c].max()}")

# ── B. days_active_before_expiry ──────────────────────────────────────────────
# Use FULL (unfiltered) events sorted by seqr; for Mort=1 clients the expired
# discharge row is still present so we can look back one row to the active episode.
ev_sorted = ev.sort_values(["ClientId", "seqr"]).reset_index(drop=True)
m_exp_full = (ev_sorted["reason"].str.contains("expired", na=False)
              & (ev_sorted["status"] == "discharged"))
ev_sorted["end_dt_full"]         = ev_sorted["EffectiveDateToTZ"].fillna(ANALYSIS_DATE)
ev_sorted["days_in_status_full"] = (ev_sorted["end_dt_full"] - ev_sorted["EffectiveDateTZ"]).dt.days.clip(lower=0)

expired_idx = ev_sorted.index[m_exp_full]
days_before = {}
for idx in expired_idx:
    cid = ev_sorted.at[idx, "ClientId"]
    if idx == 0:
        continue
    prev_idx = idx - 1
    if ev_sorted.at[prev_idx, "ClientId"] != cid:
        continue
    if ev_sorted.at[prev_idx, "status"] == "active":
        days_before[cid] = int(ev_sorted.at[prev_idx, "days_in_status_full"])

dabe = (pd.Series(days_before, name="days_active_before_expiry")
          .reset_index()
          .rename(columns={"index": "ClientId"}))
print(f"\n[B] days_active_before_expiry: {len(dabe):,} clients  "
      f"mean={dabe['days_active_before_expiry'].mean():.1f}  "
      f"max={dabe['days_active_before_expiry'].max()}")

# ── C. Ratio features ─────────────────────────────────────────────────────────
# (computed after join to ML — done at merge step below)

# ── D. Reason-code pivot ──────────────────────────────────────────────────────
def clean_rc(label: str) -> str:
    if not isinstance(label, str):
        return ""
    part = label.split(" - ")[0].strip()
    part = re.sub(r"[^a-z0-9]+", "_", part.lower())
    part = re.sub(r"_+", "_", part).strip("_")
    return part[:45]

ev_f["rc_clean"] = ev_f["ClientStatusReasoncode"].apply(clean_rc)
rc_totals = ev_f["rc_clean"].value_counts()
rare_codes = set(rc_totals[rc_totals < RARE_THRESHOLD].index) | {""}
ev_f["rc_bucket"] = ev_f["rc_clean"].apply(
    lambda x: "rc_other" if x in rare_codes else f"rc_{x}"
)

rc_pivot = (
    ev_f[ev_f["rc_bucket"] != ""]
    .groupby(["ClientId", "rc_bucket"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
rc_pivot.columns.name = None
if "rc_other" not in rc_pivot.columns:
    rc_pivot["rc_other"] = 0

rc_cols = [c for c in rc_pivot.columns if c.startswith("rc_")]
zero_cols = [c for c in rc_cols if rc_pivot[c].sum() == 0]
if zero_cols:
    rc_pivot.drop(columns=zero_cols, inplace=True)
    rc_cols = [c for c in rc_cols if c not in zero_cols]

non_rare = sorted({f"rc_{x}" for x in rc_totals[rc_totals >= RARE_THRESHOLD].index if x})
print(f"\n[D] Reason code pivot (pre-cutoff):")
print(f"    Non-rare buckets retained: {len(non_rare)}")
print(f"    Final rc_ columns: {len(rc_cols)}")

# Show if rc_client_expired is gone (it should be — expired event excluded)
has_exp = "rc_client_expired" in rc_cols
print(f"    rc_client_expired present: {has_exp}  "
      f"(should be False for Mort=1 clients by design)")
if has_exp:
    print(f"    rc_client_expired total count: {rc_pivot['rc_client_expired'].sum()}")

# Correlation of rc_client_expired with Mortality_Flag if it survived
merged_check = ml_base[["ClientId","Mortality_Flag"]].merge(rc_pivot, on="ClientId", how="left")
if "rc_client_expired" in merged_check.columns:
    corr_val = merged_check["rc_client_expired"].fillna(0).corr(merged_check["Mortality_Flag"])
    print(f"    rc_client_expired corr with Mortality_Flag: {corr_val:.4f}  "
          f"(was 0.9967 before cutoff fix)")

# ── E. Time since admission ───────────────────────────────────────────────────
tsa = (ev_f.groupby("ClientId")["AdmissionDate"]
           .min()
           .reset_index())
tsa["time_since_admission_days"] = (
    (ANALYSIS_DATE - tsa["AdmissionDate"]).dt.days.clip(lower=0)
)
tsa = tsa[["ClientId", "time_since_admission_days"]]
print(f"\n[E] time_since_admission_days: "
      f"min={tsa['time_since_admission_days'].min()}  "
      f"median={tsa['time_since_admission_days'].median():.0f}  "
      f"max={tsa['time_since_admission_days'].max()}")

# ── F. Weekday counts ─────────────────────────────────────────────────────────
def weekday_counts(df_grp, date_col, prefix):
    df_grp = df_grp.copy()
    df_grp["_dow"] = df_grp[date_col].dt.day_name()
    pivot = (
        df_grp.dropna(subset=[date_col])
        .groupby(["ClientId", "_dow"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    for day in DOW_ORDER:
        if day not in pivot.columns:
            pivot[day] = 0
    rename = {d: f"{prefix}_{d[:3]}" for d in DOW_ORDER}
    pivot = pivot.rename(columns=rename)
    return pivot[["ClientId"] + [f"{prefix}_{d[:3]}" for d in DOW_ORDER]]

dow_start = weekday_counts(ev_f, "EffectiveDateTZ",   "eff_start")
dow_end   = weekday_counts(ev_f, "EffectiveDateToTZ", "eff_end")
dow_adm   = weekday_counts(ev_f, "AdmissionDate",     "adm")
print(f"\n[F] Weekday count columns: "
      f"{dow_start.shape[1]-1 + dow_end.shape[1]-1 + dow_adm.shape[1]-1} total")

# ── G. Merge all new event features ──────────────────────────────────────────
all_new = (tsa
           .merge(agg,       on="ClientId", how="outer")
           .merge(dabe,      on="ClientId", how="outer")
           .merge(rc_pivot,  on="ClientId", how="outer")
           .merge(dow_start, on="ClientId", how="outer")
           .merge(dow_end,   on="ClientId", how="outer")
           .merge(dow_adm,   on="ClientId", how="outer"))

# Fill count/sum columns with 0; leave days_active_before_expiry as NaN
count_sum_cols = [c for c in all_new.columns
                  if c not in ("ClientId", "days_active_before_expiry")]
all_new[count_sum_cols] = all_new[count_sum_cols].fillna(0)
all_new["time_since_admission_days"] = all_new["time_since_admission_days"].astype(int)

# Ratio features (need to be computed after fillna)
safe_active  = all_new["sum_active_days"].clip(lower=1)
all_new["onhold_ratio"]        = (all_new["sum_onhold_days"] / safe_active).round(4)
all_new["hosp_hold_ratio"]     = (all_new["sum_onhold_hosp_snf_days"] /
                                   all_new["sum_onhold_days"].clip(lower=1)).round(4)
all_new["hosp_discharge_rate"] = (all_new["count_discharge_hosp_snf"] /
                                   all_new["count_discharges"].clip(lower=1)).round(4)

print(f"\n[G] New event feature table: {all_new.shape[0]:,} clients x {all_new.shape[1]} cols")

# ── H. Join to v3 ML base ─────────────────────────────────────────────────────
ml = ml_base.merge(all_new, on="ClientId", how="left")

new_feature_cols = [c for c in all_new.columns if c != "ClientId"]
fill_cols = [c for c in new_feature_cols if c != "days_active_before_expiry"]
ml[fill_cols] = ml[fill_cols].fillna(0)
ml["time_since_admission_days"] = ml["time_since_admission_days"].astype(int)

print(f"\n[H] After join: {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"    days_active_before_expiry non-null: "
      f"{ml['days_active_before_expiry'].notna().sum():,} / {len(ml):,}")

# ── I. Leakage check ──────────────────────────────────────────────────────────
print(f"\n=== LEAKAGE CHECK ===")
print(f"Shape  : {ml.shape[0]:,} rows x {ml.shape[1]} cols")
print(f"Target : Mortality_Flag=1: {ml['Mortality_Flag'].sum():,} "
      f"({ml['Mortality_Flag'].mean()*100:.1f}%)")

new_cols_in = [c for c in new_feature_cols if c in ml.columns]
corr = (ml[new_cols_in + ["Mortality_Flag"]]
        .corr()["Mortality_Flag"]
        .drop("Mortality_Flag")
        .sort_values(key=abs, ascending=False))
print(f"\nTop 30 correlations of new event features with Mortality_Flag:")
print(corr.head(30).round(4).to_string())

# Null check
null_check = ml.isnull().sum()
null_check = null_check[null_check > 0]
if len(null_check):
    print(f"\nNulls: {null_check.to_dict()}")
else:
    print("\nNulls: None (except days_active_before_expiry — intentionally sparse)")

# ── J. Save ───────────────────────────────────────────────────────────────────
out_path = save_csv(ml, f"{OUT}\\mortality_ml_ready_v6.csv")
print(f"\nSaved: {out_path}")
print("=== DONE ===")
