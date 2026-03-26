"""
build_ml_v4.py
Adds 10 client-event derived features from Client_Events.xlsx to mortality_ml_ready_v3.csv:
  1.  count_pending              - number of pending status rows
  2.  count_active               - number of active status rows
  3.  count_discharges           - total discharge events
  4.  count_discharge_hosp_snf   - discharges to hospital / SNF / nursing home
  5.  sum_pending_days           - total days spent in pending status
  6.  sum_active_days            - total days spent in active status
  7.  sum_onhold_days            - total days spent on hold
  8.  sum_onhold_hosp_snf_days   - on-hold days attributed to hospital / SNF
  9.  sum_waitlist_days          - total days on waiting list
  10. days_active_before_expiry  - duration of the active episode immediately
                                   before a discharged/Expired event
                                   (null if client never expired or prior event
                                    was not active)
"""
import sys, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import save_csv

OUT           = r"C:\Users\Ravi\Elara_Probus\Probus_Data"
ANALYSIS_DATE = pd.Timestamp("2026-03-26")

# ── 1. Load events ────────────────────────────────────────────────────────────
ev = pd.read_excel(f"{OUT}\\Client_Events.xlsx", engine="openpyxl")
ev.columns = ev.columns.str.strip()
ev.rename(columns={"ClientID": "ClientId"}, inplace=True)

print(f"[0] Events: {ev.shape[0]:,} rows x {ev.shape[1]} cols | "
      f"unique clients: {ev['ClientId'].nunique():,}")

# Normalise
ev["status"] = ev["CLIENTSTATUS"].str.lower().str.strip()
ev["reason"] = ev["ClientStatusReasoncode"].fillna("").str.lower().str.strip()

# ── 2. Compute days_in_status for every row ───────────────────────────────────
# Date Difference is pre-computed and verified accurate (100% match).
# For open rows (EffectiveDateToTZ null), fill to ANALYSIS_DATE.
ev["end_dt"]        = ev["EffectiveDateToTZ"].fillna(ANALYSIS_DATE)
ev["days_in_status"] = (ev["end_dt"] - ev["EffectiveDateTZ"]).dt.days.clip(lower=0)

# ── 3. Helper masks ───────────────────────────────────────────────────────────
m_pending    = ev["status"] == "pending"
m_active     = ev["status"] == "active"
m_discharged = ev["status"] == "discharged"
m_onhold     = ev["status"] == "on_hold"
m_waitlist   = ev["status"] == "waiting_list"
m_hosp_snf   = ev["reason"].str.contains("hospital|snf|nursing home", regex=True, na=False)
m_expired    = ev["reason"].str.contains("expired", na=False)

# ── 4. Simple group aggregations ─────────────────────────────────────────────
def sum_days(mask):
    return ev[mask].groupby("ClientId")["days_in_status"].sum()

def count_rows(mask):
    return ev[mask].groupby("ClientId").size()

agg = pd.DataFrame({
    "count_pending":           count_rows(m_pending),
    "count_active":            count_rows(m_active),
    "count_discharges":        count_rows(m_discharged),
    "count_discharge_hosp_snf":count_rows(m_discharged & m_hosp_snf),
    "sum_pending_days":        sum_days(m_pending),
    "sum_active_days":         sum_days(m_active),
    "sum_onhold_days":         sum_days(m_onhold),
    "sum_onhold_hosp_snf_days":sum_days(m_onhold & m_hosp_snf),
    "sum_waitlist_days":       sum_days(m_waitlist),
}).fillna(0).astype(int).reset_index()

print(f"[3] Aggregated event features: {agg.shape}")
for c in agg.columns[1:]:
    print(f"    {c:<30} mean={agg[c].mean():.1f}  max={agg[c].max()}")

# ── 5. days_active_before_expiry ──────────────────────────────────────────────
# Algorithm:
#   Sort each client's events by seqr (chronological order).
#   For each discharged+Expired row, look at the immediately preceding row
#   (seqr rank - 1 within the same client).
#   If that prior row has status = 'active', use its days_in_status.
#   Otherwise → NaN.

ev_sorted = ev.sort_values(["ClientId", "seqr"]).reset_index(drop=True)

# Identify expired discharge rows
expired_idx = ev_sorted.index[m_discharged & m_expired]

days_before = {}
for idx in expired_idx:
    cid = ev_sorted.at[idx, "ClientId"]
    if idx == 0:
        continue
    prev_idx = idx - 1
    # Make sure previous row belongs to the same client
    if ev_sorted.at[prev_idx, "ClientId"] != cid:
        continue
    if ev_sorted.at[prev_idx, "status"] == "active":
        days = int(ev_sorted.at[prev_idx, "days_in_status"])
        # If multiple expired events per client, take the last one (overwrite)
        days_before[cid] = days

dabe = pd.Series(days_before, name="days_active_before_expiry")\
         .reset_index().rename(columns={"index":"ClientId"})

matched = len(dabe)
print(f"\n[5] days_active_before_expiry computed for {matched:,} clients")
print(f"    Distribution: {dabe['days_active_before_expiry'].describe().round(1).to_string()}")

# ── 6. Merge all event features ───────────────────────────────────────────────
event_features = agg.merge(dabe, on="ClientId", how="left")
# days_active_before_expiry: null means client never expired or prior row wasn't active
# Leave as float NaN for ML (tree models handle NaN natively, pipelines will impute)
print(f"\n[6] Event feature table: {event_features.shape}")

# ── 7. Load v3 ML dataset and join ───────────────────────────────────────────
ml = pd.read_csv(f"{OUT}\\mortality_ml_ready_v3.csv", low_memory=False)
ml["gender_enc"] = ml["gender_enc"].fillna(ml["gender_enc"].median())
print(f"[7] ML v3 base: {ml.shape[0]:,} rows x {ml.shape[1]} cols")

ml = ml.merge(event_features, on="ClientId", how="left")

# Clients not in events file → fill count/sum cols with 0
count_sum_cols = [c for c in event_features.columns if c != "ClientId"
                  and c != "days_active_before_expiry"]
ml[count_sum_cols] = ml[count_sum_cols].fillna(0).astype(int)
# days_active_before_expiry: leave NaN as-is (tree-safe missing)
n_dabe_filled = ml["days_active_before_expiry"].notna().sum()
print(f"    days_active_before_expiry filled: {n_dabe_filled:,} / {len(ml):,}")

# ── 8. Derived ratio features ─────────────────────────────────────────────────
safe_active = ml["sum_active_days"].clip(lower=1)
# Proportion of active days spent on hold (instability ratio)
ml["onhold_ratio"]       = (ml["sum_onhold_days"] / safe_active).round(4)
# Proportion of holds attributed to hospital/SNF (acute hospitalisation burden)
ml["hosp_hold_ratio"]    = (ml["sum_onhold_hosp_snf_days"] / (ml["sum_onhold_days"].clip(lower=1))).round(4)
# Discharge-to-hospital rate per active day
ml["hosp_discharge_rate"]= (ml["count_discharge_hosp_snf"] / (ml["count_discharges"].clip(lower=1))).round(4)

print(f"[8] Ratio features added: onhold_ratio, hosp_hold_ratio, hosp_discharge_rate")

# ── 9. Final checks ───────────────────────────────────────────────────────────
null_check = ml.isnull().sum()
null_check = null_check[null_check > 0]
print(f"\n=== FINAL ML V4 DATASET ===")
print(f"Shape  : {ml.shape[0]:,} rows x {ml.shape[1]} cols  (+{ml.shape[1]-115} vs v3)")
print(f"Target : Mortality_Flag=1: {ml['Mortality_Flag'].sum():,}  ({ml['Mortality_Flag'].mean()*100:.1f}%)")
if len(null_check):
    print(f"Nulls  : {null_check.to_dict()}")
else:
    print("Nulls  : None (except days_active_before_expiry which is intentionally sparse)")

# Correlation of new features with target
new_cols = [c for c in event_features.columns if c != "ClientId"] + \
           ["onhold_ratio","hosp_hold_ratio","hosp_discharge_rate"]
corr_target = ml[new_cols + ["Mortality_Flag"]].corr()["Mortality_Flag"]\
                .drop("Mortality_Flag").sort_values(key=abs, ascending=False)
print(f"\nCorrelation of new event features with Mortality_Flag:")
print(corr_target.round(4).to_string())

# ── 10. Save ──────────────────────────────────────────────────────────────────
out_path = save_csv(ml, f"{OUT}\\mortality_ml_ready_v4.csv")
print(f"\nSaved: {out_path}")
print("=== DONE ===")
