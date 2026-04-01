import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import pyxlsb
import os

# ── Load raw xlsb ──────────────────────────────────────────────────────────
xlsb_path = (
    r"C:\Users\Ravi\AppData\Roaming\Microsoft\Excel"
    r"\Invoice%20line%20item%20and%20Adjustment%20Details312445534221293868"
    r"\Invoice%20line%20item%20and%20Adjustment%20Details((Unsaved-312446012912242721)).xlsb"
)

rows = []
with pyxlsb.open_workbook(xlsb_path) as wb:
    with wb.get_sheet(1) as sheet:
        for row in sheet.rows():
            rows.append([item.v for item in row])

df = pd.DataFrame(rows[1:], columns=rows[0])
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Clean & type-cast ─────────────────────────────────────────────────────
def xl_serial_to_date(s):
    """Convert Excel date serial (days since 1899-12-30) to datetime."""
    return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")

df["ServiceDate_dt"]     = xl_serial_to_date(pd.to_numeric(df["ServiceDate"],             errors="coerce"))
df["PostDate_dt"]        = xl_serial_to_date(pd.to_numeric(df["InvoiceLineItemPostDate"],  errors="coerce"))

df["CalcAmt"]  = pd.to_numeric(df["InvoiceLineItemCalulatedAmount"].replace("NULL", np.nan), errors="coerce").fillna(0)
df["AdjAmt"]   = pd.to_numeric(df["InvoiceAdjustmentAmount"].replace("NULL", np.nan),       errors="coerce").fillna(0)

# Use ServiceDate as the primary date dimension
df["date"] = df["ServiceDate_dt"]

print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── Derive quarter label ───────────────────────────────────────────────────
def quarter_label(dt):
    if pd.isna(dt):
        return None
    y, q = dt.year, (dt.month - 1) // 3 + 1
    starts = {1: "Jan-Mar", 2: "Apr-Jun", 3: "Jul-Sep", 4: "Oct-Dec"}
    return f"{starts[q]} {y}"

# Order key: (year, quarter_num)
def quarter_sort_key(dt):
    if pd.isna(dt):
        return (9999, 9)
    return (dt.year, (dt.month - 1) // 3 + 1)

df["quarter"]          = df["date"].apply(quarter_label)
df["quarter_sort_key"] = df["date"].apply(quarter_sort_key)

# ── Flag rows that carry an adjustment ────────────────────────────────────
df["has_adj"]        = (df["InvoiceAdjustmentType"].replace("NULL", np.nan).notna()).astype(int)
df["abs_adj"]        = df["AdjAmt"].abs()

# ── Quarterly aggregation ─────────────────────────────────────────────────
grp = (
    df.dropna(subset=["quarter"])
      .groupby(["quarter", "quarter_sort_key"], as_index=False)
      .agg(
          total_line_items    = ("CalcAmt",  "count"),
          total_billed        = ("CalcAmt",  "sum"),
          adj_line_items      = ("has_adj",  "sum"),
          total_adj_abs       = ("abs_adj",  "sum"),
          total_adj_signed    = ("AdjAmt",   "sum"),
      )
)

grp["adj_line_item_pct"] = grp["adj_line_items"] / grp["total_line_items"] * 100
grp["adj_rate_pct"]      = grp["total_adj_abs"]  / grp["total_billed"].replace(0, np.nan) * 100
grp = grp.sort_values("quarter_sort_key").reset_index(drop=True)

print("\n=== Quarterly Summary ===")
display_cols = ["quarter", "total_line_items", "total_billed", "adj_line_items",
                "adj_line_item_pct", "total_adj_abs", "total_adj_signed", "adj_rate_pct"]
print(grp[display_cols].to_string(index=False))

# ── Adjustment type breakdown by quarter ─────────────────────────────────
adj_rows = df[df["has_adj"] == 1].copy()
adj_type_qtr = (
    adj_rows.dropna(subset=["quarter"])
    .groupby(["quarter", "quarter_sort_key", "InvoiceAdjustmentType"], as_index=False)
    .agg(count=("AdjAmt", "count"), total_adj=("abs_adj", "sum"))
    .sort_values(["quarter_sort_key", "total_adj"], ascending=[True, False])
)

print("\n=== Top Adjustment Types by Quarter ===")
for q in grp["quarter"]:
    sub = adj_type_qtr[adj_type_qtr["quarter"] == q].head(5)
    print(f"\n{q}:")
    print(sub[["InvoiceAdjustmentType", "count", "total_adj"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# FIGURE — 3-panel quarterly chart
# ══════════════════════════════════════════════════════════════════════════
quarters = grp["quarter"].tolist()
x = np.arange(len(quarters))
bar_w = 0.55

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig.patch.set_facecolor("#FAFAFA")
for ax in axes:
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

BLUE  = "#1565C0"
RED   = "#C62828"
AMBER = "#F57F17"

# ── Panel 1: Total Billed vs Total Absolute Adjustment ($ bars, stacked concept) ──
ax1 = axes[0]
b1 = ax1.bar(x, grp["total_billed"],    width=bar_w, label="Total Billed ($)",        color=BLUE, alpha=0.85, zorder=2)
b2 = ax1.bar(x, grp["total_adj_abs"],   width=bar_w * 0.45, label="Total Adj |$|", color=RED, alpha=0.9, zorder=3)
ax1.set_title("Quarterly Billing vs Gross Adjustment Amount", fontsize=11, fontweight="bold", pad=8)
ax1.set_ylabel("Amount ($)", fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.legend(fontsize=9, loc="upper left", framealpha=0.85)
# Value labels on billed bars
for rect in b1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2, h + 1000, f"${h:,.0f}",
             ha="center", va="bottom", fontsize=7.5, color=BLUE, fontweight="bold")

# ── Panel 2: Adjustment Rate % of Billed ─────────────────────────────────
ax2 = axes[1]
bar_colors = [RED if v >= 10 else AMBER if v >= 5 else "#43A047" for v in grp["adj_rate_pct"]]
b3 = ax2.bar(x, grp["adj_rate_pct"], width=bar_w, color=bar_colors, alpha=0.88, zorder=2)
ax2.axhline(y=5,  color=AMBER, linestyle="--", linewidth=1.1, alpha=0.7, label="5% threshold")
ax2.axhline(y=10, color=RED,   linestyle="--", linewidth=1.1, alpha=0.7, label="10% threshold")
ax2.set_title("Adjustment Rate (% of Total Billed) by Quarter", fontsize=11, fontweight="bold", pad=8)
ax2.set_ylabel("Adj % of Billed", fontsize=9)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax2.legend(fontsize=9, loc="upper right", framealpha=0.85)
for rect, val in zip(b3, grp["adj_rate_pct"]):
    ax2.text(rect.get_x() + rect.get_width() / 2, val + 0.05, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#212121")

# ── Panel 3: Invoice Line Item % with Adjustment ─────────────────────────
ax3 = axes[2]
b4 = ax3.bar(x, grp["adj_line_item_pct"], width=bar_w, color="#6A1B9A", alpha=0.82, zorder=2, label="% Line Items Adjusted")
# Overlay: line for adj line items count
ax3b = ax3.twinx()
ax3b.plot(x, grp["adj_line_items"], color="#FB8C00", linewidth=2, marker="o",
          markersize=6, label="Adj Line Item Count", zorder=4)
ax3b.set_ylabel("Adj Line Item Count", fontsize=9, color="#FB8C00")
ax3b.tick_params(axis="y", labelcolor="#FB8C00")
ax3b.spines[["top"]].set_visible(False)

ax3.set_title("Invoice Line Items with Adjustment — Count & % by Quarter", fontsize=11, fontweight="bold", pad=8)
ax3.set_ylabel("% Line Items with Adjustment", fontsize=9)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left", framealpha=0.85)

for rect, val in zip(b4, grp["adj_line_item_pct"]):
    ax3.text(rect.get_x() + rect.get_width() / 2, val + 0.05, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=8, fontweight="bold", color="#4A148C")

# ── X-axis labels (shared) ────────────────────────────────────────────────
axes[2].set_xticks(x)
axes[2].set_xticklabels(quarters, rotation=30, ha="right", fontsize=9)

fig.suptitle(
    "RCM Invoice Analysis — Quarterly Billing, Adjustment Rate & Line Item Adjustment %",
    fontsize=13, fontweight="bold", y=1.01, color="#212121"
)
plt.tight_layout()

out_dir  = r"C:\Users\Ravi\research\data-analysis\output"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "quarterly_adj_analysis_20260401.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {out_path}")

# ── Save summary CSV ──────────────────────────────────────────────────────
from loader import save_csv
import sys
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
csv_path = os.path.join(r"C:\Users\Ravi\Downloads", "quarterly_adj_summary_20260401.csv")
save_csv(grp[display_cols], csv_path)
print(f"Summary CSV: {csv_path}")
