import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from scipy import stats
import os

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Ravi\Downloads\payer_adjustment_summary_20260331.csv")

# ── Colour zones ───────────────────────────────────────────────────────────
def zone_color(pct):
    if pct >= 50:
        return "#C62828"   # Critical  — deep red
    elif pct >= 20:
        return "#E65100"   # High      — orange
    elif pct >= 5:
        return "#F9A825"   # Moderate  — amber
    elif pct > 0:
        return "#2E7D32"   # Low       — green
    else:
        return "#B0BEC5"   # Zero      — blue-grey

# ── Panel 1 data: payers with meaningful billing (>= $5k) ──────────────────
main = (
    df[df["total_billed"] >= 5000]
    .sort_values("adj_pct_billed", ascending=True)
    .reset_index(drop=True)
)
colors = [zone_color(p) for p in main["adj_pct_billed"]]

# ── Panel 2 data: all payers histogram ─────────────────────────────────────
all_pct = df["adj_pct_billed"].dropna()
skewness  = stats.skew(all_pct)
mean_val  = all_pct.mean()
median_val = all_pct.median()
zero_pct  = (all_pct == 0).mean() * 100

# ── Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, max(12, len(main) * 0.18 + 4)))
gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1.2], wspace=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

fig.patch.set_facecolor("#FAFAFA")
for ax in (ax1, ax2):
    ax.set_facecolor("#FAFAFA")

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Horizontal bar chart
# ═══════════════════════════════════════════════════════════════════════════
bars = ax1.barh(
    range(len(main)),
    main["adj_pct_billed"],
    color=colors,
    edgecolor="none",
    height=0.78,
    zorder=2,
)

# Reference lines
ref_lines = [(5, "#F9A825", "5% threshold"),
             (20, "#E65100", "20% threshold"),
             (50, "#C62828", "50% threshold")]
for xval, col, lbl in ref_lines:
    ax1.axvline(x=xval, color=col, linestyle="--", linewidth=1.2,
                alpha=0.7, zorder=3, label=lbl)

# Annotate the top 5 outliers
top5 = main.nlargest(5, "adj_pct_billed")
for _, row in top5.iterrows():
    idx = main.index[main["payorName"] == row["payorName"]].tolist()
    if not idx:
        continue
    i = idx[0]
    ax1.annotate(
        f"  {row['adj_pct_billed']:.1f}%",
        xy=(row["adj_pct_billed"], i),
        xytext=(row["adj_pct_billed"] + 0.5, i),
        fontsize=7.5,
        color="#C62828" if row["adj_pct_billed"] >= 50 else "#E65100",
        va="center",
        fontweight="bold",
    )

ax1.set_yticks(range(len(main)))
ax1.set_yticklabels(main["payorName"], fontsize=6.5)
ax1.set_xlabel("Adjustment as % of Billed Amount", fontsize=10, labelpad=8)
ax1.set_title(
    "Payer Adjustment Rate (% of Billed)  —  Payers with Billing >= $5,000",
    fontsize=11, fontweight="bold", pad=10
)
ax1.set_xlim(0, main["adj_pct_billed"].max() * 1.12)
ax1.tick_params(axis="y", length=0)
ax1.grid(axis="x", color="#E0E0E0", linewidth=0.6, zorder=1)
ax1.spines[["top", "right", "left"]].set_visible(False)

# Legend patches for zones
legend_patches = [
    mpatches.Patch(color="#C62828", label="Critical  >= 50%"),
    mpatches.Patch(color="#E65100", label="High      20-50%"),
    mpatches.Patch(color="#F9A825", label="Moderate  5-20%"),
    mpatches.Patch(color="#2E7D32", label="Low       0-5%"),
    mpatches.Patch(color="#B0BEC5", label="Zero      0%"),
]
ax1.legend(handles=legend_patches, loc="lower right", fontsize=8,
           framealpha=0.85, edgecolor="#BDBDBD", title="Adjustment Zone",
           title_fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════
# PANEL 2 — Distribution histogram (all 151 payers)
# ═══════════════════════════════════════════════════════════════════════════
# Bin strategy: 0% bucket, then 0-5, 5-20, 20-50, 50-100
bins = [0, 0.001, 5, 20, 50, 100]
n, bin_edges, patches_hist = ax2.hist(
    all_pct, bins=bins, orientation="horizontal",
    color="#4FC3F7", edgecolor="white", linewidth=0.8, zorder=2
)

# Color the zone segments
zone_colors_hist = ["#B0BEC5", "#2E7D32", "#F9A825", "#E65100", "#C62828"]
for patch, col in zip(patches_hist, zone_colors_hist):
    patch.set_facecolor(col)
    patch.set_alpha(0.85)

# Mean / median lines
ax2.axhline(y=mean_val,   color="#1565C0", linewidth=1.4,
            linestyle="-",  label=f"Mean   {mean_val:.2f}%", zorder=3)
ax2.axhline(y=median_val, color="#6A1B9A", linewidth=1.4,
            linestyle="--", label=f"Median {median_val:.2f}%", zorder=3)

# Stats annotation box
stats_text = (
    f"n = {len(all_pct)} payers\n"
    f"Skewness : {skewness:.2f}\n"
    f"Mean     : {mean_val:.2f}%\n"
    f"Median   : {median_val:.2f}%\n"
    f"Zero adj : {zero_pct:.0f}% of payers\n"
    f"Max      : {all_pct.max():.2f}%"
)
ax2.text(
    0.97, 0.97, stats_text,
    transform=ax2.transAxes,
    fontsize=8, va="top", ha="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
              edgecolor="#BDBDBD", alpha=0.9),
    fontfamily="monospace",
)

ax2.set_ylabel("Adjustment % of Billed", fontsize=9)
ax2.set_xlabel("Number of Payers", fontsize=9)
ax2.set_title("Distribution\n(all payers)", fontsize=10, fontweight="bold", pad=8)
ax2.legend(fontsize=8, loc="center right", framealpha=0.85)
ax2.grid(axis="x", color="#E0E0E0", linewidth=0.6, zorder=1)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylim(-2, 100)

# Skewness callout on distribution panel
ax2.annotate(
    f"Skewness = {skewness:.2f}\n(heavily right-skewed)",
    xy=(n[1] + 0.5, 3),
    xytext=(n[1] + 4, 18),
    fontsize=7.5,
    color="#C62828",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#C62828", lw=1),
)

# ── Supertitle ─────────────────────────────────────────────────────────────
fig.suptitle(
    "RCM Payer Adjustment Analysis  —  Adjustment as % of Billed Amount",
    fontsize=13, fontweight="bold", y=1.005, color="#212121"
)

out_dir  = r"C:\Users\Ravi\research\data-analysis\output"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "payer_adjustment_skew_20260401.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
