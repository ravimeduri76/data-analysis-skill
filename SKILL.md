---
name: data_analysis
description: Use this skill when the user asks to analyse a dataset or file, e.g. "analyse sales.csv", "what is the average revenue by region in data.xlsx?", "show me the correlation between price and quantity", "find outliers in orders.csv", "detect anomalies in my data", "profile this dataset", "data quality", "compare sampling techniques", or any request involving CSV, Excel, JSON, or Parquet files.
---

Perform end-to-end data analysis on the tabular dataset provided by the user.

> **Mandatory:** All analysis steps operate under the Core Operating Principles
> in `C:\Users\Ravi\.claude\CORE.md`. Key for this skill: P0 (truth over convenience),
> P1 (flag PII/sensitive data before processing), P3 (structured reasoning in every insight),
> P8 (ground in prior runs via memory hooks).

---

## How to invoke this skill

The user will say something like:
- "analyse sales.csv"
- "what is the average revenue by region in data.xlsx?"
- "show me the correlation between price and quantity"
- "find outliers in orders.csv"
- "profile this dataset"
- "check the data quality of my file"
- "compare these three sample files and tell me which is best"
- "detect anomalies in my product data"
- "find outliers in these numeric columns"
- "flag anything that looks wrong in this CSV"
- "survival analysis on customer churn"
- "how long before customers churn?"
- "predict when equipment will fail"
- "Kaplan-Meier curve for my data"
- "Cox model — what drives churn?"
- "predict survival probability for a new customer"

Extract `dataset_path` (or multiple paths) and `user_query` from the user message.
If `dataset_path` is not obvious, ask: *"Which file should I analyse?"*

**If multiple files are provided**, detect whether the intent is:
- **Sampling comparison** → use Workflow B below
- **Side-by-side analysis** → run Workflow A on each file then compare

**If the intent is anomaly or outlier detection** → use Workflow C below (can run standalone or after Workflow A Steps 1–4b).

**If the intent is survival or churn analysis** → use Workflow D below. Trigger phrases: "survival", "churn", "time-to-event", "Kaplan-Meier", "Cox model", "hazard ratio", "when will X fail/churn/lapse".

**If the intent is ML experimentation or prediction** → use Workflow E below. Trigger phrases: "train a model", "predict", "experiment", "ML", "classification", "regression", "automl", "leaderboard". Workflow E also runs automatically as Phase 3b in `kaggle_daily.py`.

---

## Strict execution rules

1. **NEVER fabricate numbers.** Always run Python code and show real output.
2. **Prefer hooks over LLM reasoning** — all profiling, statistics, classification, correlation, and missing-value checks must be done by executing Python, not described from memory.
3. Operations that are **always hooks (no LLM inference needed)**:
   - Loading and profiling the dataset
   - **Domain detection** (Step 1b — always run immediately after loading)
   - Column type classification (Identifier / Dimension / Fact / Context)
   - Descriptive statistics (`describe()`)
   - Missing-value report
   - Data quality report and commentary
   - Correlation matrix
   - Outlier / anomaly detection (IQR / z-score / mod_z / Claude semantic)
   - Top-N / frequency tables
   - Group-by aggregations
   - Pivot tables
   - Sampling technique comparison scorecard
   - Survival / churn analysis (Kaplan-Meier, Weibull, LogNormal, Cox PH, prediction)
4. Operations that **do require LLM reasoning** (after Python results are in hand):
   - Interpreting what the numbers mean **in domain-specific business context**
   - Deciding which analysis to run next
   - Explaining anomalies or suggesting hypotheses
   - Writing narrative summaries
5. **Domain context is mandatory for all LLM steps.** Every narrative, summary, or recommendation must be prefixed with the `DOMAIN_CTX` block produced in Step 1b. Never give generic advice when domain-specific guidance is available.

---

## Workflow A — Single-file end-to-end analysis

### Step 0 — Memory check (hook — run if POSTGRES_URL is set)

Retrieve relevant prior runs before starting analysis.

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")

conn_string = os.getenv("POSTGRES_URL")
if conn_string:
    try:
        from hooks.db_store import get_connection, init_db
        from hooks.semantic_store import recall_semantic, render_semantic_memory
        conn = get_connection(conn_string)
        init_db(conn)
        past_runs = recall_semantic(conn, query=query, limit=5)
        print(render_semantic_memory(past_runs, current_query=query))
        conn.close()
    except ImportError:
        # Fallback: sentence-transformers not installed — use keyword-based recall
        try:
            import hashlib
            from hooks.db_store import get_connection, init_db, recall_similar_runs, render_memory
            conn = get_connection(conn_string)
            init_db(conn)
            sha = hashlib.sha256(open(r"<dataset_path>", "rb").read()).hexdigest()
            past_runs = recall_similar_runs(conn, dataset_sha256=sha,
                                             dataset_name=os.path.basename(r"<dataset_path>"), limit=5)
            print(render_memory(past_runs, current_query=query))
            conn.close()
        except Exception as e:
            print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
    except Exception as e:
        print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
else:
    print("[Memory] POSTGRES_URL not set -- skipping memory check.")
```

---

### Step 1 — Load data (hook)

Run this Python block. Detect format from extension; support CSV, JSON, Excel, Parquet.

```python
import sys, json
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import load_dataframe

df = load_dataframe(r"<dataset_path>")
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.dtypes.to_string())
```

**Notes:**
- loader.py auto-installs whitelisted dependencies (openpyxl, odfpy, xlrd, etc.) — no manual `pip install` needed.
- If auto-install is disabled in `loader_config.json`, the error message tells you exactly what to install.
- The loader automatically skips blank leading rows and drops unnamed all-null padding columns common in Excel exports.

If loading fails for any reason other than a whitelisted dependency, report the error and stop.

---

### Step 1b — Detect domain (hook — always run immediately after Step 1)

**This step is mandatory.** It establishes the `DOMAIN_CTX` that is injected into every subsequent LLM reasoning step.

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")

from hooks.domain_detector import detect_domain, domain_role_prefix

domain_ctx = detect_domain(
    df,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),  # omit or None for heuristic fallback
    claude_model="claude-sonnet-4-6",
)

DOMAIN_CTX = domain_role_prefix(domain_ctx)
print(DOMAIN_CTX)
```

**What this produces:**
```
=== DOMAIN CONTEXT ===
Domain   : Electrical Components  (detected via claude, confidence: high)
Role     : You are an electrical components product data analyst...
Dataset  : This dataset contains product catalogue data for electrical connectors...
Lens     : Focus on specification integrity (voltage/current ratings, mounting types...)
Anomalies: Flag voltages above 1000V for standard connectors, pipe-delimited multi-values...
Key metrics for this domain: catalogue completeness %, anomaly rate by attribute, ...
======================
```

**Fallback behaviour:**
- If `ANTHROPIC_API_KEY` is not set or the API is unavailable, detection falls back to a keyword heuristic against column names and sample values — still produces a valid `DOMAIN_CTX`.
- The detected domain affects the analyst **role**, **analysis lens**, **anomaly guidance**, and **key metrics** used in all LLM steps.

**Supported domains:** Healthcare, Pharmaceutical, Industrial Equipment, Electrical Components, Financial Services / Lending, Insurance, Travel & Hospitality, Retail & E-commerce, HR & People Analytics, Education, General (fallback).

---

### Step 2 — Profile (hook — no LLM)

```python
from profiler import profile_dataframe
profile = profile_dataframe(df)
print(profile.to_string(index=False))
```

Show the table. Do not describe it yet — wait for Step 8.

---

### Step 3 — Classify columns (hook — no LLM)

```python
from classifier import classify_columns
schema = classify_columns(df)
print(json.dumps(schema, indent=2))
```

Show the classification. This determines which columns are candidates for grouping (Dimensions), aggregation (Facts), and display (Context).

---

### Step 4 — Missing-value report (hook — no LLM)

```python
from hooks.missing_values import missing_report
print(missing_report(df).to_string(index=False))
```

If any column has > 20% nulls, flag it prominently before proceeding.

---

### Step 4b — Data quality report (hook — no LLM)

**Always run this step.** It produces a per-column quality score and structured commentary covering nulls, constants, skew, outliers, and duplicates.

```python
from hooks.data_quality import quality_report, quality_commentary

# Per-column quality scores
qr = quality_report(df)
print(qr.to_string(index=False))

# Plain-English commentary with recommendations
print(quality_commentary(df))
```

**Quality score interpretation:**
| Score | Grade | Meaning |
|---|---|---|
| 90–100 | A | Excellent — no significant issues |
| 70–89  | B | Good — minor issues only |
| 55–69  | C | Fair — moderate issues, review before use |
| 40–54  | D | Poor — significant issues present |
| 0–39   | F | Critical — column may be unusable |

**Issue tags returned in `quality_report`:**
| Tag | Meaning | Recommended action |
|---|---|---|
| `ALL_NULL` | Column is 100% null | Drop before analysis |
| `CRITICAL_NULLS` | 80–99% null | Treat as sparse; impute only if justified |
| `HIGH_NULLS` | 20–79% null | Investigate null pattern (MCAR/MAR/MNAR) |
| `CONSTANT` | Single value ≥99.5% of rows | Drop from models; zero predictive value |
| `HIGH_SKEW` | \|skew\| ≥ 2.0 | Apply log/sqrt transform before modelling |
| `OUTLIERS` | >5% IQR outliers | Verify then cap/winsorise if needed |
| `LOW_CARDINALITY_NUMERIC` | Numeric but <0.5% unique | Consider treating as categorical |
| `OK` | No issues detected | — |

If the overall quality score is below 70, pause and highlight the critical issues to the user before continuing analysis.

---

### Step 5 — Interpret the user query (LLM step — use DOMAIN_CTX)

**Begin this step by printing `DOMAIN_CTX`** to ground your interpretation in the detected domain. Then map the user's question to columns and operations.

```
{DOMAIN_CTX}
```

Map the user's natural language question to columns identified above.
Decide which of the following operations to run (one or more):

| Intent keyword | Operation |
|---|---|
| average / mean / total / sum / count | Aggregation |
| top / bottom / largest / smallest | Sort + head |
| correlation / relationship | Correlation matrix |
| distribution / spread / histogram | Histogram + describe |
| outlier / anomaly / flag / wrong | **-> Workflow C** (anomaly & outlier detection) |
| compare / by / per / across | Group-by |
| trend / over time | Time-series resample |
| predict / forecast | ML (RandomForest) |
| identical / different / match / compare columns E-K / compare two ranges | **Block/range comparison** — see Block Comparison note below |

**Block Comparison note:** When the user asks to compare two column ranges (e.g. "are columns E-K and R-AG identical?"):
1. Check whether the two ranges have the **same number of columns**. If not, note the asymmetry and compare **semantic field pairs** (e.g. Legal Name vs Mailing Name) rather than whole-block identity.
2. For each semantic pair: `df[col_a].eq(df[col_b])` or `.fillna("").eq(...)` for nulls.
3. Produce a row-level summary: how many rows match on each pair, how many differ.
4. Flag columns that are entirely null on one side — this typically means a downstream pipeline has not yet written back results (not a source data error).

If intent is ambiguous, ask **one** clarifying question before proceeding.

---

### Step 6 — Analyse (hook — no LLM)

Run the appropriate hook from `research/data-analysis/hooks/`:

**Aggregation:**
```python
from hooks.aggregation import group_agg
result = group_agg(df, group_cols=[...], agg_col="<fact_col>", func="mean")
print(result.to_string())
```

**Correlation:**
```python
from hooks.correlation import correlation_report
print(correlation_report(df).to_string())
```

**Outliers:**
```python
from hooks.outliers import outlier_report
print(outlier_report(df, method="iqr").to_string())
```

**Descriptive stats:**
```python
print(df.describe(include="all").to_string())
```

**Frequency / top-N:**
```python
from hooks.frequency import top_n
print(top_n(df, col="<col>", n=10).to_string())
```

**Saving results to CSV** — always use `save_csv()` to avoid Windows encoding errors:
```python
from loader import save_csv
import os, datetime
timestamp = datetime.date.today().strftime("%Y%m%d")
out_dir = os.path.dirname(r"<dataset_path>")
out_path = save_csv(result_df, os.path.join(out_dir, f"analysis_result_{timestamp}.csv"))
print(f"Saved: {out_path}")
```

---

### Step 6b — Explainability (hook — always run after analysis)

Generate provenance cards for each key finding, showing HOW it was computed.

**After aggregation (Step 6):**
```python
from hooks.explainability import explain_groupby, render_explanation
total_rows = len(df)
cards = explain_groupby(result_df, group_cols=[...], agg_col="<fact_col>", func="mean", total_rows=total_rows)
```

**After data quality report (Step 4b):**
```python
from hooks.explainability import explain_quality
quality_cards = explain_quality(qr, total_rows)
```

**After anomaly detection (Workflow C Step C3):**
```python
from hooks.explainability import explain_anomalies
anomaly_cards = explain_anomalies(summary_df, domain_ctx)
```

**After missing-value report (Step 4):**
```python
from hooks.explainability import explain_missing
missing_df = missing_report(df)
missing_cards = explain_missing(missing_df, total_rows)
```

**Collect and render all cards:**
```python
all_cards = quality_cards + missing_cards + cards
print(render_explanation(all_cards, title="Analysis Provenance"))
```

---

### Step 6c — Trace (hook — always run after Step 6b)

Link all steps, artifacts, and InsightCards into a verifiable manifest.

```python
from hooks.trace_writer import AnalysisTrace

trace = AnalysisTrace(r"<dataset_path>", query, domain_ctx)

# Log each prior step
trace.log_step("load",          "Load dataset",           {"path": r"<dataset_path>"},                        f"{len(df)} rows x {len(df.columns)} cols")
trace.log_step("domain_detect", "Detect domain",          {"method": domain_ctx["method"]},                   domain_ctx["domain"])
trace.log_step("profile",       "Profile dataframe",      {},                                                  f"{len(df.columns)} columns profiled")
trace.log_step("classify",      "Classify columns",       {},                                                  "schema produced")
trace.log_step("missing",       "Missing value report",   {},                                                  f"{missing_df.shape[0]} columns with nulls")
trace.log_step("quality",       "Data quality report",    {},                                                  f"grade computed")
trace.log_step("analyse",       "Aggregation/analysis",   {"group_cols": [...], "agg_col": "<col>", "func": "mean"}, f"{len(result_df)} groups")

# Log artifact — call after each save_csv
trace.log_artifact(out_path, "Analysis result", len(result_df), len(result_df.columns))

# Log all InsightCards
for card in all_cards:
    trace.log_insight(card)
```

---

### Step 7 — Visualise (run if helpful)

```python
from visualizer import make_chart
make_chart(df, chart_type="<bar|hist|scatter|heatmap|line|box>", x="<col>", y="<col>",
           title="<title>", save_path=r"C:\Users\Ravi\research\data-analysis\output\chart.png")
```

**Requires:** `pip install matplotlib seaborn` — if either is missing the hook will raise an `ImportError` with the exact install command. Install and retry.

Charts are saved to the output folder — show the path to the user.

---

### Step 8 — Summarise (LLM step — use DOMAIN_CTX)

**Begin by reprinting `DOMAIN_CTX`** so the summary is clearly grounded in the detected domain.

After all Python results are collected, write a domain-aware plain-English summary structured as:

1. **Domain & role** — state the detected domain and the analyst perspective being applied
2. **What the data contains** — describe the dataset in domain-specific terms
3. **Key findings** — interpret numbers using domain terminology and `domain_ctx['key_metrics']`
4. **Data quality** — grade, critical issues, and domain-specific remediation steps (from Step 4b)
5. **Anomalies** — summarise flagged values using `domain_ctx['anomaly_guidance']` as interpretive frame
6. **Actionable next steps** — 2–3 recommendations phrased for a practitioner in this domain
7. **What-if analysis** — data gaps that would make this analysis more powerful

**Guardrails:**
- Never expose raw values from columns named `name`, `email`, `phone`, `ssn`, `password`, `dob`, or `nric` — call `sanitizer.sanitize(df)` first.
- If the dataset has > 50,000 rows, warn the user and sample 50,000 rows before profiling.
- If `sklearn` or `scipy` are unavailable, skip ML / statistical test steps and note the missing dependency.
- If `domain_ctx['method'] == 'heuristic'`, note the detected domain is based on keyword matching and invite the user to confirm or correct it.

```python
# What-If analysis
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.what_if import suggest_what_if, render_what_if

suggestions = suggest_what_if(domain_ctx, df, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(render_what_if(suggestions))

# Save trace manifest
out_dir = os.path.dirname(r"<dataset_path>")
json_path, md_path = trace.save_manifest(out_dir)
print(f"Trace JSON : {json_path}")
print(f"Trace MD   : {md_path}")
print(trace.render_breadcrumbs())

# Persist to database
if conn_string:
    try:
        from hooks.db_store import get_connection, save_run
        conn = get_connection(conn_string)
        saved_run_id = save_run(
            conn, trace, domain_ctx,
            findings_summary="<one-sentence summary of key findings>",
            analysis_type="profile",
            workflows_run=["A"],
            row_count=len(df),
            col_count=len(df.columns),
            what_if_suggestions=suggestions,
        )
        conn.close()
        print(f"[DB] Run persisted: {saved_run_id}")
    except Exception as e:
        print(f"[DB] Could not persist run: {e}")
```

---

### Step 9 — Self-improvement (hook — always run at end of Workflow A)

**Always run this step last.** It logs the run and auto-updates domain keywords, profiles, and skill notes for future runs.

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.self_improver import run_self_improvement, improvement_report

run_meta = {
    "dataset_path":       r"<dataset_path>",
    "query":              "<user_query>",
    "domain_detected":    domain_ctx["domain"],
    "domain_method":      domain_ctx["method"],
    "domain_confidence":  domain_ctx.get("confidence", "?"),
    "domain_correct":     None,   # set True/False if you know; None = unknown
    "domain_actual":      None,   # set to correct domain name if domain_correct is False
    "analysis_type":      "<profile|anomaly|comparison|aggregation|sampling>",
    "workflows_run":      ["A"],
    "row_count":          len(df),
    "col_count":          len(df.columns),
    "column_names":       df.columns.tolist(),
    "issues":             [],     # list any issues encountered, e.g. "domain_mismatch: detected X, actual Y"
    "findings_summary":   "<one sentence summary of key findings>",
}

result = run_self_improvement(run_meta, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(improvement_report(result))
```

**What this does:**
- Appends the run to `run_history.jsonl`
- If Claude API is available: asks Claude to review the run and suggest improvements
- Otherwise: uses keyword heuristics to detect domain mismatches and common issues
- **Auto-applies safe improvements** to `domain_overrides.json` (new keywords, new domain profiles) — picked up immediately on the next run
- **Appends lessons** to the `## Learned Patterns` section of this SKILL.md
- **Queues complex changes** to `pending_improvements.md` for human review

---

## Workflow B — Multi-file sampling technique comparison

Use this workflow when the user provides 2 or more files that are samples of the same underlying dataset and asks which preserves diversity best.

### Step B0 — Memory check (hook — run if POSTGRES_URL is set)

Retrieve relevant prior runs before starting analysis.

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")

conn_string = os.getenv("POSTGRES_URL")
if conn_string:
    try:
        from hooks.db_store import get_connection, init_db
        from hooks.semantic_store import recall_semantic, render_semantic_memory
        conn = get_connection(conn_string)
        init_db(conn)
        past_runs = recall_semantic(conn, query=query, limit=5)
        print(render_semantic_memory(past_runs, current_query=query))
        conn.close()
    except ImportError:
        # Fallback: sentence-transformers not installed — use keyword-based recall
        try:
            import hashlib
            from hooks.db_store import get_connection, init_db, recall_similar_runs, render_memory
            conn = get_connection(conn_string)
            init_db(conn)
            sha = hashlib.sha256(open(r"<dataset_path>", "rb").read()).hexdigest()
            past_runs = recall_similar_runs(conn, dataset_sha256=sha,
                                             dataset_name=os.path.basename(r"<dataset_path>"), limit=5)
            print(render_memory(past_runs, current_query=query))
            conn.close()
        except Exception as e:
            print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
    except Exception as e:
        print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
else:
    print("[Memory] POSTGRES_URL not set -- skipping memory check.")
```

---

### Step B1 — Load all samples (hook)

```python
import sys
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import load_dataframe

# Build a dict — keys are technique names, values are DataFrames
dfs = {
    "random": load_dataframe(r"<path_to_random.csv>"),
    "pam":    load_dataframe(r"<path_to_pam.csv>"),
    "clara":  load_dataframe(r"<path_to_clara.csv>"),
    # add more as needed
}
for name, df in dfs.items():
    print(f"[{name}] {df.shape[0]} rows x {df.shape[1]} columns")
```

---

### Step B2 — Diversity scorecard (hook — no LLM)

```python
from hooks.sampling_compare import compare_samples
scorecard = compare_samples(dfs)
print(scorecard.to_string(index=False))
```

**Scorecard columns explained:**
| Column | Meaning |
|---|---|
| `avg_cat_cardinality_%` | Avg % of combined unique categorical values retained |
| `avg_num_range_coverage_%` | Avg % of the combined numeric min–max range spanned |
| `cols_with_data_%` | % of columns that have at least 1 non-null value |
| `wholly_null_cols` | Columns that are entirely null in this sample |
| `avg_null_pct` | Average null% across all columns |
| `composite_score` | Weighted overall score (0–100) — higher = more representative |

**Composite score weights:**
- Categorical cardinality: 35%
- Numeric range coverage: 30%
- Column completeness: 25%
- Non-null density: 10%

---

### Step B3 — Detailed cardinality and range reports (hook — no LLM)

```python
from hooks.sampling_compare import cardinality_report, numeric_range_report

print("=== Categorical Cardinality Retention ===")
card = cardinality_report(dfs)
print(card.to_string(index=False))

print("\n=== Numeric Range Coverage ===")
rng = numeric_range_report(dfs)
print(rng.to_string(index=False))
```

---

### Step B4 — Null pattern comparison (hook — no LLM)

```python
from hooks.sampling_compare import null_completeness_report
null_r = null_completeness_report(dfs)
print(null_r.to_string(index=False))
```

The first key in `dfs` is used as the reference. Columns where another sample is 100% null but the reference is not are highlighted as diversity losses.

---

### Step B5 — Data quality on the winning sample (hook — no LLM)

After identifying the best sample, run the full quality assessment on it:

```python
from hooks.data_quality import quality_report, quality_commentary

winner_name = scorecard.iloc[0]["sample"]
winner_df   = dfs[winner_name]

qr = quality_report(winner_df)
print(qr.to_string(index=False))
print(quality_commentary(winner_df))
```

---

### Step B6 — Plain-English commentary (hook — no LLM)

```python
from hooks.sampling_compare import sampling_commentary
print(sampling_commentary(dfs))
```

This produces a ranked recommendation with per-metric breakdown, explanation of why cluster-based methods collapse diversity, and actionable next steps.

---

### Step B7 — Visualise (run if helpful)

Suggested charts for sampling comparison:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

out = r"C:\Users\Ravi\research\data-analysis\output"
os.makedirs(out, exist_ok=True)

# Grouped bar: scorecard metrics side by side
metrics    = ["avg_cat_cardinality_%", "avg_num_range_coverage_%", "cols_with_data_%"]
colors     = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]  # extend for more samples
samples    = scorecard["sample"].tolist()

x = range(len(metrics))
w = 0.8 / len(samples)
fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, color) in enumerate(zip(samples, colors)):
    vals = [float(scorecard.loc[scorecard["sample"]==name, m].iloc[0]) for m in metrics]
    ax.bar([xi + i*w for xi in x], vals, width=w, label=name.upper(), color=color)
ax.set_xticks([xi + w*(len(samples)-1)/2 for xi in x])
ax.set_xticklabels(["Cat Cardinality %", "Num Range %", "Col Completeness %"])
ax.set_ylabel("Score (%)")
ax.set_title("Sampling Diversity Scorecard")
ax.legend()
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(out, "sampling_scorecard.png"), dpi=150)
plt.close(fig)
print(f"Saved: {os.path.join(out, 'sampling_scorecard.png')}")
```

---

### Step B8 — Save scorecard and summarise (LLM step — use DOMAIN_CTX)

```python
from loader import save_csv
import datetime, os
ts  = datetime.date.today().strftime("%Y%m%d")
out_dir = r"C:\Users\Ravi\Documents\AI_Spreadsheet_Examples"
save_csv(scorecard, os.path.join(out_dir, f"sampling_scorecard_{ts}.csv"))
```

**Begin by reprinting `DOMAIN_CTX`.**

Then write a domain-aware plain-English summary covering:
- Which technique wins and by how much
- Why the losing techniques collapse diversity (for cluster-based methods: they select medoids, optimising for typicality not breadth)
- Concrete recommendation for when each technique is appropriate in this domain (e.g. for Healthcare: random for EDA, stratified by diagnosis group for model training)
- Whether stratified sampling would be a better hybrid — suggest stratification columns relevant to `domain_ctx['key_metrics']`
- Data quality findings on the winning sample, interpreted through the domain lens

```python
# What-If analysis
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.what_if import suggest_what_if, render_what_if
from hooks.explainability import explain_quality, render_explanation
from hooks.trace_writer import AnalysisTrace

# Explainability for winner quality
quality_cards = explain_quality(quality_report(winner_df), len(winner_df))
print(render_explanation(quality_cards, title="Winning Sample Quality Provenance"))

# What-if
suggestions = suggest_what_if(domain_ctx, winner_df, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(render_what_if(suggestions))

# Trace
trace = AnalysisTrace(r"<path_to_first_sample>", query, domain_ctx)
trace.log_step("load_samples",   "Load all sample files",      {"count": len(dfs)},         f"{len(dfs)} samples loaded")
trace.log_step("diversity_score","Diversity scorecard",         {},                          f"winner: {winner_name}")
trace.log_step("quality",        "Quality on winning sample",   {"sample": winner_name},     f"grade computed")
for card in quality_cards:
    trace.log_insight(card)
out_dir = r"C:\Users\Ravi\Documents\AI_Spreadsheet_Examples"
json_path, md_path = trace.save_manifest(out_dir)
print(f"Trace JSON : {json_path}")
print(f"Trace MD   : {md_path}")
print(trace.render_breadcrumbs())

# Persist to database
if conn_string:
    try:
        from hooks.db_store import get_connection, save_run
        conn = get_connection(conn_string)
        saved_run_id = save_run(
            conn, trace, domain_ctx,
            findings_summary="<one-sentence summary: winner technique and key diversity finding>",
            analysis_type="sampling",
            workflows_run=["B"],
            row_count=len(winner_df),
            col_count=len(winner_df.columns),
            what_if_suggestions=suggestions,
        )
        conn.close()
        print(f"[DB] Run persisted: {saved_run_id}")
    except Exception as e:
        print(f"[DB] Could not persist run: {e}")
```

---

### Step B9 — Self-improvement (hook — always run at end of Workflow B)

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.self_improver import run_self_improvement, improvement_report

# Use the winner_df for column introspection
run_meta = {
    "dataset_path":       r"<path_to_first_sample>",
    "query":              "<user_query>",
    "domain_detected":    domain_ctx["domain"],
    "domain_method":      domain_ctx["method"],
    "domain_confidence":  domain_ctx.get("confidence", "?"),
    "domain_correct":     None,
    "domain_actual":      None,
    "analysis_type":      "sampling",
    "workflows_run":      ["B"],
    "row_count":          winner_df.shape[0],
    "col_count":          winner_df.shape[1],
    "column_names":       winner_df.columns.tolist(),
    "issues":             [],
    "findings_summary":   f"Best sample: {winner_name}. Composite score: {scorecard.iloc[0]['composite_score']:.1f}",
}

result = run_self_improvement(run_meta, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(improvement_report(result))
```

---

## Workflow C — Anomaly & Outlier Detection

Use this workflow when the user asks to detect anomalies, flag bad values, find outliers, or check for data entry errors in a tabular file.

**Trigger phrases:** "detect anomalies", "find outliers", "flag wrong values", "anything out of place", "data entry errors", "validate my data"

**Two detection modes run together:**
- **Numeric columns** — statistical outlier detection (modified z-score / z-score / IQR)
- **Categorical/text columns** — exact-match → fuzzy-match → Claude semantic judgment (optional, needs `ANTHROPIC_API_KEY`)

---

### Step C0 — Memory check (hook — run if POSTGRES_URL is set)

Retrieve relevant prior runs before starting analysis.

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")

conn_string = os.getenv("POSTGRES_URL")
if conn_string:
    try:
        from hooks.db_store import get_connection, init_db
        from hooks.semantic_store import recall_semantic, render_semantic_memory
        conn = get_connection(conn_string)
        init_db(conn)
        past_runs = recall_semantic(conn, query=query, limit=5)
        print(render_semantic_memory(past_runs, current_query=query))
        conn.close()
    except ImportError:
        # Fallback: sentence-transformers not installed — use keyword-based recall
        try:
            import hashlib
            from hooks.db_store import get_connection, init_db, recall_similar_runs, render_memory
            conn = get_connection(conn_string)
            init_db(conn)
            sha = hashlib.sha256(open(r"<dataset_path>", "rb").read()).hexdigest()
            past_runs = recall_similar_runs(conn, dataset_sha256=sha,
                                             dataset_name=os.path.basename(r"<dataset_path>"), limit=5)
            print(render_memory(past_runs, current_query=query))
            conn.close()
        except Exception as e:
            print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
    except Exception as e:
        print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
else:
    print("[Memory] POSTGRES_URL not set -- skipping memory check.")
```

---

### Step C1 — Load data (hook)

If not already loaded from Workflow A Step 1:

```python
import sys, os
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import load_dataframe

df = load_dataframe(r"<dataset_path>")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
```

---

### Step C2 — Configure columns (LLM step — use DOMAIN_CTX)

**Begin by reprinting `DOMAIN_CTX`.** Use `domain_ctx['anomaly_guidance']` to inform which columns are highest priority and what kinds of anomalies to expect.

Decide which columns to analyse and whether Claude semantic judgment is needed.

**Option A — Fully automatic (no config, no Claude):**
Numeric columns are auto-detected. Categorical columns are only included if you pass them in `cat_cols`.
```python
# Pure statistical detection on all auto-detected numeric columns
columns_config = {}
cat_cols_to_check = []   # add column names here to check text columns without Claude
```

**Option B — Guided with columns_config + Claude (recommended for text/categorical):**
Build a `columns_config` dict from what you know about the data.
Each entry maps a column name to its expected values and a plain-English domain description.
```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")   # loads ANTHROPIC_API_KEY

columns_config = {
    "<col_name>": {
        "expected_values": ["Value A", "Value B", "Value C"],   # canonical valid values
        "domain_prompt":   "Short description of what this column represents.",
        # "type": "numeric"   # uncomment to force numeric treatment
    },
    # add more columns as needed
}
```

**Choosing the numeric method:**
| Method | When to use |
|---|---|
| `mod_z` (default) | Robust — best for skewed data or when outliers exist |
| `z` | Use when data is normally distributed and outlier-free |
| `iqr` | Non-parametric; good for heavily skewed distributions |

---

### Step C3 — Run detection (hook — no LLM)

```python
from hooks.anomaly_detection import detect_anomalies, anomaly_summary

annotated_df, summary_df = detect_anomalies(
    df,
    columns_config=columns_config,        # omit or pass {} for fully auto mode
    # cat_cols=cat_cols_to_check,         # uncomment for Option A categorical check
    method="mod_z",                        # 'z' | 'mod_z' | 'iqr'
    z_threshold=3.0,                       # flag threshold for z/mod_z
    iqr_factor=1.5,                        # fence multiplier for iqr
    pipe_agg="mean",                       # how to resolve pipe-delimited numerics
    fuzzy_threshold=0.88,                  # min fuzzy-match ratio to accept as Normal
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),   # omit if no Claude
    claude_model="claude-sonnet-4-6",
)

print(summary_df.to_string(index=False))
```

**Summary DataFrame columns:**
| Column | Meaning |
|---|---|
| `column` | Column name |
| `type` | `numeric` or `categorical` |
| `method` | Detection method used |
| `non_null` | Count of non-null values analysed |
| `anomaly_count` | Number of anomalous values found |
| `anomaly_pct` | Percentage of non-null values flagged |
| `col_mean` / `col_std` | Distribution stats (numeric only) |
| `anomaly_min` / `anomaly_max` | Range of outlier values (numeric only) |
| `claude_calls` | Number of Claude API calls made (categorical only) |

---

### Step C4 — Plain-English commentary (hook — no LLM)

```python
print(anomaly_summary(summary_df))
```

---

### Step C5 — Inspect anomalous rows (hook — no LLM)

For any column of interest, pull the actual flagged rows:

```python
# Numeric: rows where flag == True
col = "<col_name>"
anomaly_rows = annotated_df[annotated_df[f"{col}_flag"] == True][[col, f"{col}_score", f"{col}_detected_anomaly"]]
print(f"\nNumeric anomalies in '{col}': {len(anomaly_rows)}")
print(anomaly_rows.head(20).to_string(index=False))

# Categorical: rows where flag contains "Anomaly"
col = "<col_name>"
anomaly_rows = annotated_df[annotated_df[f"{col}_flag"].str.startswith("Anomaly", na=False)]
print(f"\nCategorical anomalies in '{col}': {len(anomaly_rows)}")
print(anomaly_rows[[col, f"{col}_score", f"{col}_flag"]].drop_duplicates(subset=[col]).to_string(index=False))
```

---

### Step C6 — Save annotated output (hook — no LLM)

```python
from loader import save_csv
import datetime, os

ts = datetime.date.today().strftime("%Y%m%d")
out_dir = os.path.dirname(r"<dataset_path>")
out_path = save_csv(annotated_df, os.path.join(out_dir, f"anomaly_output_{ts}.csv"))
print(f"Saved: {out_path}")
```

The output file contains all original columns plus `{col}_score`, `{col}_flag`, and `{col}_detected_anomaly` triples inserted immediately after each analysed column.

---

### Step C7 — Visualise (run if helpful)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

out = r"C:\Users\Ravi\research\data-analysis\output"
os.makedirs(out, exist_ok=True)

# Bar chart: anomaly count per column
fig, ax = plt.subplots(figsize=(10, 5))
flagged = summary_df[summary_df["anomaly_count"] > 0].sort_values("anomaly_count", ascending=True)
colors = ["#E53935" if t == "numeric" else "#FB8C00" for t in flagged["type"]]
ax.barh(flagged["column"], flagged["anomaly_count"], color=colors)
ax.set_xlabel("Anomaly count")
ax.set_title("Anomalies per column")
# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#E53935", label="Numeric"), Patch(color="#FB8C00", label="Categorical")])
plt.tight_layout()
chart_path = os.path.join(out, "anomaly_counts.png")
fig.savefig(chart_path, dpi=150)
plt.close(fig)
print(f"Saved: {chart_path}")
```

---

### Step C8 — Summarise (LLM step — use DOMAIN_CTX)

**Begin by reprinting `DOMAIN_CTX`.** Interpret every anomaly finding through the domain lens.

After all Python results are collected, write a domain-aware anomaly summary covering:
- Detected domain and analyst role being applied
- How many columns were analysed and total anomaly count
- For each flagged column: what the anomaly pattern is, what it likely means in domain context (using `domain_ctx['anomaly_guidance']`), and the specific recommended remediation
- Whether Claude semantic judgment was used and any API issues to flag
- 2–3 prioritised next steps phrased for a practitioner in this domain (e.g. for Electrical Components: "Standardise voltage rating format before loading to PIM system")

**Guardrails:**
- If `anthropic_api_key` is None or the API returns a quota error, note that categorical judgment fell back to vocabulary-only detection and recommend topping up Anthropic credits to enable semantic scoring.
- If `scipy` is unavailable, `mod_z` will raise an ImportError — install with `pip install scipy` and retry.
- Null/missing values are never flagged as anomalies — they appear as `Normal (null/missing)` in the flag column.
- If `domain_ctx['method'] == 'heuristic'`, note the domain was auto-detected and invite the user to confirm.

```python
# Explainability for anomalies
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.explainability import explain_anomalies, render_explanation
from hooks.what_if import suggest_what_if, render_what_if
from hooks.trace_writer import AnalysisTrace

anomaly_cards = explain_anomalies(summary_df, domain_ctx)
print(render_explanation(anomaly_cards, title="Anomaly Detection Provenance"))

# What-if
suggestions = suggest_what_if(domain_ctx, df, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(render_what_if(suggestions))

# Trace
trace = AnalysisTrace(r"<dataset_path>", query, domain_ctx)
trace.log_step("load",          "Load dataset",         {"path": r"<dataset_path>"},          f"{len(df)} rows x {len(df.columns)} cols")
trace.log_step("domain_detect", "Detect domain",        {"method": domain_ctx["method"]},     domain_ctx["domain"])
trace.log_step("anomaly_detect","Run anomaly detection", {"method": "mod_z"},                  f"{int(summary_df['anomaly_count'].sum())} anomalies found")
trace.log_artifact(out_path, "Annotated anomaly output", len(annotated_df), len(annotated_df.columns))
for card in anomaly_cards:
    trace.log_insight(card)
out_dir = os.path.dirname(r"<dataset_path>")
json_path, md_path = trace.save_manifest(out_dir)
print(f"Trace JSON : {json_path}")
print(f"Trace MD   : {md_path}")
print(trace.render_breadcrumbs())

# Persist to database
if conn_string:
    try:
        from hooks.db_store import get_connection, save_run
        conn = get_connection(conn_string)
        saved_run_id = save_run(
            conn, trace, domain_ctx,
            findings_summary="<one-sentence summary: total anomalies found and top flagged column>",
            analysis_type="anomaly",
            workflows_run=["C"],
            row_count=len(df),
            col_count=len(df.columns),
            what_if_suggestions=suggestions,
        )
        conn.close()
        print(f"[DB] Run persisted: {saved_run_id}")
    except Exception as e:
        print(f"[DB] Could not persist run: {e}")
```

---

### Step C9 — Self-improvement (hook — always run at end of Workflow C)

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.self_improver import run_self_improvement, improvement_report

# Collect any issues found during this anomaly run
anomaly_issues = []
if domain_ctx.get("method") == "heuristic":
    anomaly_issues.append(f"domain_heuristic: domain '{domain_ctx['domain']}' detected by keyword matching only")

# Note columns with high anomaly rates
for _, row in summary_df[summary_df["anomaly_pct"] > 20].iterrows():
    anomaly_issues.append(f"high_anomaly_rate: {row['column']} has {row['anomaly_pct']:.1f}% anomalies")

run_meta = {
    "dataset_path":       r"<dataset_path>",
    "query":              "<user_query>",
    "domain_detected":    domain_ctx["domain"],
    "domain_method":      domain_ctx["method"],
    "domain_confidence":  domain_ctx.get("confidence", "?"),
    "domain_correct":     None,
    "domain_actual":      None,
    "analysis_type":      "anomaly",
    "workflows_run":      ["C"],
    "row_count":          len(df),
    "col_count":          len(df.columns),
    "column_names":       df.columns.tolist(),
    "issues":             anomaly_issues,
    "findings_summary":   (
        f"{int(summary_df['anomaly_count'].sum())} total anomalies across "
        f"{len(summary_df[summary_df['anomaly_count']>0])} columns"
    ),
}

result = run_self_improvement(run_meta, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(improvement_report(result))
```


---

## Workflow D — Survival & Churn Analysis

Use this workflow when the user asks about time-to-event data: customer churn, equipment failure, patient survival, employee attrition, loan default timing, etc.

### Step D0 — Memory check (hook — run if POSTGRES_URL is set)

Retrieve relevant prior runs before starting analysis.

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")

conn_string = os.getenv("POSTGRES_URL")
if conn_string:
    try:
        from hooks.db_store import get_connection, init_db
        from hooks.semantic_store import recall_semantic, render_semantic_memory
        conn = get_connection(conn_string)
        init_db(conn)
        past_runs = recall_semantic(conn, query=query, limit=5)
        print(render_semantic_memory(past_runs, current_query=query))
        conn.close()
    except ImportError:
        # Fallback: sentence-transformers not installed — use keyword-based recall
        try:
            import hashlib
            from hooks.db_store import get_connection, init_db, recall_similar_runs, render_memory
            conn = get_connection(conn_string)
            init_db(conn)
            sha = hashlib.sha256(open(r"<dataset_path>", "rb").read()).hexdigest()
            past_runs = recall_similar_runs(conn, dataset_sha256=sha,
                                             dataset_name=os.path.basename(r"<dataset_path>"), limit=5)
            print(render_memory(past_runs, current_query=query))
            conn.close()
        except Exception as e:
            print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
    except Exception as e:
        print(f"[Memory] Database unavailable: {e} -- continuing without memory.")
else:
    print("[Memory] POSTGRES_URL not set -- skipping memory check.")
```

---

**Two modelling tracks — run both by default:**
- **Without covariates:** Kaplan-Meier (non-parametric baseline) + Weibull / LogNormal comparison (parametric)
- **With covariates:** Cox Proportional Hazards (semi-parametric — learns what *drives* risk)

**Key concepts:**
| Term | Meaning |
|---|---|
| `duration` | Time from start until event or last observation |
| `event = 1` | Event occurred (churn, failure, death) |
| `event = 0` | Censored — not yet observed (still active, lost to follow-up) |
| Median survival | Time at which S(t) crosses 0.5 — half the population has experienced the event |
| Hazard ratio (HR) | Cox: HR > 1 = higher risk; HR < 1 = protective; HR = 1 = no effect |
| C-index | Cox discrimination: 0.5 = random, 0.7+ = good, 0.8+ = excellent |

**Install dependency if missing:**
```bash
pip install lifelines
```

---

### Step D1 — Load data (hook)

If not already loaded:

```python
import sys, os
sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from loader import load_dataframe

df = load_dataframe(r"<dataset_path>")
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(df.dtypes.to_string())
```

---

### Step D1b — Detect domain (hook — always run)

```python
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.domain_detector import detect_domain, domain_role_prefix

domain_ctx = detect_domain(df, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
DOMAIN_CTX = domain_role_prefix(domain_ctx)
print(DOMAIN_CTX)
```

---

### Step D1c — Feature engineering (LLM step — use DOMAIN_CTX)

**Run this step when `duration_col` does not already exist in your dataset.** This is common for membership rosters, HR records, and any dataset with start/end date columns rather than a pre-computed duration.

**Option A — Let the hook compute duration from date columns (simplest):**
```python
dataset = prepare_survival_data(
    df,
    start_date_col="Membership Start Date",        # raw date column
    end_date_col="Membership Termination Date",    # null = still active (censored)
    analysis_date=pd.Timestamp("2026-03-25"),      # reference date for censored rows
    event_col="<event_col>",
    covariate_cols=["start_year"],
    categorical_cols=["class_of_trade","org_status","region"],
)
```

**Option B — Engineer manually for full control:**
```python
import numpy as np, pandas as pd

ANALYSIS_DATE = pd.Timestamp("today").normalize()

# 1. Event: 1 = churned/inactive, 0 = still active (censored)
df["event"] = (df["<status_col>"].str.strip().str.lower() == "<inactive_value>").astype(int)

# 2. Duration: use termination date for events; analysis date for active/censored
df["<start_col>"] = pd.to_datetime(df["<start_col>"], format="mixed", dayfirst=False, errors="coerce")
df["<end_col>"]   = pd.to_datetime(df["<end_col>"],   format="mixed", dayfirst=False, errors="coerce")
df["end_date"]    = pd.to_datetime(np.where(df["event"]==0, ANALYSIS_DATE, df["<end_col>"].fillna(ANALYSIS_DATE)))
df["duration_days"] = (df["end_date"] - df["<start_col>"]).dt.days

# 3. Covariates
df["start_year"] = df["<start_col>"].dt.year   # cohort covariate

# 4. Region mapping (US example — reduce 50-state cardinality)
REGION_MAP = {
    "AL":"South","AR":"South","FL":"South","GA":"South","TX":"South","VA":"South",
    "AZ":"West","CA":"West","CO":"West","OR":"West","WA":"West",
    "IL":"Midwest","MN":"Midwest","OH":"Midwest","MI":"Midwest","WI":"Midwest",
    "CT":"Northeast","MA":"Northeast","NY":"Northeast","PA":"Northeast","NJ":"Northeast",
}
df["region"] = df["<state_col>"].str.strip().str.upper().map(REGION_MAP).fillna("Other")
```

**Notes:**
- Always use `format="mixed"` for date parsing to handle mixed formats (e.g. `MM-DD-YYYY` and `M/DD/YYYY` in the same column)
- Active/censored members contribute to the analysis up to `analysis_date` — they are NOT excluded
- If event rate < 5%, KM confidence intervals will be wide; warn the user
- High-cardinality categoricals (e.g. 50 US states) should be grouped before encoding to avoid too many dummy variables in Cox

---

### Step D2 — Prepare survival data (hook — no LLM)

**This step is mandatory.** It validates the dataset, detects or confirms the duration/event columns, encodes categoricals, removes impossible durations, and imputes missing covariates.

```python
from hooks.survival_analysis import prepare_survival_data

dataset = prepare_survival_data(
    df,
    duration_col="<duration_col>",   # e.g. "tenure_days", "months_active" — or omit to auto-detect
    event_col="<event_col>",          # e.g. "churned", "failed" (1=event, 0=censored) — or omit
    covariate_cols=["<col1>", "<col2>"],  # numeric covariates for Cox — or omit to auto-include all numeric
    categorical_cols=["<cat_col>"],       # categorical columns to one-hot encode for Cox — or omit
)

print(f"Observations : {dataset.n_total:,}")
print(f"Events       : {dataset.n_events:,}  ({dataset.n_events/dataset.n_total*100:.1f}%)")
print(f"Censored     : {dataset.n_censored:,}")
print(f"Duration col : {dataset.duration_col}")
print(f"Event col    : {dataset.event_col}")
print(f"Covariates   : {dataset.covariate_cols}")
if dataset.warnings:
    print("\nData prep notes:")
    for w in dataset.warnings:
        print(f"  - {w}")
```

**Important data requirements:**
- Duration must be **strictly positive** — zero/negative durations are dropped with a warning
- Event must be **binary** (0/1) — non-binary values are coerced with a warning
- Censored records are valid: they contribute information up to their last observation time
- If the event rate is < 5%, results may be unstable — note this to the user

**Column auto-detection** scans for names containing: `duration`, `time`, `tenure`, `days`, `months`, `survival` (duration); `event`, `churn`, `died`, `failure`, `status`, `outcome` (event).

---

### Step D3 — Kaplan-Meier (hook — no LLM)

```python
from hooks.survival_analysis import fit_kaplan_meier

km_result = fit_kaplan_meier(
    dataset,
    stratify_col="<cat_col>",   # optional: split curves by a group (e.g. "segment", "plan_type")
    time_points=None,            # default: [Q25, median, Q75, max] of duration distribution
)

print(f"Median survival time : {km_result.median_survival:.2f}")
print("\nSurvival probabilities:")
for t, p in km_result.survival_at_times.items():
    print(f"  S(t={t:.1f}) = {p:.4f}  ({p*100:.1f}% survive past this point)")

if km_result.group_kmfs:
    print(f"\nStratified by '{km_result.stratify_col}':")
    for grp, kmf_g in km_result.group_kmfs.items():
        print(f"  Group '{grp}' — median survival = {float(kmf_g.median_survival_time_):.2f}")
    if km_result.logrank_pvalue is not None:
        print(f"  Log-rank test p = {km_result.logrank_pvalue:.4f}")
```

**Reading KM results:**
- Median survival < half the study window = most entities churn/fail before the midpoint
- Flat tail = a long-surviving cohort — investigate what distinguishes them
- Log-rank p < 0.05 = the two groups have significantly different survival curves

---

### Step D4 — Parametric model comparison (hook — no LLM)

```python
from hooks.survival_analysis import fit_parametric

param_result = fit_parametric(dataset)

print(f"Weibull   AIC = {param_result.weibull_aic:.2f}  (lambda={param_result.weibull_lambda:.4f}, rho={param_result.weibull_rho:.4f})")
print(f"LogNormal AIC = {param_result.lognormal_aic:.2f}  (mu={param_result.lognormal_mu:.4f}, sigma={param_result.lognormal_sigma:.4f})")
print(f"Best fit  : {param_result.better_model}  (lower AIC wins)")
```

**Choosing between models:**
| Weibull rho | Hazard shape | Typical context |
|---|---|---|
| rho > 1 | Increasing (wear-out) | Equipment ageing, late-stage churn |
| rho ≈ 1 | Constant (memoryless) | Random failures, steady churn rate |
| rho < 1 | Decreasing (infant mortality) | Early failures, onboarding-period churn |
| LogNormal preferred | Peaks then drops | Mid-tenure churn peak, treatment-response curves |

Use the AIC delta (|ΔAIC| > 10 = decisive, 4–10 = substantial) to justify model choice.

---

### Step D5 — Cox Proportional Hazards (hook — no LLM)

Run this step when you have covariate columns (numeric or encoded categorical) and want to understand what *drives* survival risk.

```python
from hooks.survival_analysis import fit_cox

cox_result = fit_cox(
    dataset,
    covariates=dataset.covariate_cols,  # or specify a subset: ["age", "segment_B"]
    penalizer=0.1,                       # increase to 0.5 if convergence warnings appear
)

print(f"C-index : {cox_result.concordance:.4f}  (0.5=random, 0.7=good, 0.8=excellent)")
print("\nHazard Ratios:")
print(cox_result.summary_df.to_string(index=False))
print(f"\nSignificant covariates (p<0.05): {cox_result.significant_covariates}")
```

**Interpreting hazard ratios:**
- HR = 1.5 on `age` → each 1-unit increase in age raises hazard (risk) by 50%
- HR = 0.7 on `loyalty_tier` → loyalty tier reduces hazard by 30% (protective)
- Encoded dummy HR: HR = 2.0 on `segment_B` → segment B has 2× the hazard of **the reference category**

**Reference category:** When a categorical column is one-hot encoded with `drop_first=True`, the first category (alphabetically) is dropped and becomes the **reference**. All dummy HRs are *relative to that reference*. For example, if `class_of_trade` encodes to `_Ambulatory Care`, `_Home Care`, `_Other` (dropping `_Acute Care`), then HR = 2.8 on `_Home Care` means "Home Care has 2.8× the churn hazard of Acute Care", not 2.8× the population average. Always identify the reference before interpreting dummy HRs.

**`summary_df` column names:** `covariate`, `hazard_ratio`, `hr_lower_95`, `hr_upper_95`, `p_value`, `z_score`

**C-index interpretation:**
| C-index | Quality |
|---|---|
| > 0.8 | Excellent |
| 0.7–0.8 | Good |
| 0.6–0.7 | Fair — add behavioral covariates |
| < 0.6 | Poor — covariates explain little timing |

---

### Step D6 — Predict survival for a new entity (hook — no LLM)

```python
from hooks.survival_analysis import predict_survival

# Specify the time points you care about (in the same units as duration_col)
time_points = [30, 90, 180, 365]   # e.g. days

# Population-level predictions (KM + parametric)
preds = predict_survival(
    dataset,
    time_points=time_points,
    km_result=km_result,
    param_result=param_result,
)

# Individual prediction using Cox (pass covariate values for one entity)
preds = predict_survival(
    dataset,
    time_points=time_points,
    km_result=km_result,
    param_result=param_result,
    cox_result=cox_result,
    new_entity={"age": 45, "segment_B": 1.0},  # adjust to your covariate columns
)

# Display predictions table
import pandas as pd
rows = []
for model, values in preds.items():
    if model == "time_points" or not isinstance(values, dict):
        continue
    for t, p in values.items():
        rows.append({"model": model, "time": t, "survival_prob": round(p, 4)})
pred_df = pd.DataFrame(rows).pivot(index="time", columns="model", values="survival_prob")
print(pred_df.to_string())
```

---

### Step D6b — Batch prediction for a cohort (hook — run when scoring many entities)

Use this step when you need churn/survival probabilities for **all active (censored) members**, or any defined cohort. `predict_batch` handles encoding alignment internally — pass the raw un-encoded DataFrame.

```python
from hooks.survival_analysis import predict_batch
from loader import save_csv
import datetime, os

# Select the cohort to score (e.g. all currently active members)
active_df = df[df["event"] == 0].copy()

scored = predict_batch(
    dataset,
    cox_result,
    new_df=active_df,
    time_horizons=[365, 730, 1825],              # 1yr, 2yr, 5yr
    categorical_cols=["class_of_trade","org_status","region"],  # same as prepare_survival_data
    id_cols=["<member_id_col>", "<member_name_col>"],   # carried through unchanged
    risk_tier_method="percentile",               # top 5% = High, next 15% = Medium (population-relative)
    # risk_tier_method="absolute"               # use if you want fixed thresholds instead
)

print(scored["risk_tier"].value_counts())
print(scored[["<id_col>", "churn_prob_2yr", "churn_prob_5yr", "risk_tier"]].head(20).to_string(index=False))

ts = datetime.date.today().strftime("%Y%m%d")
out_path = save_csv(scored, os.path.join(out_dir, f"churn_predictions_{ts}.csv"))
print(f"Saved: {out_path}")
```

**Risk tier method choice:**
| Method | When to use |
|---|---|
| `percentile` (default) | Population is homogeneous; absolute rates are low (e.g. 9% mean 3yr churn). Top 5% = High is meaningful even when all probabilities are low. |
| `absolute` | Population has wide risk spread; domain has clear action thresholds (e.g. >35% churn = urgent intervention). |

---

### Step D7 — Plot (hook — always run)

```python
from hooks.survival_analysis import plot_km, plot_model_comparison, plot_cox
import os

out_dir = r"C:\Users\Ravi\research\data-analysis\output"

# KM curve(s)
km_paths = plot_km(km_result, out_dir, title="<descriptive title>")
for p in km_paths:
    print(f"Saved: {p}")

# Model comparison overlay (KM vs Weibull vs LogNormal)
comp_path = plot_model_comparison(km_result, param_result, out_dir)
print(f"Saved: {comp_path}")

# Cox hazard ratio forest plot + baseline survival
cox_paths = plot_cox(cox_result, out_dir)
for p in cox_paths:
    print(f"Saved: {p}")
```

---

### Step D8 — Summarise (LLM step — use DOMAIN_CTX)

**Begin by reprinting `DOMAIN_CTX`.** Interpret every finding through the domain lens.

```python
from hooks.survival_analysis import survival_summary
print(survival_summary(dataset, km_result, param_result, cox_result, domain_ctx))
```

After the hook output, write a domain-aware narrative covering:

1. **What the survival dataset represents** — define the "event" in business terms (e.g. "a customer cancelling their subscription", "a bearing reaching end of life")
2. **KM findings** — median survival time in plain language; group differences and whether they are statistically significant
3. **Hazard shape** — interpret Weibull rho or LogNormal preference in domain terms (e.g. "churn risk increases with tenure" vs "early-life failures dominate")
4. **Cox drivers** — for each significant covariate, explain the hazard ratio in plain business language (avoid "HR" jargon — say "customers aged over 60 are 40% more likely to churn per month")
5. **Individual risk** — if new_entity was predicted, state clearly what probability was estimated and what that means for action
6. **Actionable next steps** — 2–3 domain-specific recommendations (e.g. for HR churn: "target retention interventions at employees in their 18–24 month tenure band where hazard peaks")

**Guardrails:**
- If `lifelines` is not installed, the hook raises `ImportError` with the install command — do not proceed without it.
- If event rate < 5%, flag that KM confidence intervals will be wide and Cox estimates unreliable.
- If `domain_ctx['method'] == 'heuristic'`, note the domain was auto-detected.
- Censored observations are **not** anomalies — they are expected and correctly handled by all models.
- Never interpret a HR without checking p-value and CI — an HR of 2.0 with p=0.4 is not a finding.

```python
# What-If and Trace
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.what_if import suggest_what_if, render_what_if
from hooks.trace_writer import AnalysisTrace
from hooks.explainability import render_explanation, InsightCard
import dataclasses

# Explainability cards for survival results
surv_cards = []
for grp, kmf_g in km_result.group_kmfs.items():
    surv_cards.append(InsightCard(
        finding=f"Group '{grp}' median survival = {float(kmf_g.median_survival_time_):.2f}",
        metric=f"KM median",
        source_hook="hooks/survival_analysis.py::fit_kaplan_meier",
        columns_used=f"{dataset.duration_col}, {dataset.event_col}, {km_result.stratify_col or 'N/A'}",
        rows_analyzed=dataset.n_total,
        total_rows=dataset.n_total,
        method="Kaplan-Meier",
        confidence="high",
        caveats="Non-parametric estimate; confidence intervals widen with fewer events",
        sql_equivalent="N/A -- requires survival analysis library",
    ))
print(render_explanation(surv_cards, "Survival Analysis Provenance"))

# What-if
suggestions = suggest_what_if(domain_ctx, df, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(render_what_if(suggestions))

# Trace
trace = AnalysisTrace(r"<dataset_path>", query, domain_ctx)
trace.log_step("load",        "Load dataset",          {"path": r"<dataset_path>"},       f"{dataset.n_total} rows")
trace.log_step("domain",      "Detect domain",         {"method": domain_ctx["method"]},  domain_ctx["domain"])
trace.log_step("prepare",     "Prepare survival data", {"duration": dataset.duration_col, "event": dataset.event_col}, f"{dataset.n_events} events")
trace.log_step("kaplan_meier","Fit KM",                {"stratify_col": str(km_result.stratify_col)}, f"median={km_result.median_survival:.2f}")
trace.log_step("parametric",  "Fit Weibull+LogNormal", {},  f"best={param_result.better_model}" if param_result else "skipped")
trace.log_step("cox",         "Fit Cox PH",            {"penalizer": 0.1},  f"C-index={cox_result.concordance:.3f}" if cox_result else "skipped")
for card in surv_cards:
    trace.log_insight(card)
out_dir = os.path.dirname(r"<dataset_path>") or r"C:\Users\Ravi\research\data-analysis\output"
json_path, md_path = trace.save_manifest(out_dir)
print(f"Trace JSON : {json_path}")
print(f"Trace MD   : {md_path}")
print(trace.render_breadcrumbs())

# Persist to database
if conn_string:
    try:
        from hooks.db_store import get_connection, save_run
        conn = get_connection(conn_string)
        saved_run_id = save_run(
            conn, trace, domain_ctx,
            findings_summary="<one-sentence summary: median survival, top Cox covariate>",
            analysis_type="survival",
            workflows_run=["D"],
            row_count=dataset.n_total,
            col_count=len(df.columns),
            what_if_suggestions=suggestions,
        )
        conn.close()
        print(f"[DB] Run persisted: {saved_run_id}")
    except Exception as e:
        print(f"[DB] Could not persist run: {e}")
```

---

### Step D9 — Self-improvement (hook — always run at end of Workflow D)

```python
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\Ravi\Data_Analysis\.env")
from hooks.self_improver import run_self_improvement, improvement_report

run_meta = {
    "dataset_path":       r"<dataset_path>",
    "query":              "<user_query>",
    "domain_detected":    domain_ctx["domain"],
    "domain_method":      domain_ctx["method"],
    "domain_confidence":  domain_ctx.get("confidence", "?"),
    "domain_correct":     None,
    "domain_actual":      None,
    "analysis_type":      "survival",
    "workflows_run":      ["D"],
    "row_count":          dataset.n_total,
    "col_count":          len(df.columns),
    "column_names":       df.columns.tolist(),
    "issues":             dataset.warnings,
    "findings_summary":   (
        f"Median survival={km_result.median_survival:.2f}; "
        f"event_rate={dataset.n_events/dataset.n_total*100:.1f}%; "
        + (f"best_model={param_result.better_model}; " if param_result else "")
        + (f"C-index={cox_result.concordance:.3f}; sig_covariates={cox_result.significant_covariates}" if cox_result else "")
    ),
}
result = run_self_improvement(run_meta, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(improvement_report(result))
```

---

## Workflow E -- ML Research Orchestrator (4-phase)

Use this workflow when the user asks to "train a model", "predict", "experiment", "run ML", "research", or after Workflow A profiling when the dataset has a suitable prediction target.

**Trigger phrases:** "train a model", "predict", "experiment", "ML", "classification", "regression", "automl", "leaderboard", "research project"

The Research Orchestrator wraps the existing experiment loop with structured methodology:
**Understand -> Design (creative DoE) -> Prioritize -> Execute in loop**

It explores 7 research surfaces: Data, Features, Architectures, Models, Parameters, Validation, Interpretation.

### Step E0 -- Memory check (hook -- run if POSTGRES_URL is set)

Same as Step 0. Retrieve relevant prior ML runs.

```python
# Same pattern as Workflow A Step 0
# Use recall_semantic(conn, query="ml experiment " + dataset_name, limit=5)
```

### Step E1 -- Phase 1: Understand (load, profile, select target, frame objective)

If Workflow A has already run, reuse `df`, `col_classes`, `domain_ctx` from that output.
Otherwise, run Steps 1-1b from Workflow A.

```python
import sys, os; sys.path.insert(0, r"C:\Users\Ravi\research\data-analysis")
from hooks.ml_experiment import select_target

# Auto-detect or use user-specified target
target_info = select_target(df, col_classes, domain_ctx)
if target_info is None:
    print("No suitable prediction target found. Skipping ML workflow.")
else:
    print(f"Target: {target_info['col']} ({target_info['task_type']})")
    print(f"Rationale: {target_info['rationale']}")
```

### Step E2 -- Baseline run (5-model roster via `run_experiment_loop(n_rounds=0)`)

```python
from hooks.ml_experiment import build_feature_matrix, run_experiment_loop

id_cols = col_classes.get("identifiers", [])
X, y, feature_names = build_feature_matrix(df, target_info["col"], id_cols)

baseline_result = run_experiment_loop(
    X, y, feature_names,
    task_type=target_info["task_type"],
    max_trials=8,
    domain_ctx=domain_ctx,
    out_dir=out_dir,
    n_rounds=0,  # Pure roster, no LLM rounds
)
print(f"Baseline: {baseline_result['best_trial']['metrics']['Model']} "
      f"({baseline_result['primary_metric']}={baseline_result['incumbent_metric']:.4f})")
```

### Step E3 -- Phase 2: Design experiments across 7 surfaces

```python
from hooks.research_orchestrator import phase2_design

# LLM designs 8-15 creative experiments if API key available
# Heuristic fallback: template instantiation from surface definitions
experiments = phase2_design(
    project, baseline_result, quality_df, importances,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
```

**7 Research Surfaces:**
| Surface | Vectors |
|---------|---------|
| Data | quality fixes (imputation), quantity augmentation (SMOTE, noise) |
| Features | explicit (ratios, interactions), latent (PCA, clusters), engineering (log, bin) |
| Architectures | ensemble (voting, stacking), mixture of experts |
| Models | CatBoost, ExtraTrees, GradientBoosting, SVM, KNN |
| Parameters | Optuna Bayesian optimization, targeted grid search |
| Validation | K-fold CV, bootstrap CI, overfitting detection |
| Interpretation | SHAP, permutation importance, threshold analysis |

### Step E4 -- Phase 3: Prioritize with stopping threshold

```python
from hooks.research_orchestrator import phase3_prioritize, compute_stopping_threshold

# Compute stopping threshold (60% LLM reasoning + 40% heuristic)
threshold, rationale = compute_stopping_threshold(
    baseline_metrics, task_type, domain_ctx, primary_metric,
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Prioritize: score = (impact + diversity_bonus + gap_bonus) / cost
experiments = phase3_prioritize(experiments, project)
```

### Step E5 -- Phase 4: Execute loop until threshold/budget/plateau

```python
from hooks.research_orchestrator import phase4_execute

exec_result = phase4_execute(
    project, X, y, feature_names, experiments,
    X_train, X_test, y_train, y_test,
    out_dir, anthropic_api_key,
)
```

**Stopping criteria (any triggers stop):**
1. `current_best >= stopping_threshold` -> "threshold_met"
2. Budget exhausted (trials, seconds) -> "budget_exhausted"
3. Last 3 iterations each improved < 0.005 -> "plateau_detected"
4. All planned experiments done -> "all_experiments_done"

The loop repeats Phase 2-4 for up to `max_iterations`, adapting the threshold each iteration.

### Step E6 -- Render + save artifacts

```python
from hooks.research_orchestrator import update_research_project_md, render_research_summary
from hooks.ml_experiment import render_experiment_summary, append_results_tsv

# Living research artifact
md_path = update_research_project_md(project, out_dir)

# Standard experiment summary
summary = render_experiment_summary(result)
research_summary = render_research_summary(project)
print(research_summary)

# Write TSV
tsv_path = append_results_tsv(
    result["trials"], out_dir / "experiments.tsv",
    dataset_name=dataset_name, domain=domain_ctx["domain"],
    target_col=target_info["col"],
)
```

### Step E7 -- Persist to DB (hook -- run if POSTGRES_URL is set)

```python
from hooks.db_store import get_connection, init_db, save_run

conn = get_connection(os.getenv("POSTGRES_URL"))
init_db(conn)
run_id = save_run(
    conn=conn, trace=trace, domain_ctx=domain_ctx,
    findings_summary=f"Research project: best={result['best_trial'].get('metrics', {}).get('Model', 'N/A')}, "
                     f"{result['primary_metric']}={result['incumbent_metric']:.4f}, "
                     f"stop={result.get('trace', {}).get('stop_reason', 'N/A')}",
    analysis_type="ml_experiment",
    workflows_run=["E"],
    row_count=len(df), col_count=len(df.columns),
)
conn.close()
```

### Step E8 -- Self-improvement with research findings

```python
from hooks.self_improver import run_self_improvement, improvement_report

run_meta = {
    "dataset_path": dataset_path,
    "query": f"ML research project on {dataset_name}",
    "domain_detected": domain_ctx["domain"],
    "analysis_type": "ml_experiment",
    "workflows_run": ["E"],
    "row_count": len(df),
    "col_count": len(df.columns),
    "findings_summary": f"Best: {result['best_trial'].get('metrics', {}).get('Model', 'N/A')} "
                        f"({result['primary_metric']}={result['incumbent_metric']:.4f}). "
                        f"{len(result['trials'])} trials across {result.get('trace', {}).get('total_experiments', '?')} experiments. "
                        f"Stopped: {result.get('trace', {}).get('stop_reason', 'N/A')}.",
}
si_result = run_self_improvement(run_meta, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
print(improvement_report(si_result))
```

### Step E9 -- Present research project to user

Show:
1. The research project summary (surfaces explored, improvement trajectory)
2. Best model name, primary metric, and key secondary metrics
3. Stopping reason and threshold
4. Number of experiments across each surface
5. Path to `research_project.md` and `experiments.tsv`
6. Any leakage warnings from feature matrix build

**Simplified invocation (wraps all phases):**

```python
from hooks.research_orchestrator import run_research_project

result = run_research_project(
    df, col_classes, domain_ctx,
    target_col=target_col,       # optional, auto-detected if None
    objective=None,               # auto-framed from domain + target
    max_iterations=5,             # max research iterations
    budget_limit={"max_trials": 30, "max_seconds": 600},
    out_dir=out_dir,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
# result contains all run_experiment_loop keys (backward compat) plus:
# - project: full ResearchProject state
# - research_project_md_path: path to research_project.md
# - research_summary: ASCII-safe summary text
# - trace: {phases, iterations, stop_reason, total_experiments, total_trials, elapsed_s}
```

**Meta-agent integration:** For full meta-agent research with L3 experiment ledger, 4 sub-agents
(RA1–RA4), DESO integration (AG40), and structured context packets, see
`~/.claude/skills/research_orchestrator/SKILL.md`. The Research Orchestrator skill wraps this
same workflow with additional memory persistence, surface registry, and routing from DESO SA3.

---

## Learned Patterns

Auto-updated by `hooks/self_improver.py` after each run. Do not edit manually.

### 2026-03-24
- Asymmetric block comparison: when the user asks to compare two column ranges of different sizes (e.g. E-K has 7 cols, R-AG has 16 cols), compare semantic field pairs (Legal Name vs Mailing Name, City vs Cleansed City) rather than whole-block identity -- whole-block identity will always be False.
- Null cleansed fields (Cleansed City, Cleansed Province, Cleansed Postal Code, etc.) in an MDM dataset indicate the address standardisation pipeline has not yet written back results -- this is a pipeline-incomplete state, not a source data quality error.
- Master Data Management datasets contain two structural blocks: source attributes (raw from ERP/CRM) and MDM-enriched/cleansed attributes. The cleansed block will typically have more columns than the source block.
- Domain keyword "Name" alone does not distinguish HR from MDM -- look for "Cleansed", "MDM ID", "GRM", "Mailing Name", "Golden Record", "Survivorship" to identify MDM datasets.

### 2026-03-24
- Asymmetric block comparison: when the two ranges have different column counts, compare semantic field pairs (e.g. Legal Name vs Mailing Name) rather than whole-block identity.
- Null cleansed fields (Cleansed City, Cleansed Province, etc.) indicate the address standardisation pipeline has not yet written back results -- treat as pipeline-incomplete, not a data quality error in the source.

### 2026-03-26 — Clinical Mortality Prediction (Elara Probus)

**Dataset construction patterns:**
- When building a mortality/event flag from a status-history table, filter on the MOST RECENT status per client (groupby ClientId + idxmax on date), not on any row where the status appears -- a client can have multiple discharged rows.
- Sentinel dates (e.g. 2099-12-31 in diagnosis_end_date) must be converted to NaT before any duration arithmetic -- they will otherwise silently produce wildly large condition_period values that dominate the model.
- One-to-many joins (e.g. client → diagnoses) expand row count -- always aggregate back to one-row-per-client (using groupby first/max/sum) before ML dataset construction.
- When two form versions cover the same questions (e.g. FormID 38 Q_1641-1652 and FormID 142 Q_6339-6350), sum the paired columns and drop the originals -- treating them as separate features doubles cardinality and confuses tree models.
- 76%+ null questionnaire columns mean "not assessed", not "no events" -- fill with 0 for ML but flag this as a potential bias source in the summary.
- WeeklyAuthHrs: bucket to Low/Medium/High/Unknown (Q25/Q90 thresholds + Unknown for null) rather than imputing -- the Unknown category carries clinical meaning (unplanned or informal care).

**ICD-10 extraction:**
- Regex `^([A-Z]\d{2}[\d.A-Z]{0,7})` covers 94-95% of structured codes; remainder need lookup.
- Multi-condition strings (3+ comma-separated diagnoses) cannot reliably map to a single ICD-10 code -- leave as null and flag count separately.
- When Anthropic API credits are depleted mid-pipeline, fall back to a hardcoded ICD_MAP dict with expert codes for the unique free-text descriptions; 292 entries covered the full dataset.

**Imbalanced classification (6% positive rate):**
- class_weight='balanced' on LogisticRegression/RandomForest achieves high recall (0.96-0.99) but produces many false positives -- appropriate when the cost of missing a mortality case is high.
- SMOTE (sampling_strategy=0.3) + RobustScaler improves MLP from near-zero recall to usable, but reduces recall for models that were already performing well with class weights.
- For clinical mortality prediction, prefer class_weight='balanced' over SMOTE for tree/linear models; use SMOTE only for neural nets.
- PR-AUC of 0.24-0.28 at 6% prevalence represents a ~4x lift over baseline (0.061) -- meaningful, but reaching 0.50 requires: (a) Charlson Comorbidity Index from ICD-10 codes, (b) ICD chapter interaction terms, (c) direct PR-AUC optimisation (XGBoost eval_metric='aucpr'), (d) isotonic calibration, (e) trend/slope features from timestamped assessments.

**Recency features from timestamped assessments:**
- Compute per-event-type "days_since_X" using ClientFormSubmittedDatetime grouped by ClientId; fill non-assessed clients with sentinel NEVER=9999.
- Recency-weighted burden = q_score_total / (days_since_last_adverse_event/30 + 1) is more predictive than raw totals for clients with long care histories.
- Rate features (events per 30 days) normalise for care duration and outperform raw counts for clients with very different tenure lengths.

**Model comparison best practices:**
- Always compare ROC-AUC, PR-AUC, F1+ (positive class), Recall, MCC, Specificity together -- no single metric tells the full story for imbalanced clinical data.
- MCC is the most reliable single metric for imbalanced binary classification (accounts for all four confusion matrix cells).
- For clinical use cases where missing a positive case is costly, rank models primarily by Recall, use Specificity as the secondary constraint.

### 2026-03-27 — Readmission Risk Prediction (Elara Probus)

**Label construction from multi-status event tables:**
- When building a readmission/event flag from a status-history table, do NOT filter on a single CLIENTSTATUS value -- clients can have both `discharged` and `on_hold` events for the same hospitalisation episode. Query ALL relevant status values and union them.
- For the Elara Probus dataset: `discharged` rows captured ~1,188 clients; adding `on_hold` rows captured an additional ~2,189 clients and tripled the positive rate from 4.2% to 14.2%.
- Always audit ClientStatusReasonCode variants with `.value_counts()` before writing the filter -- reason codes are often free-text with inconsistent capitalisation (e.g. "Hospital - Hospital", "SNF - Skilled Nursing Facility").
- Use case-insensitive substring matching: `str.contains('hospital|snf|nursing', case=False, na=False)`.

**Temporal leakage detection and avoidance:**
- **Diagnostic signal**: if PR-AUC = 1.0 across ALL models simultaneously, the dataset has a perfectly leaky feature -- this is not a coincidence.
- **Leakage audit procedure**: for each event-derived feature, run `pd.crosstab(df[feature] > 0, df[target])`. If the cell `(feature>0, target=0)` is zero -- every client with a non-zero feature value is a positive -- the feature is perfectly leaky.
- **Root cause**: features derived from the same events table that defines the label will encode label information. Examples: `days_since_last_hosp` (not 9999 ↔ flag=1), `total_hosp_events` (>0 ↔ flag=1).
- **Fix**: exclude all full-history event features from training; retain them in the output file for downstream scoring but add them to `ALWAYS_EXCLUDE` in the AutoML script.
- **Count feature correction**: when an event-count column (e.g. `rc_hospital`) is computed from all events including the qualifying event, subtract 1 from all values: `df['rc_hospital'] = (df['rc_hospital'] - 1).clip(lower=0)`. Verify the fix with a post-correction cross-tab.
- **Structural leakage** (unavoidable in retrospective designs): after correction, clients with 2+ events will still have `prior_count > 0` AND `True_Readmission_Flag = 1` -- this is definitionally true, not a data error. The only fix is a prospective temporal train/test split.

**Prior-event feature engineering (leakage-safe recency features):**
- Identify clients with >1 qualifying event: `multi_ids = hosp[hosp.duplicated('ClientID', keep=False)]['ClientID'].unique()`
- Sort events by date, then exclude the most recent event using `grp.iloc[:-1]` -- this ensures all features are computed from events that occurred *before* the most recent hospitalisation.
- Fill single-event clients (no prior history) with sentinel values: `days_since_prior_hosp = 9999`, LOS features = 0.
- Prior-event features are valid for the broad readmission flag (≥1 event) but structurally leaky for the strict flag (>1 events, True_Readmission_Flag) -- exclude them from the strict-flag model via a separate `extra_exclude` parameter.

**Two-flag modeling strategy for readmission:**
- **Broad flag** (≥1 qualifying event = 1): optimised for population-level screening; prior-event features are valid; higher positive rate (~14%) means more stable model estimates.
- **Strict flag** (>1 qualifying events = 1): optimised for targeted intervention (clients who have already been readmitted); lower positive rate (~2.8%) requires excluding prior-event features to avoid structural leakage.
- Implement separate `ALWAYS_EXCLUDE` and `PRIOR_FEATURES` sets; run two passes in the same AutoML script with `extra_exclude=None` (Pass 1) and `extra_exclude=PRIOR_FEATURES` (Pass 2).
- Expected lift: broad flag ~6× (PR-AUC / prevalence), strict flag ~30× -- both are meaningful given clinical context.

**Label expansion impact on model performance:**
- Adding `on_hold` + Hospital/SNF events alongside `discharged` events: positive rate 4.2% → 14.2%, PR-AUC 0.308 → 0.841 -- single biggest model improvement in the project.
- This pattern generalises: whenever PR-AUC is unexpectedly low, check whether the label definition is incomplete (missing status values, missing reason code variants).

**AutoML script architecture for two-flag leakage-safe runs:**
```python
ALWAYS_EXCLUDE = {
    "ClientId", "Readmission_Flag", "True_Readmission_Flag",
    # Full-history leaky features (excluded from all runs):
    "days_since_last_hosp", "last_hosp_los", "mean_hosp_los", "total_hosp_events",
}
PRIOR_FEATURES = {
    "prior_hosp_count", "days_since_prior_hosp",
    "prior_hosp_los", "mean_prior_hosp_los",
}

def run_pass(df, target, extra_exclude=None):
    exclude = ALWAYS_EXCLUDE | {target} | (extra_exclude or set())
    features = [c for c in df.columns if c not in exclude]
    # ... train models ...

# Pass 1: broad flag -- prior features included
run_pass(df, target="Readmission_Flag", extra_exclude=None)

# Pass 2: strict flag -- prior features excluded (structurally leaky)
run_pass(df, target="True_Readmission_Flag", extra_exclude=PRIOR_FEATURES)
```

### 2026-03-31 — Healthcare Revenue Cycle Management (Invoice & Adjustment Analysis)

**Domain detection:**
- Billing/claims datasets with columns like `payorName`, `InvoiceAdjustmentType`, `ClaimBillStatus`, `invoiceServiceLine`, `adjustmentCategory` are **Healthcare Revenue Cycle Management**, NOT generic Healthcare. Generic Healthcare signals (patient, diagnosis, ICD, CPT) are absent; RCM signals are structural column names not free-text values.
- Added "Healthcare Revenue Cycle Management" domain to `domain_detector.py` with dedicated keywords, profile, and ambiguity resolver (RCM beats Healthcare when RCM score >= Healthcare score).

**Sparse-by-design columns — do not treat as data quality failures:**
- In RCM billing datasets, adjustment columns (`InvoiceAdjustmentAmount`, `InvoiceAdjustmentType`, `InvoiceAdjustmentGroupDesc`, `adjustmentName`, `adjustmentCategory`) are structurally null on non-adjusted lines — 84% null is correct and expected, not a quality defect.
- When `adjustmentCategory` has only 1 unique value (e.g. "ADJ"), this is also expected — it is a type discriminator populated only when an adjustment exists, not a meaningful categorical dimension.
- Pattern: if a cluster of 4-5 related columns all have the same null% (e.g. all exactly 84.27%), they are a sparse attribute group — flag them together as "sparse by design" rather than individually as CRITICAL_NULLS.

**RCM aggregation pattern — always compute these five dimensions:**
1. By `payorName` — absolute write-off $ and adj % of billed (reveals concentration)
2. By `payorTypeName` — adj rate % and adj % of billed (reveals structural contract issues)
3. By `payorFequency` — PERDIEM vs FFS vs PDGM (PERDIEM almost always dominates RCM losses)
4. By `InvoiceAdjustmentType` / `InvoiceAdjustmentGroupDesc` — controllable vs non-controllable split
5. By `invoiceServiceLine` — Home Health vs Hospice vs Behavioral Health (Hospice typically concentrates risk)

**Controllable vs non-controllable classification:**
- `InvoiceAdjustmentGroupDesc` values containing "CONTROLLABLE" but not "NON" = controllable losses (process-fixable)
- "REVENUE OFFSET" = structural contract adjustment (requires renegotiation, not process fix)
- "NON-CONTROLLABLE" = external factors (e.g. Medicaid spenddown, sequestration)
- Always compute and report the controllable $ total — it is the actionable budget for RCM improvement

**Payment speed proxy when no AR timestamp exists:**
- `(invoicepostDate - ServiceDate).dt.days` is the closest available proxy for billing velocity when actual cash receipt date is absent
- Segment by `payorTypeName` and `has_adjustment` boolean to reveal that adjusted invoices post slower
- Highlight payers with median > 50 days as AR risk; payers with median < 10 days as benchmarks

**February spike pattern:**
- In period-end reconciliation datasets, a single month may contain 70-80% of all adjustment activity (batch write-off runs). Always plot the monthly trend before drawing conclusions about steady-state adjustment rates.

**What-if priorities for RCM datasets (added to `what_if.py`):**
- ERA 835 / remittance advice — needed to compute true net collection rate
- Denial timestamps (date denied / appealed / resolved) — needed for days-in-AR and appeal overturn rate
- Prior authorization log — needed to root-cause ELIG/AUTH write-offs
- Eligibility verification log — needed to distinguish payer data error from verification failure
- Claim submission date — needed for true days-in-AR (vs invoicepostDate proxy)

### 2026-04-03
- consider the impact of seasonal trends on sales of consumable goods
- keep a lookout for anomalies in revenue and cost metrics

### 2026-04-03
- Be cautious of seasonal trends in sales data
- Always verify the integrity of revenue and cost calculations

### 2026-04-06
- When analyzing global conflict datasets, pay close attention to correlations between economic loss, military deaths, and climate zone.
- Look for anomalies in refugee numbers and ceasefire outcomes to identify potential areas of concern.

### 2026-04-07
- Verify domain detection confidence when it's low
- Explore column interactions for better insights
