# data-analysis-skill

An end-to-end tabular data analysis harness for Claude Code.
Supports profiling, anomaly detection, sampling comparison, survival/churn analysis, ML classification, and cross-session memory via PostgreSQL.

---

## Structure

```
data-analysis/
├── SKILL.md              # Claude Code skill definition — 4 workflows (A/B/C/D)
├── loader.py             # Universal file loader (CSV, JSON, Excel, Parquet, ODS)
├── profiler.py           # Descriptive profiling
├── classifier.py         # Column type classifier (Identifier/Dimension/Fact/Context)
├── sanitizer.py          # PII redaction
├── visualizer.py         # Chart generation (matplotlib/seaborn)
├── domain_overrides.json # User-maintained domain keyword overrides
├── pending_improvements.md  # Queued improvements from self_improver
├── run_history.jsonl     # Per-run metadata log (appended by self_improver)
├── requirements.txt
└── hooks/
    ├── aggregation.py        # group_agg, multi_agg
    ├── anomaly_detection.py  # detect_anomalies, anomaly_summary
    ├── correlation.py        # correlation_report, top_correlations
    ├── data_quality.py       # quality_report, quality_commentary
    ├── db_store.py           # PostgreSQL persistence (runs, steps, insights, what-if)
    ├── domain_detector.py    # detect_domain, domain_role_prefix
    ├── explainability.py     # InsightCard, render_explanation, explain_*
    ├── frequency.py          # top_n
    ├── missing_values.py     # missing_report
    ├── outliers.py           # outlier_report, get_outlier_rows
    ├── sampling_compare.py   # compare_samples, cardinality_report, sampling_commentary
    ├── self_improver.py      # run_self_improvement, improvement_report
    ├── survival_analysis.py  # Kaplan-Meier, Weibull, LogNormal, Cox PH, predict_batch
    ├── trace_writer.py       # AnalysisTrace — audit manifests (JSON + Markdown)
    └── what_if.py            # suggest_what_if, render_what_if
```

---

## Workflows (defined in SKILL.md)

| Workflow | Trigger | Key steps |
|----------|---------|-----------|
| **A** | Single-file analysis | Memory check → Load → Domain detect → Profile → Quality → Analyse → Explain → Trace → Persist |
| **B** | Multi-file sampling comparison | Diversity scorecard, cardinality/range/null reports, winning sample quality |
| **C** | Anomaly & outlier detection | Statistical (mod-z/IQR/z) + Claude semantic judgment for categoricals |
| **D** | Survival & churn analysis | Kaplan-Meier, Weibull/LogNormal, Cox PH, batch churn scoring |

All workflows include:
- **Step 0** — PostgreSQL memory check (prior runs for same file or domain)
- **Step 8** — Persist run to PostgreSQL
- **Step 9** — Self-improvement (auto-updates `domain_overrides.json` and `SKILL.md`)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env` (never committed)
```
ANTHROPIC_API_KEY=sk-ant-...
POSTGRES_URL=postgresql://user:password@localhost:5432/data_analysis
```

### 3. Initialise the database
```python
from dotenv import load_dotenv
load_dotenv(".env")
from hooks.db_store import get_connection, init_db
conn = get_connection()
init_db(conn)   # idempotent — creates 6 tables + 2 indexes
conn.close()
```

### 4. (Windows) Start local PostgreSQL
If using the portable PostgreSQL install:
```bat
%USERPROFILE%\postgresql17\start-postgres.bat
```
This script is also placed in the Windows Startup folder to auto-start on login.

---

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Optional | Claude-enhanced domain detection, semantic anomaly scoring, what-if suggestions, self-improvement |
| `POSTGRES_URL` | Optional | Cross-session run memory. If unset, all DB steps are silently skipped — no workflow breaks |

---

## PostgreSQL schema

Six tables created by `init_db`:

| Table | Purpose |
|-------|---------|
| `datasets` | SHA-256 fingerprint per unique file |
| `analysis_runs` | One row per run (domain, type, findings summary, domain_ctx JSON) |
| `run_steps` | Step-level audit trail |
| `run_artifacts` | Saved output files |
| `run_insights` | InsightCards from explainability hooks |
| `what_if_suggestions` | Data-gap suggestions per run |

`recall_similar_runs` matches by: exact file hash → same domain → similar filename.

---

## Supported file formats

CSV, Excel (`.xlsx`/`.xls`), JSON, Parquet, ODS

---

## Supported domains (auto-detected)

Healthcare, Pharmaceutical, Industrial Equipment, Electrical Components,
Financial Services / Lending, Insurance, Travel & Hospitality, Retail & E-commerce,
HR & People Analytics, Education, Master Data Management, Healthcare Supply Chain / GPO, General (fallback)

Custom domain keywords can be added to `domain_overrides.json`.
