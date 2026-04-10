"""
what_if.py -- Domain-aware suggestions for data that would improve the analysis.

Public API
----------
suggest_what_if(domain_ctx, df, anthropic_api_key=None) -> list[dict]
render_what_if(suggestions)                              -> str
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import pandas as pd


# ── Domain knowledge base ─────────────────────────────────────────────────────

_DOMAIN_WHAT_IF: dict = {
    "Healthcare Revenue Cycle Management": [
        {
            "missing_data": "ERA 835 / remittance advice data (actual cash received per claim)",
            "current_limitation": "Dataset shows billed amounts and adjustments but not final cash receipts — cannot compute true net collection rate or days-cash-outstanding",
            "what_unlocks": "True net collection rate by payer; cash flow forecasting; underpayment detection (paid less than contractual rate); denial-to-payment reconciliation",
            "data_source": "Clearinghouse ERA feed; practice management system payment posting module",
            "priority": "high",
        },
        {
            "missing_data": "Denial timestamps (date denied, date appealed, date resolved)",
            "current_limitation": "Cannot compute time-to-denial, appeal success rate, or days in AR by denial reason — the single most actionable metric for RCM speed improvement",
            "what_unlocks": "Days-in-AR by payer and denial type; appeal overturn rate; SLA compliance tracking; identification of payers that delay then deny",
            "data_source": "Claims management system denial log; clearinghouse 277CA transaction",
            "priority": "high",
        },
        {
            "missing_data": "Prior authorization tracking (PA obtained Y/N, PA date, auth number)",
            "current_limitation": "PAYOR ENTRY ERROR – ELIG/AUTH is the #1 write-off driver ($263K) but we cannot tell whether auth was obtained, expired, or never requested",
            "what_unlocks": "Root cause split: missing vs expired vs wrong auth; pre-service auth compliance rate; auth-to-denial correlation by payer and service type",
            "data_source": "Utilisation management system; EMR order entry auth field",
            "priority": "high",
        },
        {
            "missing_data": "Eligibility verification log (verified Y/N, verification date, eligibility source)",
            "current_limitation": "Cannot distinguish Elig/Auth write-offs caused by verification failure vs payer data errors vs patient coverage gaps",
            "what_unlocks": "Eligibility verification compliance rate; payer-specific eligibility error patterns; ROI on real-time eligibility tools",
            "data_source": "Patient access / front-end eligibility verification system; 270/271 transaction log",
            "priority": "high",
        },
        {
            "missing_data": "Claim submission date and payer adjudication date",
            "current_limitation": "Only invoicepostDate is available — cannot compute true days-in-AR (submission to payment) or identify payers that hold claims beyond contractual timely payment obligations",
            "what_unlocks": "Days-in-AR by payer; timely payment compliance monitoring; late payment interest calculation; payer SLA benchmarking",
            "data_source": "Clearinghouse submission log; 835 adjudication date field",
            "priority": "medium",
        },
        {
            "missing_data": "Patient financial responsibility (co-pay, deductible, co-insurance amounts)",
            "current_limitation": "Self Pay write-offs ($7,973) cannot be distinguished from primary payer balance transfers vs true self-pay accounts",
            "what_unlocks": "Patient responsibility estimation accuracy; collection agency referral timing; charity care vs bad debt split",
            "data_source": "Eligibility 271 response; explanation of benefits; patient financial counselling notes",
            "priority": "medium",
        },
    ],
    "Healthcare": [
        {
            "missing_data": "Contract and spend data per provider",
            "current_limitation": "Can only count members/encounters; cannot weight by contract value or spend",
            "what_unlocks": "Value-based contracting analytics; network cost modelling; high-value member identification",
            "data_source": "Procurement/contracting system; claims adjudication platform",
            "priority": "high",
        },
        {
            "missing_data": "NPI (National Provider Identifier) cross-reference",
            "current_limitation": "Cannot link providers to CMS public data for specialty, sanctions, or quality scores",
            "what_unlocks": "CMS linkage; provider quality benchmarking; network adequacy checks",
            "data_source": "CMS NPI registry (public); internal credentialing system",
            "priority": "high",
        },
        {
            "missing_data": "Diagnosis codes (ICD-10) or procedure codes (CPT)",
            "current_limitation": "Cannot stratify utilisation by clinical category or disease burden",
            "what_unlocks": "Clinical utilisation analysis; risk stratification; DRG-level benchmarking",
            "data_source": "Claims data; EHR clinical extract",
            "priority": "medium",
        },
    ],
    "Pharmaceutical": [
        {
            "missing_data": "CTCAE adverse event severity grades",
            "current_limitation": "Adverse event counts cannot be risk-stratified by clinical severity",
            "what_unlocks": "Safety signal prioritisation; dose-response modelling; regulatory submission support",
            "data_source": "Clinical trial database; pharmacovigilance system",
            "priority": "high",
        },
        {
            "missing_data": "Comparator arm / control group data",
            "current_limitation": "Cannot compute relative risk reduction or NNT without a comparator",
            "what_unlocks": "Comparative effectiveness analysis; NNT/NNH; regulatory-grade efficacy tables",
            "data_source": "RCT protocol database; synthetic control arm modelling",
            "priority": "high",
        },
        {
            "missing_data": "Batch/lot traceability codes",
            "current_limitation": "Cannot trace product defects or efficacy variations back to manufacturing batch",
            "what_unlocks": "Batch recall scoping; manufacturing root-cause analysis; QA trend monitoring",
            "data_source": "ERP manufacturing module; batch release records",
            "priority": "medium",
        },
    ],
    "Industrial Equipment": [
        {
            "missing_data": "Maintenance event log with timestamps",
            "current_limitation": "Cannot compute MTBF (Mean Time Between Failures) or MTTR without event history",
            "what_unlocks": "Predictive maintenance scheduling; MTBF/MTTR KPIs; warranty claim validation",
            "data_source": "CMMS (Computerised Maintenance Management System); IoT event log",
            "priority": "high",
        },
        {
            "missing_data": "Operating hours / cycle counters",
            "current_limitation": "Failure analysis is calendar-time based; cannot model Weibull failure distributions",
            "what_unlocks": "Weibull reliability modelling; usage-based maintenance; asset utilisation scoring",
            "data_source": "SCADA/DCS historian; equipment onboard meters",
            "priority": "high",
        },
        {
            "missing_data": "Environmental operating conditions (temperature, humidity, load)",
            "current_limitation": "Cannot correlate failures with operating environment stress factors",
            "what_unlocks": "Stress-failure correlation; condition-based maintenance triggers; warranty dispute resolution",
            "data_source": "IoT sensor streams; environmental monitoring systems",
            "priority": "medium",
        },
    ],
    "Electrical Components": [
        {
            "missing_data": "Customer application / end-use environment",
            "current_limitation": "Cannot assess whether spec ratings (voltage, current) are appropriate for customer use case",
            "what_unlocks": "Application-fit scoring; upsell/cross-sell recommendations; field failure root cause",
            "data_source": "CRM application field; customer BOM submissions; design registration system",
            "priority": "high",
        },
        {
            "missing_data": "Cross-reference / alternate part numbers",
            "current_limitation": "Cannot identify functional equivalents or assess substitution risk",
            "what_unlocks": "Competitive cross-reference analysis; supply chain substitution; distributor stocking strategy",
            "data_source": "Product information management (PIM) system; IHS Markit cross-reference database",
            "priority": "high",
        },
        {
            "missing_data": "Lifecycle status with dates (active / EOL / NRND)",
            "current_limitation": "Cannot alert customers on obsolescence risk or plan last-time-buy windows",
            "what_unlocks": "Obsolescence management; LTB planning; customer migration programmes",
            "data_source": "PIM lifecycle management module; manufacturer PCN/EOL notices",
            "priority": "medium",
        },
    ],
    "Financial Services/Lending": [
        {
            "missing_data": "Bureau tradeline history (trended credit attributes)",
            "current_limitation": "Point-in-time bureau snapshot cannot capture behavioural trends (utilisation trajectory, delinquency patterns)",
            "what_unlocks": "Trended credit scoring; early delinquency signals; IFRS 9 lifetime loss modelling",
            "data_source": "Credit bureau trended data feed (Experian, Equifax, TransUnion)",
            "priority": "high",
        },
        {
            "missing_data": "Acquisition channel and campaign ID",
            "current_limitation": "Cannot attribute credit performance to origination channel or marketing cohort",
            "what_unlocks": "Channel-level risk-adjusted ROI; adverse selection by channel; marketing mix modelling",
            "data_source": "CRM origination system; marketing attribution platform",
            "priority": "high",
        },
        {
            "missing_data": "Collateral valuation and LTV at origination",
            "current_limitation": "Cannot compute loss-given-default (LGD) without collateral coverage",
            "what_unlocks": "LGD modelling; stress testing; Basel III capital requirement calculation",
            "data_source": "Appraisal management system; AVM (Automated Valuation Model)",
            "priority": "medium",
        },
    ],
    "Insurance": [
        {
            "missing_data": "Exposure units (policy-years for life/health; payroll for workers comp)",
            "current_limitation": "Claim counts cannot be normalised into loss rates without exposure denominators",
            "what_unlocks": "Loss rate / frequency calculation; actuarial pricing; reinsurance treaty analysis",
            "data_source": "Policy administration system; premium earned calculation engine",
            "priority": "high",
        },
        {
            "missing_data": "Adjuster ID and reserve revision history",
            "current_limitation": "Cannot detect leakage patterns, reserve adequacy issues, or adjuster performance outliers",
            "what_unlocks": "Leakage detection; adjuster benchmarking; IBNR reserve adequacy review",
            "data_source": "Claims management system adjuster activity log",
            "priority": "high",
        },
        {
            "missing_data": "Subrogation and salvage recovery amounts",
            "current_limitation": "Net loss calculations overstate ultimate cost without recovery offsets",
            "what_unlocks": "Net loss modelling; recovery rate benchmarking; subrogation opportunity scoring",
            "data_source": "Claims recovery sub-ledger; salvage auction platform",
            "priority": "medium",
        },
    ],
    "Travel & Hospitality": [
        {
            "missing_data": "Booking lead time (days between booking date and check-in/travel date)",
            "current_limitation": "Cannot build dynamic pricing or demand forecasting models without lead-time signals",
            "what_unlocks": "Dynamic pricing optimisation; demand curve modelling; revenue management",
            "data_source": "PMS (Property Management System); GDS booking feed",
            "priority": "high",
        },
        {
            "missing_data": "Loyalty tier at time of stay / booking",
            "current_limitation": "Cannot segment revenue or satisfaction by loyalty status; cannot measure tier upgrade ROI",
            "what_unlocks": "Loyalty economics (incremental revenue per tier); upgrade propensity model; churn risk by tier",
            "data_source": "Loyalty platform member profile API; PMS loyalty field",
            "priority": "high",
        },
        {
            "missing_data": "Channel of acquisition (OTA, direct, GDS, group)",
            "current_limitation": "Cannot compute channel-adjusted RevPAR or net ADR after commission costs",
            "what_unlocks": "Net RevPAR by channel; direct booking incentive ROI; channel mix optimisation",
            "data_source": "CRS (Central Reservation System) channel code; OTA reconciliation feed",
            "priority": "medium",
        },
    ],
    "Retail & E-commerce (Time Series)": [
        {
            "missing_data": "Promotional calendar and event markers (Black Friday, holiday sales, clearance events)",
            "current_limitation": "Forecast models cannot distinguish organic demand from promotion-driven spikes — seasonal decomposition conflates promotion effects with true seasonality",
            "what_unlocks": "Promotion-adjusted forecasting; change-point attribution; true baseline demand estimation; promo ROI calculation",
            "data_source": "Marketing calendar system; promotional planning tool; POS event flags",
            "priority": "high",
        },
        {
            "missing_data": "External demand signals (weather data, local events, competitor pricing)",
            "current_limitation": "Demand forecast relies on endogenous patterns only — cannot account for weather-driven demand shifts or competitive actions",
            "what_unlocks": "Weather-adjusted demand forecasting; competitive response modelling; external regressor integration into Prophet/ARIMAX",
            "data_source": "Weather API (OpenWeatherMap); local event calendars; competitor price scraping",
            "priority": "medium",
        },
    ],
    "Healthcare (Time Series)": [
        {
            "missing_data": "Patient volume by department and service line with daily/hourly granularity",
            "current_limitation": "Cannot forecast department-level demand or identify capacity bottlenecks with weekly/monthly aggregates",
            "what_unlocks": "Department-level demand forecasting; staffing optimisation; bed capacity planning; ED boarding prediction",
            "data_source": "ADT (Admit-Discharge-Transfer) system; ED tracking board; scheduling system",
            "priority": "high",
        },
        {
            "missing_data": "Staffing levels and shift schedules aligned to patient volume timestamps",
            "current_limitation": "Cannot model staffing demand as a function of predicted patient volume — forecast exists but actionable workforce plan does not",
            "what_unlocks": "Staffing demand forecasting; nurse-to-patient ratio optimisation; overtime cost prediction; agency staff trigger thresholds",
            "data_source": "Workforce management system; nurse scheduling platform; timekeeping system",
            "priority": "high",
        },
    ],
    "General (Time Series)": [
        {
            "missing_data": "Intervention or event markers for change-point attribution",
            "current_limitation": "Observed level shifts or trend changes cannot be attributed to specific causes (policy changes, system updates, external events)",
            "what_unlocks": "Change-point attribution; interrupted time series analysis; intervention impact quantification; before/after comparison",
            "data_source": "Change management log; incident tracking system; policy revision history",
            "priority": "medium",
        },
    ],
    "Retail & E-commerce": [
        {
            "missing_data": "Customer lifetime transaction history",
            "current_limitation": "Cannot compute CLTV or distinguish new vs returning buyers",
            "what_unlocks": "CLTV segmentation; churn prediction; personalisation engine inputs",
            "data_source": "Order management system full history; CDP customer profile",
            "priority": "high",
        },
        {
            "missing_data": "Product margin / cost of goods (COGS)",
            "current_limitation": "Revenue analysis cannot be converted to contribution margin; cannot rank SKUs by profitability",
            "what_unlocks": "Contribution margin ranking; promo ROI; category profitability mix",
            "data_source": "ERP cost accounting module; merchandise planning system",
            "priority": "high",
        },
        {
            "missing_data": "Return reason codes",
            "current_limitation": "Return rates are visible but root causes (sizing, quality, expectation mismatch) are unknown",
            "what_unlocks": "Return root cause analysis; product improvement prioritisation; NPS correlation",
            "data_source": "Returns management system; customer service platform",
            "priority": "medium",
        },
    ],
    "HR & People Analytics": [
        {
            "missing_data": "Manager / reporting hierarchy (org tree)",
            "current_limitation": "Cannot compute span of control, management layers, or team-level aggregations",
            "what_unlocks": "Span of control analysis; organisational design; manager effectiveness benchmarking",
            "data_source": "HRIS org chart feed; Workday/SAP SuccessFactors hierarchy API",
            "priority": "high",
        },
        {
            "missing_data": "Exit interview reason codes",
            "current_limitation": "Attrition counts are visible but the why is unknown; cannot segment by flight risk factor",
            "what_unlocks": "Flight risk factor modelling; targeted retention intervention; manager-level attrition diagnostics",
            "data_source": "Exit interview platform; HRIS termination reason field",
            "priority": "high",
        },
        {
            "missing_data": "Performance rating history (multi-period)",
            "current_limitation": "Single-period snapshot cannot capture performance trajectory or identify high-potential talent",
            "what_unlocks": "High-potential identification; performance trajectory analysis; calibration bias detection",
            "data_source": "Talent management system; performance review platform",
            "priority": "medium",
        },
    ],
    "Education": [
        {
            "missing_data": "Socioeconomic status / free-and-reduced-lunch eligibility",
            "current_limitation": "Outcome analysis cannot be equity-adjusted; disadvantaged student performance is conflated with overall results",
            "what_unlocks": "Equity analysis; SES-adjusted attainment gaps; targeted intervention prioritisation",
            "data_source": "Student information system SES/FRL flag; NCES demographic data",
            "priority": "high",
        },
        {
            "missing_data": "Prior attainment scores (baseline assessment)",
            "current_limitation": "Cannot compute value-added (student growth) without a prior-year baseline",
            "what_unlocks": "Value-added modelling (VAM); teacher/school effectiveness; growth vs attainment distinction",
            "data_source": "Student information system historical assessment records; state longitudinal data system",
            "priority": "high",
        },
        {
            "missing_data": "Attendance and engagement metrics",
            "current_limitation": "Cannot model the attendance-attainment relationship or identify chronic absenteeism risk",
            "what_unlocks": "Chronic absenteeism early warning; attendance-outcome correlation; intervention targeting",
            "data_source": "Student information system daily attendance feed",
            "priority": "medium",
        },
    ],
    "Master Data Management": [
        {
            "missing_data": "Source system provenance metadata (system ID, last-updated timestamp)",
            "current_limitation": "Cannot design survivorship rules without knowing which source system is most authoritative",
            "what_unlocks": "Survivorship rule design; source system trust scoring; golden record lineage",
            "data_source": "MDM hub source system registry; ETL metadata store",
            "priority": "high",
        },
        {
            "missing_data": "Match / merge audit log (candidate pairs and decision outcomes)",
            "current_limitation": "Cannot evaluate false positive / false negative rates of the entity resolution algorithm",
            "what_unlocks": "Precision/recall evaluation; threshold tuning; reviewer productivity analysis",
            "data_source": "MDM match engine decision log; stewardship workflow system",
            "priority": "high",
        },
        {
            "missing_data": "Data stewardship override history",
            "current_limitation": "Cannot distinguish algorithmically-set vs human-corrected attribute values",
            "what_unlocks": "Algorithm accuracy measurement; steward effort tracking; training data curation",
            "data_source": "MDM stewardship portal audit trail",
            "priority": "medium",
        },
    ],
    "Healthcare Supply Chain / GPO": [
        {
            "missing_data": "Contract spend / purchasing volume per member",
            "current_limitation": "Can only count member organisations; cannot weight churn risk or retention value by purchasing activity",
            "what_unlocks": "Value-at-risk churn scoring; high-value member retention prioritisation; contract penetration rate by segment",
            "data_source": "GPO contract management system; distributor purchase order feed; ERP purchasing module",
            "priority": "high",
        },
        {
            "missing_data": "Member engagement events (portal logins, contract utilisation, support interactions)",
            "current_limitation": "Churn model uses only static attributes (class of trade, org status); cannot capture declining engagement signals before formal termination",
            "what_unlocks": "Early-warning churn signals; engagement score by member; proactive outreach triggers (e.g. member has not logged in for 90 days)",
            "data_source": "Member portal clickstream; CRM activity log; contract utilisation platform",
            "priority": "high",
        },
        {
            "missing_data": "Product category participation breadth (number of contract categories used)",
            "current_limitation": "Cannot distinguish deeply-engaged multi-category members from single-category members with similar tenure",
            "what_unlocks": "Category penetration analysis; upsell opportunity scoring; member health index; churn risk by engagement depth",
            "data_source": "GPO contract catalogue system; purchasing category hierarchy",
            "priority": "medium",
        },
    ],
    "General": [
        {
            "missing_data": "Timestamps (created_at, updated_at, event_date)",
            "current_limitation": "Cannot perform trend analysis, seasonality detection, or time-series forecasting",
            "what_unlocks": "Trend analysis; seasonality modelling; cohort analysis; time-to-event metrics",
            "data_source": "Source transaction system; event log",
            "priority": "high",
        },
        {
            "missing_data": "Unique entity identifier (customer ID, product ID, record ID)",
            "current_limitation": "Cannot deduplicate records, perform joins, or track entity-level history",
            "what_unlocks": "Deduplication; entity linkage; longitudinal tracking; join to enrichment sources",
            "data_source": "Source system primary key; MDM golden record ID",
            "priority": "high",
        },
    ],
}

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ── Public API ────────────────────────────────────────────────────────────────

def suggest_what_if(
    domain_ctx: dict,
    df: pd.DataFrame,
    anthropic_api_key: Optional[str] = None,
) -> List[dict]:
    """Return domain-aware suggestions for data that would improve the analysis.

    Parameters
    ----------
    domain_ctx         : domain context dict from domain_detector
    df                 : the analysed DataFrame (used for schema heuristics)
    anthropic_api_key  : if provided, call Claude for 2-3 extra novel suggestions

    Returns
    -------
    list[dict] sorted high -> medium -> low priority
    """
    domain = domain_ctx.get("domain", "General")

    # 1. Domain-specific entries (fallback to General if unknown)
    base = _DOMAIN_WHAT_IF.get(domain, _DOMAIN_WHAT_IF["General"])
    suggestions: List[dict] = [dict(s) for s in base]

    # 2. Always append General entries (skip if domain IS General to avoid duplication)
    if domain != "General":
        suggestions.extend(dict(s) for s in _DOMAIN_WHAT_IF["General"])

    # 3. Schema-driven heuristics
    existing_missing_lower = {s["missing_data"].lower() for s in suggestions}
    col_names_lower = " ".join(c.lower() for c in df.columns)

    # No datetime column and no date-like column name
    has_date = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns) or any(
        kw in col_names_lower for kw in ("date", "time", " dt", "timestamp")
    )
    if not has_date:
        ts_suggestion = {
            "missing_data": "Date / timestamp column",
            "current_limitation": "Cannot perform trend analysis or time-series modelling",
            "what_unlocks": "Trend analysis; seasonality detection; cohort analysis",
            "data_source": "Source transaction system; event log",
            "priority": "high",
        }
        # Deduplicate: skip if any existing entry already covers timestamps/dates
        if not any("timestamp" in e or "date" in e for e in existing_missing_lower):
            suggestions.append(ts_suggestion)

    # No numeric columns
    has_numeric = any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    if not has_numeric:
        num_suggestion = {
            "missing_data": "Quantitative measure columns (counts, amounts, scores)",
            "current_limitation": "Dataset is entirely categorical -- no aggregations or statistical modelling possible",
            "what_unlocks": "Aggregation; statistical testing; predictive modelling",
            "data_source": "Transaction system; scoring model output",
            "priority": "high",
        }
        if not any("quantitative" in e or "measure" in e for e in existing_missing_lower):
            suggestions.append(num_suggestion)

    # High average null rate
    if len(df) > 0:
        avg_null_rate = float(df.isnull().mean().mean()) * 100
        if avg_null_rate > 30:
            completeness_suggestion = {
                "missing_data": "Complete / validated data for high-null columns",
                "current_limitation": (
                    f"Average null rate is {avg_null_rate:.1f}% -- "
                    "analysis results may be unreliable"
                ),
                "what_unlocks": "More reliable aggregations; better model training data; reduced imputation bias",
                "data_source": "Source system data quality remediation; upstream ETL fix",
                "priority": "medium",
            }
            if not any("completeness" in e or "null" in e for e in existing_missing_lower):
                suggestions.append(completeness_suggestion)

    # No spend/volume/amount column
    has_spend = any(
        kw in col_names_lower for kw in ("spend", "amount", "revenue", "volume", "sales", "cost", "price", "value")
    )
    if not has_spend:
        spend_suggestion = {
            "missing_data": "Transaction value / spend amount column",
            "current_limitation": "Analysis is count-based only; cannot weight entities by economic value or prioritise by revenue at risk",
            "what_unlocks": "Value-weighted analysis; revenue-at-risk quantification; high-value segment identification",
            "data_source": "Transaction system; ERP sales module; billing platform",
            "priority": "high",
        }
        if not any("spend" in e or "amount" in e or "value" in e for e in existing_missing_lower):
            suggestions.append(spend_suggestion)

    # No engagement/activity/event column
    has_engagement = any(
        kw in col_names_lower for kw in ("login", "visit", "click", "event", "activity", "interaction", "session", "engagement")
    )
    if not has_engagement and domain not in ("General",):
        engage_suggestion = {
            "missing_data": "Behavioural engagement / activity log",
            "current_limitation": "Model uses only static attributes; cannot capture early-warning signals from declining activity",
            "what_unlocks": "Engagement score; early churn / lapse detection; proactive outreach triggers",
            "data_source": "Platform event log; CRM activity history; web analytics system",
            "priority": "medium",
        }
        if not any("engagement" in e or "activity" in e or "behavioural" in e for e in existing_missing_lower):
            suggestions.append(engage_suggestion)

    # 4 (optional). Claude-enhanced suggestions
    from hooks.llm_client import call_llm
    col_sample = ", ".join(list(df.columns)[:30])
    llm_prompt = (
        f"You are a data strategy advisor. A {domain} dataset has been analysed.\n"
        f"Columns: {col_sample}\n"
        f"Suggest 2-3 additional data sources that would most improve this analysis.\n"
        f"Return only a JSON array of objects, each with keys: "
        f"missing_data, current_limitation, what_unlocks, data_source, priority "
        f"(priority must be 'high', 'medium', or 'low').\n"
        f"No commentary, no code fences -- raw JSON array only."
    )
    text = call_llm(llm_prompt, max_tokens=600, expect_json=True)
    if text:
        try:
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            extra = json.loads(text)
            if isinstance(extra, list):
                suggestions.extend(extra)
        except Exception:
            pass  # Silently fall back on any error

    # 5. Sort: high -> medium -> low
    suggestions.sort(key=lambda x: _PRIORITY_ORDER.get(str(x.get("priority", "low")).lower(), 2))

    return suggestions


def render_what_if(suggestions: List[dict]) -> str:
    """Render what-if suggestions as ASCII plain-text."""
    if not suggestions:
        lines = [
            "=== WHAT-IF ANALYSIS ===",
            "No suggestions available.",
            "========================",
        ]
        return "\n".join(lines)

    high = [s for s in suggestions if str(s.get("priority", "")).lower() == "high"]
    medium = [s for s in suggestions if str(s.get("priority", "")).lower() == "medium"]
    low = [s for s in suggestions if str(s.get("priority", "")).lower() == "low"]

    lines = [
        "=== WHAT-IF ANALYSIS ===",
        "Additional data that would make this analysis more powerful:",
        "",
    ]

    counter = 1
    sep = "-" * 56
    for group_label, group in [
        ("HIGH PRIORITY", high),
        ("MEDIUM PRIORITY", medium),
        ("LOW PRIORITY", low),
    ]:
        if not group:
            continue
        lines.append(f"[{group_label}]")
        for s in group:
            lines.append(f"  {counter}. Missing data  : {s.get('missing_data', '')}")
            lines.append(f"     Limitation   : {s.get('current_limitation', '')}")
            lines.append(f"     Unlocks      : {s.get('what_unlocks', '')}")
            lines.append(f"     Data source  : {s.get('data_source', '')}")
            lines.append(f"     {sep}")
            counter += 1
        lines.append("")

    lines.append("========================")
    lines.append(
        f"Total: {len(suggestions)} suggestions "
        f"({len(high)} high / {len(medium)} medium / {len(low)} low priority)"
    )

    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
