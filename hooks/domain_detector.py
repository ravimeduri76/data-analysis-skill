"""
domain_detector.py — Detect the business domain of a DataFrame.

Uses Claude for semantic domain detection (when an API key is available),
with a keyword-heuristic fallback for offline use.

Public API
----------
detect_domain(df, anthropic_api_key, claude_model)  -> dict
domain_role_prefix(domain_ctx)                       -> str   (prepend to every LLM step)
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import pandas as pd

# Path to the auto-updated domain overrides file (written by self_improver.py)
_OVERRIDES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "domain_overrides.json",
)


# ── Known domains and their keyword signals ──────────────────────────────────

_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Healthcare Supply Chain / GPO", [
        "gpo", "group purchasing", "healthtrust", "premier", "vizient", "provista",
        "class of trade", "membership", "member status", "member id", "facility",
        "roster", "contract tier", "purchasing", "supplier", "distributor",
        "acute care", "ambulatory", "home care", "long term care", "pharmacy",
        "membership start", "membership termination", "active member",
    ]),
    ("Healthcare", [
        "patient", "diagnosis", "icd", "cpt", "admission", "discharge", "clinical",
        "medication", "dosage", "ehr", "emr", "procedure", "symptom", "encounter",
        "provider", "payer", "claim", "readmission", "mortality", "lab", "vitals",
    ]),
    ("Pharmaceutical", [
        "drug", "molecule", "trial", "phase", "fda", "adverse", "efficacy", "dose",
        "formulation", "excipient", "bioavailability", "pharmacokinetics", "smiles",
        "compound", "assay", "indication", "contraindication", "generic", "brand",
    ]),
    ("Industrial Equipment", [
        "voltage", "current", "ampere", "watt", "torque", "rpm", "bearing", "motor",
        "actuator", "sensor", "plc", "scada", "pump", "valve", "compressor",
        "terminal", "connector", "relay", "fuse", "circuit", "din", "ip rating",
        "mounting", "enclosure", "cable", "wire", "gauge",
    ]),
    ("Electrical Components", [
        "voltage_rating", "current_rating", "ip_rating", "din_rail", "terminal_block",
        "connector", "plug", "socket", "pcb", "smd", "resistor", "capacitor",
        "inductance", "impedance", "frequency", "poles", "contact", "crimp", "screw",
    ]),
    ("Financial Services / Lending", [
        "loan", "credit", "interest", "apr", "fico", "ltv", "dti", "delinquency",
        "default", "principal", "amortization", "underwriting", "origination",
        "collateral", "mortgage", "refinance", "borrower", "lender", "grade",
    ]),
    ("Insurance", [
        "policy", "premium", "claim", "deductible", "coverage", "underwriting",
        "actuary", "loss_ratio", "exposure", "peril", "indemnity", "beneficiary",
        "insured", "reinsurance", "endorsement", "adjuster",
    ]),
    ("Travel & Hospitality", [
        "booking", "reservation", "itinerary", "flight", "hotel", "check_in",
        "check_out", "passenger", "destination", "origin", "fare", "airline",
        "seat", "cabin", "loyalty", "cancellation", "refund", "delay",
    ]),
    ("Retail & E-commerce", [
        "sku", "product", "category", "price", "discount", "inventory", "stock",
        "order", "cart", "checkout", "customer", "return", "review", "rating",
        "supplier", "warehouse", "shipment", "delivery",
    ]),
    ("HR & People Analytics", [
        "employee", "headcount", "salary", "tenure", "department", "attrition",
        "hire", "performance", "review", "compensation", "benefits", "leave",
        "absence", "job_level", "band", "promotion", "diversity",
    ]),
    ("Education", [
        "student", "grade", "course", "enrollment", "attendance", "score", "exam",
        "teacher", "school", "district", "gpa", "module", "assignment", "cohort",
    ]),
]

# Domain-specific role descriptions and analytical lenses
_DOMAIN_PROFILES: dict[str, dict] = {
    "Healthcare Supply Chain / GPO": {
        "role": "healthcare supply chain analyst with expertise in GPO membership, contract analytics, and provider network performance",
        "analysis_lens": "Focus on member retention and churn risk, contract penetration rates, purchasing compliance by class of trade, facility segmentation, and network growth/attrition trends.",
        "anomaly_guidance": "Flag members with missing Class of Trade, duplicate member IDs, impossible membership dates (start after termination), active members with termination dates, and facilities with no purchasing activity.",
        "key_metrics": ["member retention rate", "contract penetration %", "purchasing compliance rate", "churn probability by segment", "active member count by class of trade", "member tenure distribution"],
    },
    "Healthcare": {
        "role": "healthcare data analyst with expertise in clinical operations, patient outcomes, and medical coding",
        "analysis_lens": "Focus on patient safety signals, readmission risk, care pathway efficiency, and compliance with clinical coding standards (ICD/CPT).",
        "anomaly_guidance": "Anomalies in healthcare data include impossible clinical values (negative ages, future dates), out-of-range vitals, duplicate encounters, and inconsistent diagnosis/procedure code combinations.",
        "key_metrics": ["readmission rate", "length of stay", "mortality rate", "claim denial rate", "medication error rate"],
    },
    "Pharmaceutical": {
        "role": "pharmaceutical data scientist with expertise in drug development, clinical trials, and regulatory data",
        "analysis_lens": "Focus on compound efficacy signals, adverse event frequencies, trial phase outcomes, and regulatory compliance data quality.",
        "anomaly_guidance": "Flag impossible dose values, adverse event severity mismatches, trial protocol deviations, and molecular property outliers.",
        "key_metrics": ["efficacy rate", "adverse event incidence", "trial dropout rate", "bioavailability", "IC50"],
    },
    "Industrial Equipment": {
        "role": "industrial equipment data analyst specialising in specifications, maintenance, and asset performance",
        "analysis_lens": "Focus on specification completeness, out-of-range operating parameters, maintenance patterns, and equipment failure signals.",
        "anomaly_guidance": "Anomalies include impossible operating voltages/currents, incorrect mounting specifications, non-standard IP ratings, pipe-delimited multi-values indicating data migration errors, and range strings where single values are expected.",
        "key_metrics": ["MTBF", "MTTR", "OEE", "specification completeness %", "out-of-spec rate"],
    },
    "Electrical Components": {
        "role": "electrical components product data analyst with expertise in connector, terminal block, and passive component specifications",
        "analysis_lens": "Focus on specification integrity (voltage/current ratings, mounting types, IP protection), catalogue data quality, and attribute standardisation across product families.",
        "anomaly_guidance": "Flag voltages above 1000V for standard connectors, pipe-delimited multi-values in single-value fields, non-standard mounting type labels, range strings where a single numeric value is expected, and float-integer duplicates (e.g. 12 vs 12.0).",
        "key_metrics": ["catalogue completeness %", "anomaly rate by attribute", "unique value cardinality", "missing attribute %"],
    },
    "Financial Services / Lending": {
        "role": "lending and credit risk analyst with expertise in underwriting, portfolio quality, and regulatory reporting",
        "analysis_lens": "Focus on credit quality distributions, delinquency trends, concentration risks, and model fairness signals.",
        "anomaly_guidance": "Flag impossible FICO scores (outside 300-850), negative loan amounts, DTI > 100%, future origination dates, and inconsistent loan status transitions.",
        "key_metrics": ["default rate", "delinquency rate", "average FICO", "LTV ratio", "DTI ratio", "charge-off rate"],
    },
    "Insurance": {
        "role": "insurance analytics specialist with expertise in underwriting, claims management, and loss modelling",
        "analysis_lens": "Focus on loss ratio trends, claim severity distributions, underwriting concentration, and fraud signals.",
        "anomaly_guidance": "Flag claims exceeding policy limits, negative premiums, future loss dates, duplicate claim IDs, and extreme loss ratios (>200%).",
        "key_metrics": ["loss ratio", "combined ratio", "claim frequency", "average severity", "premium written"],
    },
    "Travel & Hospitality": {
        "role": "travel and hospitality data analyst with expertise in booking behaviour, revenue management, and customer experience",
        "analysis_lens": "Focus on booking lead times, cancellation patterns, fare class distributions, and customer loyalty trends.",
        "anomaly_guidance": "Flag check-out before check-in, negative fares, booking dates after travel dates, impossible seat/cabin combinations, and duplicate booking IDs.",
        "key_metrics": ["occupancy rate", "RevPAR", "ADR", "cancellation rate", "NPS", "load factor"],
    },
    "Retail & E-commerce": {
        "role": "retail and e-commerce data analyst with expertise in merchandising, inventory, and customer behaviour",
        "analysis_lens": "Focus on sales velocity, stockout patterns, return rates, price anomalies, and customer lifetime value.",
        "anomaly_guidance": "Flag negative prices, quantities > stock levels, future order dates, returns exceeding original purchase quantity, and extreme discount rates (>100%).",
        "key_metrics": ["conversion rate", "AOV", "return rate", "inventory turnover", "sell-through %", "CLTV"],
    },
    "HR & People Analytics": {
        "role": "HR and people analytics specialist with expertise in workforce planning, compensation analysis, and talent management",
        "analysis_lens": "Focus on attrition signals, compensation equity, headcount trends, and performance distribution.",
        "anomaly_guidance": "Flag negative salaries, tenure < 0, performance scores outside valid range, duplicate employee IDs, and future hire dates.",
        "key_metrics": ["attrition rate", "time-to-fill", "span of control", "gender pay gap", "engagement score"],
    },
    "Education": {
        "role": "education data analyst with expertise in student outcomes, curriculum effectiveness, and institutional performance",
        "analysis_lens": "Focus on grade distributions, attendance-outcome correlations, at-risk student signals, and course completion rates.",
        "anomaly_guidance": "Flag grades outside valid range, negative attendance counts, future exam dates, duplicate student IDs, and impossible cohort/year combinations.",
        "key_metrics": ["pass rate", "dropout rate", "attendance rate", "average GPA", "course completion rate"],
    },
    "General": {
        "role": "data analyst with broad expertise in tabular data quality, statistical analysis, and business intelligence",
        "analysis_lens": "Focus on data completeness, statistical distributions, anomalous values, and actionable patterns relevant to the dataset's apparent purpose.",
        "anomaly_guidance": "Flag statistical outliers, impossible values, format inconsistencies, and duplicate records.",
        "key_metrics": ["null rate", "duplicate rate", "outlier rate", "data quality score"],
    },
}


# ── Domain override helpers (populated by self_improver.py) ──────────────────

def _load_overrides() -> dict:
    """Load domain_overrides.json; return empty structure on missing/corrupt file."""
    if os.path.exists(_OVERRIDES_PATH):
        try:
            with open(_OVERRIDES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"extra_keywords": {}, "extra_profiles": {}}


def _get_domain_keywords() -> list[tuple[str, list[str]]]:
    """Return _DOMAIN_KEYWORDS merged with any overrides from domain_overrides.json."""
    overrides  = _load_overrides()
    extra_kws  = overrides.get("extra_keywords", {})
    result     = [list(entry) for entry in _DOMAIN_KEYWORDS]   # mutable copy
    known      = {d for d, _ in result}

    # Extend keywords for existing domains
    for i, (domain, kws) in enumerate(result):
        if domain in extra_kws:
            extras = [k for k in extra_kws[domain] if k not in kws]
            result[i] = [domain, kws + extras]

    # Add brand-new domains from overrides
    for domain, kws in extra_kws.items():
        if domain not in known:
            result.append([domain, kws])

    return [(d, kws) for d, kws in result]


def _get_domain_profiles() -> dict[str, dict]:
    """Return _DOMAIN_PROFILES merged with any extra profiles from domain_overrides.json."""
    overrides = _load_overrides()
    extra_profiles = overrides.get("extra_profiles", {})
    return {**_DOMAIN_PROFILES, **extra_profiles}


# ── helpers ───────────────────────────────────────────────────────────────────

def _col_signature(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """Build a compact text signature of column names + sample values."""
    lines = [f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist()[:60])}"]
    sample = df.dropna(how="all").head(max_sample_rows)
    if not sample.empty:
        lines.append("Sample rows (first few non-null):")
        for _, row in sample.iterrows():
            trimmed = {k: str(v)[:60] for k, v in row.items() if pd.notna(v)}
            if trimmed:
                lines.append("  " + str(trimmed)[:200])
    return "\n".join(lines)


def _heuristic_domain(df: pd.DataFrame) -> str:
    """Keyword-based domain detection — no API required. Includes override domains."""
    col_text = " ".join(df.columns.str.lower().tolist())
    try:
        sample_vals = " ".join(
            df.head(3).astype(str).values.flatten().tolist()
        ).lower()
    except Exception:
        sample_vals = ""

    combined = col_text + " " + sample_vals

    scores: dict[str, int] = {}
    for domain, keywords in _get_domain_keywords():
        score = sum(1 for kw in keywords if kw.replace("_", " ") in combined or kw in combined)
        if score:
            scores[domain] = score

    if not scores:
        return "General"
    best = max(scores, key=lambda d: scores[d])
    # Resolve ambiguity: GPO/supply chain beats generic Healthcare
    if best == "Healthcare" and scores.get("Healthcare Supply Chain / GPO", 0) >= scores["Healthcare"]:
        best = "Healthcare Supply Chain / GPO"
    # Resolve ambiguity between Industrial Equipment and Electrical Components
    if best == "Industrial Equipment" and scores.get("Electrical Components", 0) >= scores["Industrial Equipment"]:
        best = "Electrical Components"
    return best


# ── public API ────────────────────────────────────────────────────────────────

def detect_domain(
    df: pd.DataFrame,
    anthropic_api_key: Optional[str] = None,
    claude_model: str = "claude-sonnet-4-6",
) -> dict:
    """Detect the business domain of a DataFrame.

    Returns a dict with keys:
      domain        : str — detected domain name
      confidence    : str — 'high' | 'medium' | 'low'
      role          : str — analyst persona for LLM steps
      summary       : str — one-paragraph description of what this data appears to contain
      analysis_lens : str — domain-specific analytical focus
      anomaly_guidance : str — domain-specific anomaly patterns to watch for
      key_metrics   : list[str] — relevant KPIs / metrics for this domain
      method        : str — 'claude' | 'heuristic'
    """
    signature = _col_signature(df)

    if anthropic_api_key:
        domain_names = [d for d, _ in _get_domain_keywords()] + ["General"]
        prompt = (
            f"You are a data domain classification expert.\n\n"
            f"Examine the following dataset signature and determine its business domain.\n\n"
            f"{signature}\n\n"
            f"Choose the single best-fitting domain from this list:\n"
            f"{', '.join(domain_names)}\n\n"
            f"Then write a one-paragraph plain-English summary of what this dataset appears to contain "
            f"and what business questions it could answer.\n\n"
            f"Reply with exactly this format (nothing else):\n"
            f"DOMAIN: <domain name from list above>\n"
            f"CONFIDENCE: high or medium or low\n"
            f"SUMMARY: <one paragraph, max 80 words>"
        )
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=anthropic_api_key)
            resp = client.messages.create(
                model=claude_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()

            domain = "General"
            confidence = "medium"
            summary = ""

            for line in text.splitlines():
                line = line.strip()
                if line.upper().startswith("DOMAIN:"):
                    raw_domain = line.split(":", 1)[1].strip()
                    # match to known domain
                    for d in _DOMAIN_PROFILES:
                        if d.lower() in raw_domain.lower() or raw_domain.lower() in d.lower():
                            domain = d
                            break
                    else:
                        domain = raw_domain if raw_domain else "General"
                elif line.upper().startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip().lower()
                elif line.upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()

            profile = _get_domain_profiles().get(domain, _DOMAIN_PROFILES["General"])
            return {
                "domain":            domain,
                "confidence":        confidence,
                "role":              profile["role"],
                "summary":           summary or f"A {domain} dataset.",
                "analysis_lens":     profile["analysis_lens"],
                "anomaly_guidance":  profile["anomaly_guidance"],
                "key_metrics":       profile["key_metrics"],
                "method":            "claude",
            }

        except Exception as e:
            # Fall through to heuristic
            pass

    # Heuristic fallback
    domain = _heuristic_domain(df)
    profile = _get_domain_profiles().get(domain, _DOMAIN_PROFILES["General"])
    return {
        "domain":            domain,
        "confidence":        "medium",
        "role":              profile["role"],
        "summary":           f"Heuristic detection identified this as a {domain} dataset based on column names and sample values.",
        "analysis_lens":     profile["analysis_lens"],
        "anomaly_guidance":  profile["anomaly_guidance"],
        "key_metrics":       profile["key_metrics"],
        "method":            "heuristic",
    }


def domain_role_prefix(domain_ctx: dict) -> str:
    """Return a formatted role-and-context block to prepend to every LLM reasoning step.

    Paste this at the start of your analysis narrative to ground all
    interpretations in the detected domain.
    """
    lines = [
        f"=== DOMAIN CONTEXT ===",
        f"Domain   : {domain_ctx['domain']}  (detected via {domain_ctx['method']}, confidence: {domain_ctx.get('confidence','?')})",
        f"Role     : You are a {domain_ctx['role']}.",
        f"Dataset  : {domain_ctx['summary']}",
        f"Lens     : {domain_ctx['analysis_lens']}",
        f"Anomalies: {domain_ctx['anomaly_guidance']}",
        f"Key metrics for this domain: {', '.join(domain_ctx.get('key_metrics', []))}",
        f"======================",
    ]
    return "\n".join(lines)
