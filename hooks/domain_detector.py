"""
domain_detector.py — Detect the business domain of a DataFrame and scope
                      the most likely solution type within that domain.

Uses Claude for semantic detection (when an API key is available),
with a keyword-heuristic fallback for offline use.

Public API
----------
detect_domain(df, anthropic_api_key, claude_model)          -> dict
scope_solution(df, domain_ctx, user_query)                  -> dict
domain_role_prefix(domain_ctx)                              -> str

Solution registry
-----------------
Domain-solution pairs are defined in hooks/domain_solutions.json.
Edit that file to add or extend solution types without touching this code.
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

# Path to the domain-solution reference registry
_SOLUTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "domain_solutions.json",
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


# ── Solution registry ─────────────────────────────────────────────────────────

def _load_solutions() -> dict[str, list[dict]]:
    """Load domain_solutions.json; return empty dict on missing/corrupt file."""
    if os.path.exists(_SOLUTIONS_PATH):
        try:
            with open(_SOLUTIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Strip the _comment key before returning
            return {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            pass
    return {}


def _heuristic_solution(df: pd.DataFrame, domain: str, user_query: str = "") -> dict:
    """Score solution types for a detected domain against column names + user query.

    Returns the best-matching solution dict from domain_solutions.json,
    or a generic fallback if no triggers match.
    """
    registry = _load_solutions()
    solutions = registry.get(domain, registry.get("General", []))
    if not solutions:
        return _generic_solution(domain)

    col_text = " ".join(df.columns.str.lower().tolist())
    try:
        sample_vals = " ".join(
            df.head(3).astype(str).values.flatten().tolist()
        ).lower()
    except Exception:
        sample_vals = ""
    combined = col_text + " " + sample_vals + " " + user_query.lower()

    best_solution: dict = {}
    best_score: int = -1

    for sol in solutions:
        score = sum(1 for t in sol.get("triggers", []) if t.replace("_", " ") in combined or t in combined)
        if score > best_score:
            best_score = score
            best_solution = sol

    # If nothing matched even the first solution, default to the first entry
    if best_score == 0 and solutions:
        best_solution = solutions[0]

    return best_solution if best_solution else _generic_solution(domain)


def _generic_solution(domain: str) -> dict:
    return {
        "name": "Exploratory Data Analysis",
        "description": f"Profile and surface patterns in this {domain} dataset.",
        "triggers": [],
        "typical_models": ["Descriptive Statistics", "Correlation Analysis", "Anomaly Detection"],
        "key_outputs": ["data_profile", "quality_report", "top_findings"],
    }


def _resolve_solution_by_name(domain: str, name_raw: str) -> Optional[dict]:
    """Find a solution entry by fuzzy name match within a domain (and across all domains).

    Returns None if no match found.
    """
    if not name_raw:
        return None
    registry = _load_solutions()
    name_lower = name_raw.lower()

    # Try domain-local match first, then global
    candidates = registry.get(domain, []) + [
        s for d, sols in registry.items() if d != domain for s in sols
    ]
    for sol in candidates:
        if not isinstance(sol, dict):
            continue
        sol_name = sol.get("name", "").lower()
        if sol_name == name_lower or sol_name in name_lower or name_lower in sol_name:
            return sol
    return None


# ── public API ────────────────────────────────────────────────────────────────

def scope_solution(
    df: pd.DataFrame,
    domain_ctx: dict,
    user_query: str = "",
) -> dict:
    """Determine the most likely solution type for a detected domain.

    Uses trigger-keyword scoring against column names, sample values, and the
    optional user query. Falls back to 'Exploratory Data Analysis' when no
    triggers match.

    Returns a dict with keys:
      solution_name        : str — named solution type (e.g. 'Mortality Prediction')
      solution_description : str — what this solution does
      typical_models       : list[str] — models commonly used
      key_outputs          : list[str] — expected deliverables
    """
    sol = _heuristic_solution(df, domain_ctx.get("domain", "General"), user_query)
    return {
        "solution_name":        sol.get("name", "Exploratory Data Analysis"),
        "solution_description": sol.get("description", ""),
        "typical_models":       sol.get("typical_models", []),
        "key_outputs":          sol.get("key_outputs", []),
    }


def detect_domain(
    df: pd.DataFrame,
    anthropic_api_key: Optional[str] = None,
    claude_model: str = "claude-sonnet-4-6",
    user_query: str = "",
) -> dict:
    """Detect the business domain of a DataFrame and scope the solution type.

    Returns a dict with keys:
      domain               : str — detected domain name
      confidence           : str — 'high' | 'medium' | 'low'
      role                 : str — analyst persona for LLM steps
      summary              : str — one-paragraph description of what this data contains
      analysis_lens        : str — domain-specific analytical focus
      anomaly_guidance     : str — domain-specific anomaly patterns to watch for
      key_metrics          : list[str] — relevant KPIs / metrics for this domain
      method               : str — 'claude' | 'heuristic'
      solution_name        : str — most likely solution type (from domain_solutions.json)
      solution_description : str — what this solution does
      typical_models       : list[str] — models commonly used for this solution
      key_outputs          : list[str] — expected deliverables
    """
    signature = _col_signature(df)

    if anthropic_api_key:
        domain_names = [d for d, _ in _get_domain_keywords()] + ["General"]
        registry = _load_solutions()

        # Build solution options for the detected-domain hint (all domains shown to Claude)
        all_solution_names: list[str] = []
        for sols in registry.values():
            all_solution_names.extend(s["name"] for s in sols if isinstance(s, dict))

        prompt = (
            f"You are a data domain and solution classification expert.\n\n"
            f"Examine the following dataset signature and determine:\n"
            f"1. The business domain\n"
            f"2. The most likely solution type being built with this data\n\n"
            f"{signature}\n\n"
            f"{'User query: ' + user_query if user_query else ''}\n\n"
            f"Choose the single best-fitting domain from:\n"
            f"{', '.join(domain_names)}\n\n"
            f"Choose the single best-fitting solution type from:\n"
            f"{', '.join(all_solution_names)}\n\n"
            f"Then write a one-paragraph plain-English summary of what this dataset appears to contain "
            f"and what business questions it could answer.\n\n"
            f"Reply with exactly this format (nothing else):\n"
            f"DOMAIN: <domain name from list above>\n"
            f"CONFIDENCE: high or medium or low\n"
            f"SOLUTION: <solution type from list above>\n"
            f"SUMMARY: <one paragraph, max 80 words>"
        )
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=anthropic_api_key)
            resp = client.messages.create(
                model=claude_model,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()

            domain = "General"
            confidence = "medium"
            solution_name_raw = ""
            summary = ""

            for line in text.splitlines():
                line = line.strip()
                if line.upper().startswith("DOMAIN:"):
                    raw_domain = line.split(":", 1)[1].strip()
                    for d in _DOMAIN_PROFILES:
                        if d.lower() in raw_domain.lower() or raw_domain.lower() in d.lower():
                            domain = d
                            break
                    else:
                        domain = raw_domain if raw_domain else "General"
                elif line.upper().startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip().lower()
                elif line.upper().startswith("SOLUTION:"):
                    solution_name_raw = line.split(":", 1)[1].strip()
                elif line.upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()

            profile = _get_domain_profiles().get(domain, _DOMAIN_PROFILES["General"])

            # Resolve solution: try to match Claude's answer to a known registry entry
            sol = _resolve_solution_by_name(domain, solution_name_raw) or \
                  _heuristic_solution(df, domain, user_query)

            return {
                "domain":               domain,
                "confidence":           confidence,
                "role":                 profile["role"],
                "summary":              summary or f"A {domain} dataset.",
                "analysis_lens":        profile["analysis_lens"],
                "anomaly_guidance":     profile["anomaly_guidance"],
                "key_metrics":          profile["key_metrics"],
                "method":               "claude",
                "solution_name":        sol.get("name", "Exploratory Data Analysis"),
                "solution_description": sol.get("description", ""),
                "typical_models":       sol.get("typical_models", []),
                "key_outputs":          sol.get("key_outputs", []),
            }

        except Exception:
            pass  # Fall through to heuristic

    # Heuristic fallback
    domain = _heuristic_domain(df)
    profile = _get_domain_profiles().get(domain, _DOMAIN_PROFILES["General"])
    sol = _heuristic_solution(df, domain, user_query)
    return {
        "domain":               domain,
        "confidence":           "medium",
        "role":                 profile["role"],
        "summary":              f"Heuristic detection identified this as a {domain} dataset based on column names and sample values.",
        "analysis_lens":        profile["analysis_lens"],
        "anomaly_guidance":     profile["anomaly_guidance"],
        "key_metrics":          profile["key_metrics"],
        "method":               "heuristic",
        "solution_name":        sol.get("name", "Exploratory Data Analysis"),
        "solution_description": sol.get("description", ""),
        "typical_models":       sol.get("typical_models", []),
        "key_outputs":          sol.get("key_outputs", []),
    }


def domain_role_prefix(domain_ctx: dict) -> str:
    """Return a formatted role-and-context block to prepend to every LLM reasoning step.

    Paste this at the start of your analysis narrative to ground all
    interpretations in the detected domain and solution scope.
    """
    solution_name = domain_ctx.get("solution_name", "")
    solution_desc = domain_ctx.get("solution_description", "")
    typical_models = domain_ctx.get("typical_models", [])
    key_outputs = domain_ctx.get("key_outputs", [])

    lines = [
        f"=== DOMAIN CONTEXT ===",
        f"Domain   : {domain_ctx['domain']}  (detected via {domain_ctx['method']}, confidence: {domain_ctx.get('confidence','?')})",
        f"Role     : You are a {domain_ctx['role']}.",
        f"Dataset  : {domain_ctx['summary']}",
        f"Lens     : {domain_ctx['analysis_lens']}",
        f"Anomalies: {domain_ctx['anomaly_guidance']}",
        f"Key metrics: {', '.join(domain_ctx.get('key_metrics', []))}",
    ]
    if solution_name:
        lines += [
            f"",
            f"=== SOLUTION SCOPE ===",
            f"Solution : {solution_name}",
            f"Purpose  : {solution_desc}",
        ]
        if typical_models:
            lines.append(f"Models   : {', '.join(typical_models)}")
        if key_outputs:
            lines.append(f"Outputs  : {', '.join(key_outputs)}")
    lines.append(f"======================")
    return "\n".join(lines)
