"""
domain_detector.py — Detect the business domain of a DataFrame and scope
                      the most likely solution type and lifecycle stage.

Uses Claude for semantic detection (when an API key is available),
with a keyword-heuristic fallback for offline use.

Public API
----------
detect_domain(df, anthropic_api_key, claude_model, user_query)  -> dict
scope_solution(df, domain_ctx, user_query)                      -> dict
domain_role_prefix(domain_ctx)                                  -> str

Solution registry
-----------------
Domain-solution pairs are defined in hooks/domain_solutions.json.
Each solution entry carries a lifecycle_stage field:
  sourcing | extraction | enrichment | preparation | publishing |
  syndication | analytics | ml_modeling | survival_analysis
Edit that file to add or extend solution types without touching this code.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

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
    ("Healthcare Revenue Cycle Management", [
        "invoiceadjustment", "invoicelineitem", "claimbillstatus", "invoiceserviceline",
        "payorname", "payorid", "payortypename", "payorfrequency", "payorfrequency",
        "adjustmenttype", "adjustmentgroupdesc", "adjustmentcategory", "adjustmentname",
        "invoiceno", "invoicestatus", "invoicepostdate", "invoicelineitemrate",
        "denial", "denialcode", "remittance", "era", "eob", "835", "277",
        "prior authorization", "preauth", "eligibility", "elig", "auth",
        "clean claim", "writeoff", "write-off", "contractual adjustment",
        "revenue offset", "timely filing", "undo transfer balance",
        "payor entry error", "hipps", "pdgm", "perdiem", "per diem",
        "home health", "hospice", "behavioral health",
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
    ("Automotive Aftermarket / Heavy Truck", [
        "pies", "aces", "autocare", "aftermarket", "oe part", "oem part",
        "cross reference", "cross-reference", "xref", "interchange",
        "heavy duty", "heavy-duty", "truck parts", "truck-lite", "tectran",
        "bendix", "meritor", "eaton", "atro", "dorman", "gates", "dayco",
        "fleetpride", "finditparts", "truckpro", "napa", "o'reilly",
        "part number", "vendor part", "sku", "upc", "gtin", "ean",
        "category l1", "category l2", "category l3",
        "brake", "air dryer", "lighting", "suspension", "exhaust",
        "transmission", "clutch", "wheel seal", "hub", "drum", "rotor",
        "air line", "gladhand", "air fitting", "valve", "compressor",
    ]),
    ("Mechanical Components", [
        "fastener", "bolt", "screw", "nut", "washer", "rivet", "pin", "anchor",
        "thread", "tensile_strength", "shear_strength", "torque_spec", "thread_pitch",
        "thread_size", "hex", "socket_head", "carriage_bolt", "stud", "clamp",
        "flange", "gasket", "o_ring", "seal", "bearing", "bushing", "shaft",
        "keyway", "coupling", "ISO 4014", "DIN 931", "grade", "pitch",
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
        "inductance", "impedance", "frequency", "poles", "contact", "crimp",
    ]),
    ("Wellness", [
        "wellness", "wellbeing", "pillar", "phq", "gad", "fitness_score",
        "physical_fitness", "mental_health", "mood", "stress", "habit",
        "substance_use", "substance_abuse", "diet_quality", "nutrition",
        "social_relations", "sleep_quality", "mindfulness", "resilience",
        "vo2max", "resting_hr", "clean_days", "audit", "cage",
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
        "student", "grade", "course", "enrollment", "attendance", "exam",
        "teacher", "school", "district", "gpa", "module", "assignment", "cohort",
    ]),
    ("Master Data Management", [
        "master data", "mdm", "pim", "golden record", "golden_record",
        "entity resolution", "record linkage", "match merge", "survivorship",
        "data steward", "data governance", "source of record", "system of record",
        "deduplication", "party data", "product master", "customer master",
        "supplier master", "material master", "hierarchy management",
    ]),
]

# Domain-specific role descriptions and analytical lenses
_DOMAIN_PROFILES: dict[str, dict] = {
    "Healthcare Revenue Cycle Management": {
        "role": "healthcare revenue cycle analyst with expertise in claims billing, payer contract management, denial management, and cash collections",
        "analysis_lens": "Focus on adjustment rate by payer and service line, controllable vs non-controllable write-offs, denial root cause (eligibility/auth vs clinical vs billing), days in AR, clean claim rate, and concentration of loss across the payer mix.",
        "anomaly_guidance": "Flag adjustment rates > 10% of billed for any payer (investigate contract terms or billing errors), PERDIEM payers with high write-off % (R&B rate issues), Payor Entry Error / Elig-Auth adjustments (real-time eligibility check gap), Undo Transfer Balance spikes (billing system workflow defect), negative invoicepostDate-to-ServiceDate gaps (data entry errors), and single-payer concentration driving > 30% of total write-offs.",
        "key_metrics": ["adjustment rate % of billed", "controllable write-off amount", "clean claim rate", "days in AR", "denial rate by payer type", "first-pass resolution rate", "payer concentration (HHI)"],
    },
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
    "Automotive Aftermarket / Heavy Truck": {
        "role": "automotive aftermarket product data analyst with expertise in PIES/ACES standards, heavy-duty truck parts cataloging, and OEM cross-referencing",
        "analysis_lens": "Focus on PIES category completeness (L1/L2/L3), cross-reference coverage, brand distribution, physical dimension availability (part and package), attribute standardization, and OE-to-aftermarket interchange accuracy.",
        "anomaly_guidance": "Flag SKUs missing PIES category at any level, duplicate part numbers across brands, cross-references pointing to discontinued parts, missing or zero physical dimensions for shippable items, description inconsistencies vs category assignment, and UPC/GTIN format violations.",
        "key_metrics": ["PIES category completeness %", "cross-reference coverage %", "physical dimension fill rate %", "brand concentration (HHI)", "attribute completeness by category", "duplicate part rate"],
    },
    "Mechanical Components": {
        "role": "mechanical engineering data analyst specialising in fastener specifications, material grades, and joining technology",
        "analysis_lens": "Focus on specification completeness (thread size, grade, material, torque), standards compliance (ISO/DIN/ASME), and compatibility between fastener and mating components.",
        "anomaly_guidance": "Flag missing grade or material fields, impossible thread pitch/size combinations, tensile strength values outside grade norms, duplicate part numbers, and non-standard coating codes.",
        "key_metrics": ["specification completeness %", "grade distribution", "material compliance rate", "catalogue coverage %", "anomaly rate by attribute"],
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
    "Wellness": {
        "role": "wellness and preventive health analyst with expertise in multi-pillar wellness scoring, behaviour change, and population health",
        "analysis_lens": "Focus on the six wellness pillars: physical fitness, psychological wellbeing, diet/nutrition, social relations, medical conditions, and substance absence. Assess pillar scores, identify gaps, and surface individuals at risk of wellness decline.",
        "anomaly_guidance": "Flag pillar scores outside validated scale ranges (e.g. PHQ-9 > 27), missing mandatory assessments, extreme diet or activity outliers, inconsistent substance use logs, and duplicate user records.",
        "key_metrics": ["overall wellness score", "pillar completion rate", "at-risk population %", "PHQ-9 score", "GAD-7 score", "diet quality index", "active days/week", "clean days"],
    },
    "Master Data Management": {
        "role": "master data management specialist with expertise in entity resolution, data governance, golden record design, and multi-source data integration",
        "analysis_lens": "Focus on source system coverage, match/merge quality, survivorship rule effectiveness, golden record completeness, and downstream syndication health. Identify orphan records, duplicate clusters, unmatched entities, and attribute authority conflicts.",
        "anomaly_guidance": "Flag duplicate golden records (same entity, multiple IDs), unmatched source records, attribute conflicts with no survivorship rule, broken parent-child hierarchies, and downstream sync failures.",
        "key_metrics": ["match rate", "duplicate rate", "golden record completeness %", "survivorship conflict rate", "downstream sync latency", "stewardship exception backlog"],
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


# ── Lifecycle stage detection ─────────────────────────────────────────────────

_LIFECYCLE_STAGE_KEYWORDS: dict[str, list[str]] = {
    "sourcing": [
        "data source", "source system", "system of record", "data acquisition",
        "onboarding", "supplier data", "raw data", "ingest source", "source inventory",
        "content onboarding", "feed", "api source", "sensor data",
    ],
    "extraction": [
        "extract", "etl", "elt", "cdc", "pipeline", "ingest", "batch load",
        "landing zone", "staging", "file parse", "api pull", "connector",
        "data pull", "data dump",
    ],
    "enrichment": [
        "enrich", "augment", "attribute mapping", "taxonomy", "classification",
        "entity resolution", "record linkage", "fuzzy match", "standardise",
        "normalise attribute", "reference data", "lookup", "geocode", "annotate",
    ],
    "preparation": [
        "cleanse", "clean", "deduplicate", "dedup", "transform", "normalise",
        "validate", "golden record", "master record", "survivorship", "merge",
        "quality rule", "data prep", "make analysis ready",
    ],
    "publishing": [
        "publish", "data product", "data mart", "data hub", "catalogue entry",
        "api endpoint", "expose data", "serve data", "data contract",
        "semantic layer", "lakehouse", "warehouse", "ods",
    ],
    "syndication": [
        "syndicate", "distribute", "downstream", "push to", "channel feed",
        "partner feed", "crm sync", "erp sync", "data subscription",
        "event stream", "webhook", "broadcast",
    ],
    "analytics": [
        "analyse", "analyze", "report", "dashboard", "insight", "profile",
        "segment", "summarise", "kpi", "trend", "compare", "explore",
        "data quality", "anomaly", "outlier",
    ],
    "ml_modeling": [
        "predict", "model", "train", "classification", "regression", "cluster",
        "feature engineer", "machine learning", "deep learning", "xgboost",
        "random forest", "churn model", "risk model", "propensity",
    ],
    "survival_analysis": [
        "survival", "time to event", "churn", "kaplan meier", "hazard",
        "cox", "retention", "time to failure", "event study", "tenure",
    ],
}


def _detect_lifecycle_stage(user_query: str, sol: dict | None = None) -> str:
    """Infer the lifecycle stage from user query text and/or a matched solution entry.

    Priority order:
      1. If the solution entry already has a lifecycle_stage, return it.
      2. Score the user query against _LIFECYCLE_STAGE_KEYWORDS.
      3. Default to 'analytics'.
    """
    # 1. Trust the solution registry first
    if sol and sol.get("lifecycle_stage"):
        return sol["lifecycle_stage"]

    # 2. Semantic lifecycle detection
    semantic_stage = _semantic_lifecycle(user_query)
    if semantic_stage:
        return semantic_stage

    # 3. Score query against keywords (fallback)
    if user_query:
        q = user_query.lower()
        scores: dict[str, int] = {
            stage: sum(1 for kw in kws if kw in q)
            for stage, kws in _LIFECYCLE_STAGE_KEYWORDS.items()
        }
        best = max(scores, key=lambda s: scores[s])
        if scores[best] > 0:
            return best

    # 4. Default
    return "analytics"


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


def _get_domain_conn_safe():
    """Return DB connection or None (no crash if DB down)."""
    try:
        import psycopg2
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "Data_Analysis", ".env",
        )
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        url = os.environ.get(
            "DOMAIN_DB_URL",
            "postgresql://postgres:memorystorepg@localhost:5432/domain",
        )
        return psycopg2.connect(url)
    except Exception:
        return None


def _semantic_domain(df: pd.DataFrame) -> Optional[str]:
    """Embedding-based domain detection. Returns domain name or None."""
    try:
        from hooks.embedding_registry import embed_texts, cosine_rank
    except Exception:
        return None

    conn = _get_domain_conn_safe()
    if conn is None:
        return None

    try:
        sig = _col_signature(df)
        q_vec = embed_texts([sig])[0]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT domain_name, embedding FROM domain_keyword_embeddings WHERE category = 'domain'"
            )
            rows = cur.fetchall()

        if not rows:
            return None

        candidates = [(i, row[1]) for i, row in enumerate(rows)]
        ranked = cosine_rank(q_vec, candidates)
        if not ranked:
            return None

        top_idx, top_score = ranked[0]
        if top_score < 0.40:
            return None

        top_domain = rows[top_idx][0]

        # Ambiguity tie-breaker: apply domain-knowledge precedence rules
        if len(ranked) >= 2:
            second_idx, second_score = ranked[1]
            if top_score - second_score < 0.08:
                second_domain = rows[second_idx][0]
                # Apply same precedence as _heuristic_domain
                if top_domain == "Healthcare" and second_domain == "Healthcare Revenue Cycle Management":
                    return second_domain
                if top_domain == "Healthcare" and second_domain == "Healthcare Supply Chain / GPO":
                    return second_domain
                if top_domain == "Healthcare" and second_domain == "Wellness":
                    return second_domain
                if top_domain == "Electrical Components" and second_domain == "Mechanical Components":
                    return second_domain
                if top_domain == "Industrial Equipment" and second_domain == "Electrical Components":
                    return second_domain

        return top_domain
    except Exception as exc:
        log.debug("Semantic domain detection failed: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _semantic_solution(
    df: pd.DataFrame, domain: str, user_query: str = "",
) -> Optional[dict]:
    """Embedding-based solution matching. Returns solution dict or None."""
    try:
        from hooks.embedding_registry import embed_texts, cosine_rank
    except Exception:
        return None

    conn = _get_domain_conn_safe()
    if conn is None:
        return None

    try:
        cols = ", ".join(df.columns.tolist()[:30])
        embed_text = f"{user_query}. Domain: {domain}. Columns: {cols}"
        q_vec = embed_texts([embed_text])[0]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT domain_name, embed_text, embedding FROM domain_keyword_embeddings WHERE category = 'solution'"
            )
            rows = cur.fetchall()

        if not rows:
            return None

        candidates = [(i, row[2]) for i, row in enumerate(rows)]
        ranked = cosine_rank(q_vec, candidates)
        if not ranked:
            return None

        top_idx, top_score = ranked[0]
        if top_score < 0.45:
            return None

        # Parse key from domain_name: "domain::solution_name"
        key = rows[top_idx][0]
        parts = key.split("::", 1)
        if len(parts) != 2:
            return None

        sol_domain, sol_name = parts
        # Look up in solutions registry
        registry = _load_solutions()
        for sol in registry.get(sol_domain, []):
            if isinstance(sol, dict) and sol.get("name") == sol_name:
                return sol

        return None
    except Exception as exc:
        log.debug("Semantic solution matching failed: %s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _semantic_lifecycle(user_query: str) -> Optional[str]:
    """Embedding-based lifecycle stage detection. Returns stage name or None."""
    if not user_query:
        return None
    try:
        from hooks.embedding_registry import embed_texts, cosine_rank
    except Exception:
        return None

    conn = _get_domain_conn_safe()
    if conn is None:
        return None

    try:
        q_vec = embed_texts([user_query])[0]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT domain_name, embedding FROM domain_keyword_embeddings WHERE category = 'lifecycle'"
            )
            rows = cur.fetchall()

        if not rows:
            return None

        candidates = [(i, row[1]) for i, row in enumerate(rows)]
        ranked = cosine_rank(q_vec, candidates)
        if not ranked:
            return None

        top_idx, top_score = ranked[0]
        if top_score < 0.45:
            return None

        return rows[top_idx][0]
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _semantic_resolve_solution_by_name(domain: str, name_raw: str) -> Optional[dict]:
    """Embedding-based solution name matching. Returns solution dict or None."""
    if not name_raw:
        return None
    try:
        from hooks.embedding_registry import embed_texts, cosine_rank
    except Exception:
        return None

    conn = _get_domain_conn_safe()
    if conn is None:
        return None

    try:
        q_vec = embed_texts([name_raw])[0]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT domain_name, embedding FROM domain_keyword_embeddings WHERE category = 'solution'"
            )
            rows = cur.fetchall()

        if not rows:
            return None

        candidates = [(i, row[1]) for i, row in enumerate(rows)]
        ranked = cosine_rank(q_vec, candidates)
        if not ranked:
            return None

        top_idx, top_score = ranked[0]
        if top_score < 0.60:
            return None

        key = rows[top_idx][0]
        parts = key.split("::", 1)
        if len(parts) != 2:
            return None

        sol_domain, sol_name = parts
        registry = _load_solutions()
        for sol in registry.get(sol_domain, []):
            if isinstance(sol, dict) and sol.get("name") == sol_name:
                return sol
        return None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
    # Resolve ambiguity: RCM beats generic Healthcare when billing/claims signals dominate
    if best == "Healthcare" and scores.get("Healthcare Revenue Cycle Management", 0) >= scores["Healthcare"]:
        best = "Healthcare Revenue Cycle Management"
    # Resolve ambiguity: GPO/supply chain beats generic Healthcare
    if best == "Healthcare" and scores.get("Healthcare Supply Chain / GPO", 0) >= scores["Healthcare"]:
        best = "Healthcare Supply Chain / GPO"
    # Resolve ambiguity: Wellness beats Healthcare when wellness signals dominate
    if best == "Healthcare" and scores.get("Wellness", 0) >= scores["Healthcare"]:
        best = "Wellness"
    # Resolve ambiguity: Mechanical Components beats Electrical Components when fastener signals present
    if best == "Electrical Components" and scores.get("Mechanical Components", 0) >= scores["Electrical Components"]:
        best = "Mechanical Components"
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
        "lifecycle_stage": "analytics",
        "description": f"Profile and surface patterns in this {domain} dataset.",
        "triggers": [],
        "typical_tools": ["Descriptive Statistics", "Correlation Analysis", "Anomaly Detection"],
        "key_outputs": ["data_profile", "quality_report", "top_findings"],
    }


def _resolve_solution_by_name(domain: str, name_raw: str) -> Optional[dict]:
    """Find a solution entry by fuzzy name match within a domain (and across all domains).

    Tries semantic matching first, falls back to bidirectional substring.
    Returns None if no match found.
    """
    if not name_raw:
        return None

    # Semantic match first
    semantic = _semantic_resolve_solution_by_name(domain, name_raw)
    if semantic:
        return semantic

    # Keyword fallback: bidirectional substring
    registry = _load_solutions()
    name_lower = name_raw.lower()

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
    """Determine the most likely solution type and lifecycle stage for a detected domain.

    Uses trigger-keyword scoring against column names, sample values, and the
    optional user query. Falls back to 'Exploratory Data Analysis' when no
    triggers match.

    Returns a dict with keys:
      solution_name        : str — named solution type (e.g. 'Mortality Prediction')
      solution_description : str — what this solution does
      lifecycle_stage      : str — sourcing|extraction|enrichment|preparation|publishing|syndication|analytics|ml_modeling|survival_analysis
      typical_tools        : list[str] — tools / models commonly used
      key_outputs          : list[str] — expected deliverables
    """
    sol = _heuristic_solution(df, domain_ctx.get("domain", "General"), user_query)
    tools = sol.get("typical_tools") or sol.get("typical_models") or []
    return {
        "solution_name":        sol.get("name", "Exploratory Data Analysis"),
        "solution_description": sol.get("description", ""),
        "lifecycle_stage":      _detect_lifecycle_stage(user_query, sol),
        "typical_tools":        tools,
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
      lifecycle_stage      : str — where in the data lifecycle this solution sits
      typical_tools        : list[str] — tools / models commonly used for this solution
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

        lifecycle_stages = list(_LIFECYCLE_STAGE_KEYWORDS.keys())

        prompt = (
            f"You are a data domain and solution classification expert.\n\n"
            f"Examine the following dataset signature and determine:\n"
            f"1. The business domain\n"
            f"2. The most likely solution type being built with this data\n"
            f"3. The data lifecycle stage this solution operates in\n\n"
            f"{signature}\n\n"
            f"{'User query: ' + user_query if user_query else ''}\n\n"
            f"Choose the single best-fitting domain from:\n"
            f"{', '.join(domain_names)}\n\n"
            f"Choose the single best-fitting solution type from:\n"
            f"{', '.join(all_solution_names)}\n\n"
            f"Choose the single best-fitting lifecycle stage from:\n"
            f"{', '.join(lifecycle_stages)}\n\n"
            f"Then write a one-paragraph plain-English summary of what this dataset appears to contain "
            f"and what business questions it could answer.\n\n"
            f"Reply with exactly this format (nothing else):\n"
            f"DOMAIN: <domain name from list above>\n"
            f"CONFIDENCE: high or medium or low\n"
            f"SOLUTION: <solution type from list above>\n"
            f"LIFECYCLE: <lifecycle stage from list above>\n"
            f"SUMMARY: <one paragraph, max 80 words>"
        )
        try:
            from hooks.llm_client import call_llm
            text = call_llm(prompt, max_tokens=300)
            if text is None:
                raise RuntimeError("No LLM response")

            domain = "General"
            confidence = "medium"
            solution_name_raw = ""
            lifecycle_stage_raw = ""
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
                elif line.upper().startswith("LIFECYCLE:"):
                    lifecycle_stage_raw = line.split(":", 1)[1].strip().lower()
                elif line.upper().startswith("SUMMARY:"):
                    summary = line.split(":", 1)[1].strip()

            profile = _get_domain_profiles().get(domain, _DOMAIN_PROFILES["General"])

            # Resolve solution: try to match Claude's answer to a known registry entry
            sol = _resolve_solution_by_name(domain, solution_name_raw) or \
                  _heuristic_solution(df, domain, user_query)

            # Resolve lifecycle: trust Claude if valid, else derive from solution/query
            lifecycle_stage = (
                lifecycle_stage_raw
                if lifecycle_stage_raw in _LIFECYCLE_STAGE_KEYWORDS
                else _detect_lifecycle_stage(user_query, sol)
            )
            tools = sol.get("typical_tools") or sol.get("typical_models") or []

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
                "lifecycle_stage":      lifecycle_stage,
                "typical_tools":        tools,
                "key_outputs":          sol.get("key_outputs", []),
            }

        except Exception:
            pass  # Fall through to heuristic

    # Semantic + heuristic fallback
    semantic_domain = _semantic_domain(df)
    domain = semantic_domain or _heuristic_domain(df)
    method = "semantic" if semantic_domain else "heuristic"
    profile = _get_domain_profiles().get(domain, _DOMAIN_PROFILES["General"])
    sol = _semantic_solution(df, domain, user_query) or _heuristic_solution(df, domain, user_query)
    tools = sol.get("typical_tools") or sol.get("typical_models") or []
    return {
        "domain":               domain,
        "confidence":           "medium",
        "role":                 profile["role"],
        "summary":              f"{'Semantic' if method == 'semantic' else 'Heuristic'} detection identified this as a {domain} dataset based on column names and sample values.",
        "analysis_lens":        profile["analysis_lens"],
        "anomaly_guidance":     profile["anomaly_guidance"],
        "key_metrics":          profile["key_metrics"],
        "method":               method,
        "solution_name":        sol.get("name", "Exploratory Data Analysis"),
        "solution_description": sol.get("description", ""),
        "lifecycle_stage":      _detect_lifecycle_stage(user_query, sol),
        "typical_tools":        tools,
        "key_outputs":          sol.get("key_outputs", []),
    }


def domain_role_prefix(domain_ctx: dict) -> str:
    """Return a formatted role-and-context block to prepend to every LLM reasoning step.

    Paste this at the start of your analysis narrative to ground all
    interpretations in the detected domain and solution scope.
    """
    solution_name  = domain_ctx.get("solution_name", "")
    solution_desc  = domain_ctx.get("solution_description", "")
    lifecycle      = domain_ctx.get("lifecycle_stage", "")
    # Support both old 'typical_models' and new 'typical_tools' key
    typical_tools  = domain_ctx.get("typical_tools") or domain_ctx.get("typical_models") or []
    key_outputs    = domain_ctx.get("key_outputs", [])

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
            f"Lifecycle: {lifecycle}" if lifecycle else "",
            f"Purpose  : {solution_desc}",
        ]
        lines = [l for l in lines if l != ""]  # remove blank placeholder lines
        if typical_tools:
            lines.append(f"Tools    : {', '.join(typical_tools)}")
        if key_outputs:
            lines.append(f"Outputs  : {', '.join(key_outputs)}")
    lines.append(f"======================")
    return "\n".join(lines)
