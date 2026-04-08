"""
seed_careaxl_intents.py — Seed CareAXL-scoped intent nodes, triplets, and aliases.

Run (idempotent):
    python hooks/seed_careaxl_intents.py

Coverage
--------
  Mid level       (3 nodes)  : app-wide healthcare subject areas
  Specific level  (24 nodes) : concrete operations per solution area
  IS_A triplets   ~30
  Other triplets  ~15  (REQUIRES, ENABLES, PRECEDES)
  Aliases         ~40

All nodes have app_id='careaxl' and solution_id per solution area (or None for app-wide).
"""

from __future__ import annotations

import sys
import os

_hooks_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_hooks_dir)
sys.path.insert(0, _hooks_dir)
sys.path.insert(0, _repo_dir)

from intent_graph import (
    get_intent_conn,
    init_intent_db,
    upsert_intent_node,
    add_intent_triplet,
    add_intent_alias,
    embed_intent_nodes,
)


# ---------------------------------------------------------------------------
# Node definitions
# format: (name, label, description, level, domain, lifecycle_stage,
#          app_id, solution_id)
# ---------------------------------------------------------------------------

NODES = [

    # ── MID-LEVEL (app-wide, no specific solution) ──────────────────────────

    (
        "care_risk_analytics",
        "Care Risk Analytics",
        "Analyse patient risk factors, stratify populations by clinical risk, plan interventions, and monitor adverse events and deterioration.",
        "mid", "Healthcare", "analytics",
        "careaxl", None,
    ),
    (
        "patient_master_data",
        "Patient Master Data",
        "Construct and maintain golden patient records via entity resolution, longitudinal views, and care coordination data quality.",
        "mid", "Healthcare", "analytics",
        "careaxl", None,
    ),
    (
        "wellness_analytics",
        "Wellness Analytics",
        "Track preventative care compliance, identify care gaps, monitor chronic disease metrics, and evaluate population health programs.",
        "mid", "Healthcare", "analytics",
        "careaxl", None,
    ),

    # ── SPECIFIC: Revenue Cycle Management ──────────────────────────────────

    (
        "denial_analysis",
        "Denial Analysis",
        "Analyse claim denial patterns by payer, reason code, and service line to identify root causes and recovery opportunities.",
        "specific", "Healthcare RCM", "analytics",
        "careaxl", "revenue_cycle_management",
    ),
    (
        "reimbursement_optimization",
        "Reimbursement Optimization",
        "Identify under-reimbursed claims by comparing paid amounts to contract rates and benchmark expected versus actual payment.",
        "specific", "Healthcare RCM", "analytics",
        "careaxl", "revenue_cycle_management",
    ),
    (
        "coding_accuracy_audit",
        "Coding Accuracy Audit",
        "Audit medical coding accuracy by comparing assigned ICD-10/CPT codes against clinical documentation and payer edit rules.",
        "specific", "Healthcare RCM", "analytics",
        "careaxl", "revenue_cycle_management",
    ),
    (
        "ar_aging_analysis",
        "AR Aging Analysis",
        "Analyse accounts receivable aging buckets, identify stale claims, and prioritise collection follow-up by dollar impact.",
        "specific", "Healthcare RCM", "analytics",
        "careaxl", "revenue_cycle_management",
    ),
    (
        "charge_capture_review",
        "Charge Capture Review",
        "Audit charge capture completeness by comparing documented services against posted charges to identify revenue leakage.",
        "specific", "Healthcare RCM", "analytics",
        "careaxl", "revenue_cycle_management",
    ),
    (
        "payer_performance",
        "Payer Performance Analysis",
        "Benchmark payer performance on denial rates, payment turnaround, reimbursement accuracy, and contract compliance.",
        "specific", "Healthcare RCM", "analytics",
        "careaxl", "revenue_cycle_management",
    ),

    # ── SPECIFIC: Care Risk Management ──────────────────────────────────────

    (
        "risk_stratification",
        "Risk Stratification",
        "Stratify patient populations into risk tiers (high/medium/low) using clinical scores, comorbidities, and predictive models.",
        "specific", "Healthcare", "analytics",
        "careaxl", "care_risk_management",
    ),
    (
        "intervention_planning",
        "Intervention Planning",
        "Design and prioritise clinical interventions based on risk scores, resource availability, and evidence-based protocols.",
        "specific", "Healthcare", "analytics",
        "careaxl", "care_risk_management",
    ),
    (
        "adverse_event_analysis",
        "Adverse Event Analysis",
        "Analyse adverse event patterns (falls, infections, medication errors) to identify systemic causes and prevention opportunities.",
        "specific", "Healthcare", "analytics",
        "careaxl", "care_risk_management",
    ),
    (
        "deterioration_detection",
        "Deterioration Detection",
        "Monitor early warning scores (NEWS2, MEWS) and vital sign trends to detect clinical deterioration before critical events.",
        "specific", "Healthcare", "analytics",
        "careaxl", "care_risk_management",
    ),
    (
        "medication_adherence_analysis",
        "Medication Adherence Analysis",
        "Measure medication adherence rates (PDC, MPR) and identify patients at risk of non-compliance for targeted outreach.",
        "specific", "Healthcare", "analytics",
        "careaxl", "care_risk_management",
    ),
    (
        "transition_analysis",
        "Care Transition Analysis",
        "Analyse care transitions (hospital-to-home, ICU-to-floor) for handoff risk factors and readmission predictors.",
        "specific", "Healthcare", "analytics",
        "careaxl", "care_risk_management",
    ),

    # ── SPECIFIC: Patient 360 ───────────────────────────────────────────────

    (
        "patient_entity_resolution",
        "Patient Entity Resolution",
        "Match and deduplicate patient records across facilities using deterministic and probabilistic algorithms to build the EMPI.",
        "specific", "Healthcare", "analytics",
        "careaxl", "patient_360",
    ),
    (
        "longitudinal_view",
        "Longitudinal Patient View",
        "Assemble a chronological view of a patient's clinical history across all encounters, facilities, and care settings.",
        "specific", "Healthcare", "analytics",
        "careaxl", "patient_360",
    ),
    (
        "care_coordination",
        "Care Coordination Analytics",
        "Analyse care team composition, referral patterns, and coordination effectiveness across providers for a patient.",
        "specific", "Healthcare", "analytics",
        "careaxl", "patient_360",
    ),
    (
        "consent_management",
        "Consent Management",
        "Track and validate patient consent records for data sharing, treatment authorisation, and research participation.",
        "specific", "Healthcare", "analytics",
        "careaxl", "patient_360",
    ),
    (
        "patient_data_quality",
        "Patient Data Quality",
        "Profile and score data quality of patient master records — completeness, accuracy, consistency, and timeliness.",
        "specific", "Healthcare", "analytics",
        "careaxl", "patient_360",
    ),

    # ── SPECIFIC: Wellness & Preventative Health ────────────────────────────

    (
        "care_gap_analysis",
        "Care Gap Analysis",
        "Identify patients with open preventative care gaps by evaluating eligibility against screening protocols and service history.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
    (
        "screening_compliance",
        "Screening Compliance",
        "Measure and report compliance rates for preventative screenings (mammography, colonoscopy, A1C) against HEDIS measures.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
    (
        "chronic_disease_mgmt",
        "Chronic Disease Management",
        "Monitor chronic disease control metrics (A1C, blood pressure, lipids) and identify patients with uncontrolled conditions.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
    (
        "population_health_analysis",
        "Population Health Analysis",
        "Analyse population-level health trends, risk factor prevalence, and health equity disparities across demographics.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
    (
        "program_effectiveness",
        "Program Effectiveness",
        "Evaluate the impact and ROI of wellness programs by comparing outcomes, utilisation, and cost before and after intervention.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
    (
        "vaccination_tracking",
        "Vaccination Tracking",
        "Track vaccination records, series completion, and registry reporting compliance across patient populations.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
    (
        "lifestyle_risk_assessment",
        "Lifestyle Risk Assessment",
        "Assess patient lifestyle risk factors (smoking, BMI, physical activity, nutrition) and prioritise wellness interventions.",
        "specific", "Healthcare", "analytics",
        "careaxl", "wellness_preventative_health",
    ),
]


# ---------------------------------------------------------------------------
# Triplets
# format: (subject_name, predicate, object_name, weight)
# ---------------------------------------------------------------------------

TRIPLETS = [

    # ── IS_A: Mid → Abstract (link to existing base abstract nodes) ─────────
    # These reference abstract nodes from the base seed (seed_intent_graph.py)
    ("care_risk_analytics", "IS_A", "predictive_analytics", 1.0),
    ("patient_master_data", "IS_A", "data_management", 1.0),
    ("wellness_analytics", "IS_A", "exploratory_analytics", 1.0),

    # ── IS_A: Specific → Mid (RCM → existing base mid-level "revenue_cycle") ──
    # Note: RCM specifics link to a base mid-level node if it exists, else to
    # the "exploratory_analytics" abstract as a fallback. We use the mid-level
    # careaxl nodes + direct abstract fallback for cleaner hierarchy.
    ("denial_analysis", "IS_A", "care_risk_analytics", 0.7),
    ("reimbursement_optimization", "IS_A", "care_risk_analytics", 0.5),
    ("coding_accuracy_audit", "IS_A", "care_risk_analytics", 0.5),
    ("ar_aging_analysis", "IS_A", "care_risk_analytics", 0.5),
    ("charge_capture_review", "IS_A", "care_risk_analytics", 0.5),
    ("payer_performance", "IS_A", "care_risk_analytics", 0.5),

    # ── IS_A: Specific → Mid (Care Risk) ────────────────────────────────────
    ("risk_stratification", "IS_A", "care_risk_analytics", 1.0),
    ("intervention_planning", "IS_A", "care_risk_analytics", 1.0),
    ("adverse_event_analysis", "IS_A", "care_risk_analytics", 1.0),
    ("deterioration_detection", "IS_A", "care_risk_analytics", 1.0),
    ("medication_adherence_analysis", "IS_A", "care_risk_analytics", 0.8),
    ("transition_analysis", "IS_A", "care_risk_analytics", 0.8),

    # ── IS_A: Specific → Mid (Patient 360) ──────────────────────────────────
    ("patient_entity_resolution", "IS_A", "patient_master_data", 1.0),
    ("longitudinal_view", "IS_A", "patient_master_data", 1.0),
    ("care_coordination", "IS_A", "patient_master_data", 0.8),
    ("consent_management", "IS_A", "patient_master_data", 0.7),
    ("patient_data_quality", "IS_A", "patient_master_data", 1.0),

    # ── IS_A: Specific → Mid (Wellness) ─────────────────────────────────────
    ("care_gap_analysis", "IS_A", "wellness_analytics", 1.0),
    ("screening_compliance", "IS_A", "wellness_analytics", 1.0),
    ("chronic_disease_mgmt", "IS_A", "wellness_analytics", 1.0),
    ("population_health_analysis", "IS_A", "wellness_analytics", 1.0),
    ("program_effectiveness", "IS_A", "wellness_analytics", 0.8),
    ("vaccination_tracking", "IS_A", "wellness_analytics", 0.8),
    ("lifestyle_risk_assessment", "IS_A", "wellness_analytics", 0.8),

    # ── REQUIRES ────────────────────────────────────────────────────────────
    ("intervention_planning", "REQUIRES", "risk_stratification", 1.0),
    ("reimbursement_optimization", "REQUIRES", "denial_analysis", 0.8),
    ("longitudinal_view", "REQUIRES", "patient_entity_resolution", 1.0),
    ("care_coordination", "REQUIRES", "longitudinal_view", 0.8),
    ("screening_compliance", "REQUIRES", "care_gap_analysis", 1.0),
    ("program_effectiveness", "REQUIRES", "screening_compliance", 0.7),

    # ── ENABLES ─────────────────────────────────────────────────────────────
    ("risk_stratification", "ENABLES", "intervention_planning", 1.0),
    ("denial_analysis", "ENABLES", "reimbursement_optimization", 0.8),
    ("patient_entity_resolution", "ENABLES", "longitudinal_view", 1.0),
    ("care_gap_analysis", "ENABLES", "screening_compliance", 1.0),
    ("deterioration_detection", "ENABLES", "adverse_event_analysis", 0.7),

    # ── PRECEDES ────────────────────────────────────────────────────────────
    ("coding_accuracy_audit", "PRECEDES", "denial_analysis", 0.8),
    ("charge_capture_review", "PRECEDES", "coding_accuracy_audit", 0.7),
    ("patient_data_quality", "PRECEDES", "patient_entity_resolution", 0.9),
    ("lifestyle_risk_assessment", "PRECEDES", "chronic_disease_mgmt", 0.6),
]


# ---------------------------------------------------------------------------
# Aliases
# format: (intent_name, alias, alias_type)
# ---------------------------------------------------------------------------

ALIASES = [
    # RCM
    ("denial_analysis", "denial management", "phrase"),
    ("denial_analysis", "claim denial", "keyword"),
    ("denial_analysis", "denial rate", "keyword"),
    ("denial_analysis", "CARC", "keyword"),
    ("reimbursement_optimization", "underpayment", "keyword"),
    ("reimbursement_optimization", "contract variance", "phrase"),
    ("reimbursement_optimization", "payment recovery", "phrase"),
    ("coding_accuracy_audit", "coding audit", "phrase"),
    ("coding_accuracy_audit", "ICD-10", "keyword"),
    ("coding_accuracy_audit", "CPT coding", "phrase"),
    ("ar_aging_analysis", "AR days", "keyword"),
    ("ar_aging_analysis", "accounts receivable", "phrase"),
    ("ar_aging_analysis", "aging bucket", "phrase"),
    ("charge_capture_review", "charge capture", "phrase"),
    ("charge_capture_review", "revenue leakage", "phrase"),
    ("payer_performance", "payer mix", "phrase"),
    ("payer_performance", "payer contract", "phrase"),

    # Care Risk
    ("risk_stratification", "patient risk", "phrase"),
    ("risk_stratification", "risk score", "keyword"),
    ("risk_stratification", "risk tier", "keyword"),
    ("intervention_planning", "care plan", "phrase"),
    ("intervention_planning", "intervention", "keyword"),
    ("adverse_event_analysis", "adverse event", "phrase"),
    ("adverse_event_analysis", "patient safety", "phrase"),
    ("deterioration_detection", "early warning", "phrase"),
    ("deterioration_detection", "NEWS2", "keyword"),
    ("deterioration_detection", "clinical deterioration", "phrase"),
    ("medication_adherence_analysis", "medication adherence", "phrase"),
    ("medication_adherence_analysis", "PDC", "keyword"),
    ("transition_analysis", "care transition", "phrase"),
    ("transition_analysis", "readmission risk", "phrase"),

    # Patient 360
    ("patient_entity_resolution", "entity resolution", "phrase"),
    ("patient_entity_resolution", "patient matching", "phrase"),
    ("patient_entity_resolution", "EMPI", "keyword"),
    ("patient_entity_resolution", "MPI", "keyword"),
    ("longitudinal_view", "patient 360", "phrase"),
    ("longitudinal_view", "patient timeline", "phrase"),
    ("longitudinal_view", "longitudinal record", "phrase"),
    ("care_coordination", "care team", "phrase"),
    ("care_coordination", "referral pattern", "phrase"),
    ("consent_management", "patient consent", "phrase"),
    ("consent_management", "data sharing", "phrase"),
    ("patient_data_quality", "patient data quality", "phrase"),
    ("patient_data_quality", "golden record quality", "phrase"),

    # Wellness
    ("care_gap_analysis", "care gap", "phrase"),
    ("care_gap_analysis", "gap in care", "phrase"),
    ("screening_compliance", "screening compliance", "phrase"),
    ("screening_compliance", "HEDIS", "keyword"),
    ("screening_compliance", "mammography", "keyword"),
    ("chronic_disease_mgmt", "chronic disease", "phrase"),
    ("chronic_disease_mgmt", "diabetes management", "phrase"),
    ("chronic_disease_mgmt", "A1C", "keyword"),
    ("population_health_analysis", "population health", "phrase"),
    ("population_health_analysis", "health equity", "phrase"),
    ("program_effectiveness", "program ROI", "phrase"),
    ("program_effectiveness", "wellness program", "phrase"),
    ("vaccination_tracking", "vaccination", "keyword"),
    ("vaccination_tracking", "immunization", "keyword"),
    ("vaccination_tracking", "vaccine registry", "phrase"),
    ("lifestyle_risk_assessment", "lifestyle risk", "phrase"),
    ("lifestyle_risk_assessment", "BMI", "keyword"),
    ("lifestyle_risk_assessment", "smoking cessation", "phrase"),
]


# ---------------------------------------------------------------------------
# Main seed function
# ---------------------------------------------------------------------------

def seed() -> None:
    conn = get_intent_conn()
    init_intent_db(conn)

    # 1. Upsert nodes
    node_count = 0
    for row in NODES:
        name, label, desc, level, domain, lifecycle, app_id, solution_id = row
        nid = upsert_intent_node(
            conn, name, label, desc, level, domain, lifecycle,
            app_id=app_id, solution_id=solution_id,
        )
        conn.commit()
        print(f"  NODE  {name} (id={nid}, level={level}, app={app_id}, sol={solution_id})")
        node_count += 1

    # 2. Add triplets
    triplet_count = 0
    for subj, pred, obj, weight in TRIPLETS:
        try:
            tid = add_intent_triplet(conn, subj, pred, obj, weight)
            conn.commit()
            triplet_count += 1
        except ValueError as e:
            print(f"  SKIP  triplet {subj} -{pred}-> {obj}: {e}")

    # 3. Add aliases
    alias_count = 0
    for intent_name, alias, alias_type in ALIASES:
        try:
            add_intent_alias(conn, intent_name, alias, alias_type)
            conn.commit()
            alias_count += 1
        except ValueError as e:
            print(f"  SKIP  alias {intent_name}/{alias}: {e}")

    # 4. Embed new nodes
    print("  Embedding new nodes...")
    embedded = embed_intent_nodes(conn)
    print(f"  Embedded {embedded} nodes.")

    conn.close()
    print(f"Seed complete: {node_count} nodes, {triplet_count} triplets, {alias_count} aliases.")


if __name__ == "__main__":
    seed()
