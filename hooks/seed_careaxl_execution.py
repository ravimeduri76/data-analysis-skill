"""
seed_careaxl_execution.py — Seed CareAXL-scoped execution plans, steps, deps, and examples.

Run (idempotent):
    python hooks/seed_careaxl_execution.py

Coverage
--------
  Plans      : 6  (2 RCM, 2 Care Risk, 1 Patient 360, 1 Wellness)
  Steps      : ~40
  Dependencies: ~34
  Examples   : 6  (one per plan)

All plans have app_id='careaxl' and solution_id per solution area.
Agent cards are NOT re-seeded — they use the existing base AG01-AG40.
"""

from __future__ import annotations

import json
import sys
import os

_hooks_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_hooks_dir)
sys.path.insert(0, _hooks_dir)
sys.path.insert(0, _repo_dir)

from execution_graph import (
    get_execution_conn,
    init_execution_db,
    upsert_execution_plan,
    add_execution_step,
    add_execution_dependency,
    add_execution_example,
    backfill_execution_embeddings,
)


def log(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

PLANS = [
    # ── 1. RCM Analytics Pipeline ──────────────────────────────────────────
    {
        "name": "rcm_analytics_pipeline",
        "label": "RCM Analytics Pipeline",
        "description": "Revenue cycle analytics: ingest claims data, profile, detect anomalies, assess data quality, and produce RCM performance dashboards.",
        "intent_patterns": [
            "revenue cycle", "RCM analytics", "claims analysis", "AR aging",
            "charge capture", "payer performance", "reimbursement analysis",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Healthcare", "Healthcare RCM"],
        "requires_dataset": True,
        "app_id": "careaxl",
        "solution_id": "revenue_cycle_management",
    },
    # ── 2. RCM Denial Management Pipeline ──────────────────────────────────
    {
        "name": "rcm_denial_management_pipeline",
        "label": "RCM Denial Management Pipeline",
        "description": "Denial-focused pipeline: ingest denial data, profile denial patterns, apply DQ rules, detect anomalies, analyse trends, and generate recovery recommendations.",
        "intent_patterns": [
            "denial management", "denial analysis", "claim denial", "denial rate",
            "appeal management", "denial root cause",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Healthcare", "Healthcare RCM"],
        "requires_dataset": True,
        "app_id": "careaxl",
        "solution_id": "revenue_cycle_management",
    },
    # ── 3. Care Risk Stratification Pipeline ───────────────────────────────
    {
        "name": "care_risk_stratification_pipeline",
        "label": "Care Risk Stratification Pipeline",
        "description": "Patient risk analytics: ingest clinical data, profile, enrich with external risk factors, detect anomalies, produce risk stratification and dashboards.",
        "intent_patterns": [
            "risk stratification", "patient risk", "fall risk", "clinical deterioration",
            "risk score", "risk tier", "adverse event analysis",
        ],
        "lifecycle_stages": ["analytics", "ml_modeling"],
        "domains": ["Healthcare"],
        "requires_dataset": True,
        "app_id": "careaxl",
        "solution_id": "care_risk_management",
    },
    # ── 4. Intervention Planning Pipeline ──────────────────────────────────
    {
        "name": "intervention_planning_pipeline",
        "label": "Intervention Planning Pipeline",
        "description": "Intervention design pipeline: ingest risk-stratified data, analyse intervention needs, generate recommendations, and produce content for care plans.",
        "intent_patterns": [
            "intervention planning", "care plan", "intervention", "medication adherence",
            "care transition", "readmission prevention",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Healthcare"],
        "requires_dataset": True,
        "app_id": "careaxl",
        "solution_id": "care_risk_management",
    },
    # ── 5. Patient 360 Pipeline ────────────────────────────────────────────
    {
        "name": "patient_360_pipeline",
        "label": "Patient 360 Pipeline",
        "description": "Master patient construction: ingest multi-source patient data, profile, recommend schema, harmonize, build ontology, enrich, resolve entities, build 360 view, and produce golden records.",
        "intent_patterns": [
            "patient 360", "golden record", "entity resolution", "EMPI",
            "patient matching", "longitudinal view", "master patient index",
            "patient master data",
        ],
        "lifecycle_stages": ["enrichment", "preparation"],
        "domains": ["Healthcare", "MDM"],
        "requires_dataset": True,
        "app_id": "careaxl",
        "solution_id": "patient_360",
    },
    # ── 6. Wellness Analytics Pipeline ─────────────────────────────────────
    {
        "name": "wellness_analytics_pipeline",
        "label": "Wellness Analytics Pipeline",
        "description": "Population health analytics: ingest patient wellness data, profile, detect anomalies, enrich with screening protocols, produce care gap reports and compliance dashboards.",
        "intent_patterns": [
            "wellness", "screening compliance", "care gap", "chronic disease",
            "vaccination tracking", "population health", "HEDIS", "AWV",
            "preventative care",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Healthcare"],
        "requires_dataset": True,
        "app_id": "careaxl",
        "solution_id": "wellness_preventative_health",
    },
]


# ---------------------------------------------------------------------------
# Steps
# format: (plan_name, step_order, agent_id, purpose,
#          ingress_map, egress_keys, is_optional, condition, hitl_gate, hitl_reason)
# ---------------------------------------------------------------------------

STEPS = [

    # ── 1. rcm_analytics_pipeline: AG04 → AG06 → AG07 → AG14 → AG17 → AG36
    ("rcm_analytics_pipeline", 1, "AG04", "Ingest claims/billing data from source systems",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("rcm_analytics_pipeline", 2, "AG06", "Initial profiling: route RCM columns to LLM vs program",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("rcm_analytics_pipeline", 3, "AG07", "Full profiling: schema, table, and column-level statistics",
     {"dataset": "$step.1.egress.dataset", "column_routing": "$step.2.egress", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "statistical_summary"], False, None, None, None),
    ("rcm_analytics_pipeline", 4, "AG14", "Detect anomalies in claims data: denial spikes, charge outliers, billing irregularities",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["anomaly_report", "outlier_rows"], False, None, None, None),
    ("rcm_analytics_pipeline", 5, "AG17", "Score data quality across completeness, correctness, and consistency dimensions",
     {"profiling_output": "$step.3.egress", "anomaly_report": "$step.4.egress"},
     ["overall_score", "dimension_scores", "score_report"], False, None, None, None),
    ("rcm_analytics_pipeline", 6, "AG36", "Produce RCM analytics: AR aging, denial rates, clean claim rates, payer performance",
     {"dataset": "$step.1.egress.dataset", "profile_report": "$step.3.egress", "quality_score": "$step.5.egress", "domain_ctx": "$input.dcp"},
     ["analysis_report", "charts", "kpi_summary"], False, None, None, None),

    # ── 2. rcm_denial_management_pipeline: AG04 → AG06 → AG11 → AG14 → AG36 → AG29
    ("rcm_denial_management_pipeline", 1, "AG04", "Ingest denial records and associated claims data",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("rcm_denial_management_pipeline", 2, "AG06", "Initial profiling of denial data structure and patterns",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("rcm_denial_management_pipeline", 3, "AG11", "Create and apply denial validation rules: CARC/RARC code checks, filing limit compliance",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.2.egress", "domain_ctx": "$input.dcp"},
     ["rules", "rule_validation_results"], False, None, None, None),
    ("rcm_denial_management_pipeline", 4, "AG14", "Detect anomalous denial patterns: payer-specific spikes, seasonal trends, coding-related clusters",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["anomaly_report", "outlier_rows"], False, None, None, None),
    ("rcm_denial_management_pipeline", 5, "AG36", "Analyse denial trends: root cause breakdown, recovery opportunity quantification, payer comparison",
     {"dataset": "$step.1.egress.dataset", "rules": "$step.3.egress", "anomaly_report": "$step.4.egress", "domain_ctx": "$input.dcp"},
     ["analysis_report", "charts", "denial_summary"], False, None, None, None),
    ("rcm_denial_management_pipeline", 6, "AG29", "Generate denial prevention and recovery recommendations prioritised by financial impact",
     {"analysis_report": "$step.5.egress", "domain_ctx": "$input.dcp"},
     ["recommendations", "action_plan", "expected_recovery"],
     False, None, "before", "Review denial recovery recommendations before presenting to billing team"),

    # ── 3. care_risk_stratification_pipeline: AG04 → AG06 → AG07 → AG25 → AG14 → AG36 → AG29
    ("care_risk_stratification_pipeline", 1, "AG04", "Ingest patient clinical data: assessments, vitals, medications, encounters",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("care_risk_stratification_pipeline", 2, "AG06", "Initial profiling: identify clinical columns, scoring instruments, temporal fields",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("care_risk_stratification_pipeline", 3, "AG07", "Full profiling: statistical distributions of risk factors, completeness of scoring fields",
     {"dataset": "$step.1.egress.dataset", "column_routing": "$step.2.egress", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "statistical_summary"], False, None, None, None),
    ("care_risk_stratification_pipeline", 4, "AG25", "Enrich patient data with external risk factors: social determinants, medication databases, lab reference ranges",
     {"dataset": "$step.1.egress.dataset", "profile_report": "$step.3.egress", "domain_ctx": "$input.dcp"},
     ["enriched_dataset", "enrichment_log"], False, None, None, None),
    ("care_risk_stratification_pipeline", 5, "AG14", "Detect anomalous risk patterns: outlier scores, unusual deterioration trajectories, data quality flags",
     {"dataset": "$step.4.egress.enriched_dataset", "domain_ctx": "$input.dcp"},
     ["anomaly_report", "outlier_rows"], False, None, None, None),
    ("care_risk_stratification_pipeline", 6, "AG36", "Produce risk stratification analysis: tier distribution, risk factor prevalence, temporal trends",
     {"dataset": "$step.4.egress.enriched_dataset", "profile_report": "$step.3.egress", "anomaly_report": "$step.5.egress", "domain_ctx": "$input.dcp"},
     ["analysis_report", "charts", "risk_stratification_summary"], False, None, None, None),
    ("care_risk_stratification_pipeline", 7, "AG29", "Generate risk-based recommendations: high-risk patient lists, intervention priorities, monitoring protocols",
     {"analysis_report": "$step.6.egress", "domain_ctx": "$input.dcp"},
     ["recommendations", "high_risk_patients", "monitoring_protocol"], False, None, None, None),

    # ── 4. intervention_planning_pipeline: AG04 → AG06 → AG36 → AG29 → AG30
    ("intervention_planning_pipeline", 1, "AG04", "Ingest risk-stratified patient data and intervention history",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("intervention_planning_pipeline", 2, "AG06", "Profile intervention data: success rates, resource utilisation, patient segments",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("intervention_planning_pipeline", 3, "AG36", "Analyse intervention effectiveness: outcomes by type, cost-benefit, risk-tier alignment",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.2.egress", "domain_ctx": "$input.dcp"},
     ["analysis_report", "charts", "effectiveness_summary"], False, None, None, None),
    ("intervention_planning_pipeline", 4, "AG29", "Generate intervention recommendations: new protocols, resource allocation, escalation rules",
     {"analysis_report": "$step.3.egress", "domain_ctx": "$input.dcp"},
     ["recommendations", "intervention_plan", "resource_requirements"], False, None, None, None),
    ("intervention_planning_pipeline", 5, "AG30", "Generate care plan content: patient-facing materials, provider guidelines, monitoring checklists",
     {"recommendations": "$step.4.egress", "domain_ctx": "$input.dcp"},
     ["care_plan_content", "provider_guidelines", "patient_materials"],
     False, None, "before", "Review generated care plan content before distribution to clinical teams"),

    # ── 5. patient_360_pipeline: AG04 → AG06 → AG08 → AG09 → AG23 → AG25 → AG26 → AG27 → AG28
    ("patient_360_pipeline", 1, "AG04", "Ingest multi-source patient data: EHR, ADT, insurance, pharmacy feeds",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("patient_360_pipeline", 2, "AG06", "Profile patient data sources: identify demographic, clinical, and identity fields",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("patient_360_pipeline", 3, "AG08", "Recommend schema mapping to canonical patient model (FHIR Patient)",
     {"customer_schema": "$step.1.egress.metadata", "domain_ctx": "$input.dcp"},
     ["gap_analysis", "recommended_entities", "conformance_score"], False, None, None, None),
    ("patient_360_pipeline", 4, "AG09", "Harmonize patient data to canonical schema: normalize names, addresses, dates",
     {"schema": "$step.1.egress.metadata", "gap_analysis": "$step.3.egress"},
     ["harmonized_schema", "rules_applied"], False, None, None, None),
    ("patient_360_pipeline", 5, "AG23", "Build patient-specific ontology for matching: name variants, address hierarchies, identifier patterns",
     {"customer_data": "$step.1.egress.dataset", "harmonized_schema": "$step.4.egress"},
     ["customer_graph", "entities", "relationships"], False, None, None, None),
    ("patient_360_pipeline", 6, "AG25", "Enrich with external data: address validation, insurance eligibility, death registry",
     {"dataset": "$step.1.egress.dataset", "customer_graph": "$step.5.egress", "domain_ctx": "$input.dcp"},
     ["enriched_dataset", "enrichment_log"], False, None, None, None),
    ("patient_360_pipeline", 7, "AG26", "Enrich with internal data: cross-facility encounter history, medication reconciliation",
     {"dataset": "$step.6.egress.enriched_dataset", "customer_graph": "$step.5.egress"},
     ["enriched_dataset", "internal_enrichment_log"], False, None, None, None),
    ("patient_360_pipeline", 8, "AG27", "Execute entity resolution: deterministic + probabilistic matching, build 360 view",
     {"enriched_dataset": "$step.7.egress.enriched_dataset", "customer_graph": "$step.5.egress", "domain_ctx": "$input.dcp"},
     ["match_results", "patient_360_view", "match_scores"], False, None, None, None),
    ("patient_360_pipeline", 9, "AG28", "Produce golden records: apply survivorship rules, resolve conflicts, calculate confidence scores",
     {"match_results": "$step.8.egress", "domain_ctx": "$input.dcp"},
     ["golden_records", "conflict_log", "confidence_scores"],
     False, None, "before", "Review entity resolution matches and survivorship decisions before finalizing golden records"),

    # ── 6. wellness_analytics_pipeline: AG04 → AG06 → AG07 → AG14 → AG25 → AG36 → AG29
    ("wellness_analytics_pipeline", 1, "AG04", "Ingest patient wellness data: screenings, vaccinations, chronic disease registries, care gaps",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("wellness_analytics_pipeline", 2, "AG06", "Initial profiling: identify screening fields, vaccination records, HEDIS measure columns",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("wellness_analytics_pipeline", 3, "AG07", "Full profiling: compliance rates, gap distributions, chronic disease metric summaries",
     {"dataset": "$step.1.egress.dataset", "column_routing": "$step.2.egress", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "statistical_summary"], False, None, None, None),
    ("wellness_analytics_pipeline", 4, "AG14", "Detect anomalies: unusual gap patterns, compliance drops, vaccination coverage outliers",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["anomaly_report", "outlier_rows"], False, None, None, None),
    ("wellness_analytics_pipeline", 5, "AG25", "Enrich with screening protocols, HEDIS specifications, and USPSTF guidelines for gap calculation",
     {"dataset": "$step.1.egress.dataset", "profile_report": "$step.3.egress", "domain_ctx": "$input.dcp"},
     ["enriched_dataset", "enrichment_log"], False, None, None, None),
    ("wellness_analytics_pipeline", 6, "AG36", "Produce wellness analytics: care gap reports, compliance dashboards, chronic disease trends",
     {"dataset": "$step.5.egress.enriched_dataset", "profile_report": "$step.3.egress", "anomaly_report": "$step.4.egress", "domain_ctx": "$input.dcp"},
     ["analysis_report", "charts", "wellness_summary"], False, None, None, None),
    ("wellness_analytics_pipeline", 7, "AG29", "Generate recommendations: outreach priorities, programme adjustments, high-risk patient lists",
     {"analysis_report": "$step.6.egress", "domain_ctx": "$input.dcp"},
     ["recommendations", "outreach_priorities", "programme_recommendations"], False, None, None, None),
]


# ---------------------------------------------------------------------------
# Dependencies
# format: (plan_name, from_agent, to_agent, from_step, to_step, dep_type, description)
# ---------------------------------------------------------------------------

DEPENDENCIES = [
    # rcm_analytics_pipeline
    ("rcm_analytics_pipeline", "AG04", "AG06", 1, 2, "data", "Profiler needs claims data"),
    ("rcm_analytics_pipeline", "AG06", "AG07", 2, 3, "data", "Full profiler needs column routing"),
    ("rcm_analytics_pipeline", "AG07", "AG14", 3, 4, "data", "Anomaly detector needs profile context"),
    ("rcm_analytics_pipeline", "AG14", "AG17", 4, 5, "data", "Quality scorer needs anomaly report"),
    ("rcm_analytics_pipeline", "AG17", "AG36", 5, 6, "data", "Analytics needs quality context"),

    # rcm_denial_management_pipeline
    ("rcm_denial_management_pipeline", "AG04", "AG06", 1, 2, "data", "Profiler needs denial data"),
    ("rcm_denial_management_pipeline", "AG06", "AG11", 2, 3, "data", "DQ rules need profile"),
    ("rcm_denial_management_pipeline", "AG06", "AG14", 2, 4, "data", "Anomaly detector needs profile"),
    ("rcm_denial_management_pipeline", "AG11", "AG36", 3, 5, "data", "Analysis needs DQ rules"),
    ("rcm_denial_management_pipeline", "AG14", "AG36", 4, 5, "data", "Analysis needs anomaly report"),
    ("rcm_denial_management_pipeline", "AG36", "AG29", 5, 6, "sequence", "Recommendations need analysis"),

    # care_risk_stratification_pipeline
    ("care_risk_stratification_pipeline", "AG04", "AG06", 1, 2, "data", "Profiler needs clinical data"),
    ("care_risk_stratification_pipeline", "AG06", "AG07", 2, 3, "data", "Full profiler needs routing"),
    ("care_risk_stratification_pipeline", "AG07", "AG25", 3, 4, "data", "Enricher needs profile context"),
    ("care_risk_stratification_pipeline", "AG25", "AG14", 4, 5, "data", "Anomaly detector needs enriched data"),
    ("care_risk_stratification_pipeline", "AG14", "AG36", 5, 6, "data", "Analytics needs anomaly context"),
    ("care_risk_stratification_pipeline", "AG36", "AG29", 6, 7, "sequence", "Recommendations need analysis"),

    # intervention_planning_pipeline
    ("intervention_planning_pipeline", "AG04", "AG06", 1, 2, "data", "Profiler needs intervention data"),
    ("intervention_planning_pipeline", "AG06", "AG36", 2, 3, "data", "Analysis needs profile"),
    ("intervention_planning_pipeline", "AG36", "AG29", 3, 4, "sequence", "Recommendations need analysis"),
    ("intervention_planning_pipeline", "AG29", "AG30", 4, 5, "sequence", "Content generator needs recommendations"),

    # patient_360_pipeline
    ("patient_360_pipeline", "AG04", "AG06", 1, 2, "data", "Profiler needs multi-source data"),
    ("patient_360_pipeline", "AG06", "AG08", 2, 3, "data", "Schema recommender needs profile"),
    ("patient_360_pipeline", "AG08", "AG09", 3, 4, "data", "Harmonizer needs schema gaps"),
    ("patient_360_pipeline", "AG09", "AG23", 4, 5, "data", "Ontology builder needs harmonized schema"),
    ("patient_360_pipeline", "AG23", "AG25", 5, 6, "data", "External enricher needs ontology"),
    ("patient_360_pipeline", "AG25", "AG26", 6, 7, "data", "Internal enricher needs external enrichment"),
    ("patient_360_pipeline", "AG26", "AG27", 7, 8, "data", "Entity resolver needs enriched data"),
    ("patient_360_pipeline", "AG27", "AG28", 8, 9, "hitl_gate", "Golden records after HITL review of matches"),

    # wellness_analytics_pipeline
    ("wellness_analytics_pipeline", "AG04", "AG06", 1, 2, "data", "Profiler needs wellness data"),
    ("wellness_analytics_pipeline", "AG06", "AG07", 2, 3, "data", "Full profiler needs routing"),
    ("wellness_analytics_pipeline", "AG07", "AG14", 3, 4, "data", "Anomaly detector needs profile"),
    ("wellness_analytics_pipeline", "AG07", "AG25", 3, 5, "data", "Enricher needs profile for protocol matching"),
    ("wellness_analytics_pipeline", "AG25", "AG36", 5, 6, "data", "Analytics needs enriched data"),
    ("wellness_analytics_pipeline", "AG14", "AG36", 4, 6, "data", "Analytics needs anomaly context"),
    ("wellness_analytics_pipeline", "AG36", "AG29", 6, 7, "sequence", "Recommendations need analysis"),
]


# ---------------------------------------------------------------------------
# Few-shot Examples (one per plan)
# ---------------------------------------------------------------------------

EXAMPLES = [
    # ── 1. RCM Analytics ──────────────────────────────────────────────────
    {
        "example_name": "rcm_claims_performance_analysis",
        "intent_context": {
            "matched_node": {"name": "ar_aging_analysis", "label": "AR Aging Analysis", "level": "specific"},
            "confidence": 0.89,
            "abstract_parents": [{"name": "care_risk_analytics", "label": "Care Risk Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare RCM",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Analyse our claims data to show AR aging trends, denial rates by payer, and identify the biggest revenue recovery opportunities.",
        },
        "selected_plan": "rcm_analytics_pipeline",
        "reasoning": "User wants comprehensive RCM analytics including AR aging and denial rates. rcm_analytics_pipeline: AG04 (ingest claims) -> AG06 (initial profile) -> AG07 (full profile) -> AG14 (anomaly detection for billing irregularities) -> AG17 (quality score) -> AG36 (produce RCM dashboards). Direct match to revenue cycle analytics intent.",
        "plan_overrides": None,
    },
    # ── 2. Denial Management ──────────────────────────────────────────────
    {
        "example_name": "denial_root_cause_analysis",
        "intent_context": {
            "matched_node": {"name": "denial_analysis", "label": "Denial Analysis", "level": "specific"},
            "confidence": 0.93,
            "abstract_parents": [{"name": "care_risk_analytics", "label": "Care Risk Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare RCM",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Our denial rate spiked last quarter. Analyse the denial data, find root causes by CARC code, and recommend recovery actions.",
        },
        "selected_plan": "rcm_denial_management_pipeline",
        "reasoning": "User needs denial-specific analysis with root cause and recovery. rcm_denial_management_pipeline: AG04 (ingest denials) -> AG06 (profile) -> AG11 (DQ rules for CARC/RARC validation) -> AG14 (detect anomalous denial patterns) -> AG36 (denial trend analysis) -> AG29 (recovery recommendations). HITL gate before AG29 ensures billing team reviews recommendations. Chosen over generic rcm_analytics_pipeline because user specifically targets denial root causes.",
        "plan_overrides": None,
    },
    # ── 3. Care Risk Stratification ───────────────────────────────────────
    {
        "example_name": "patient_fall_risk_stratification",
        "intent_context": {
            "matched_node": {"name": "risk_stratification", "label": "Risk Stratification", "level": "specific"},
            "confidence": 0.91,
            "abstract_parents": [{"name": "care_risk_analytics", "label": "Care Risk Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "We have patient assessment data with Morse fall scores and medication lists. Stratify patients by fall risk and identify who needs immediate intervention.",
        },
        "selected_plan": "care_risk_stratification_pipeline",
        "reasoning": "User wants risk stratification from clinical assessment data. care_risk_stratification_pipeline: AG04 (ingest clinical data) -> AG06 (profile) -> AG07 (full profile of risk factors) -> AG25 (enrich with medication risk databases) -> AG14 (detect anomalous risk patterns) -> AG36 (produce risk tier analysis) -> AG29 (generate intervention priorities). Enrichment step (AG25) adds medication fall risk flags. Direct match on risk_stratification intent.",
        "plan_overrides": None,
    },
    # ── 4. Intervention Planning ──────────────────────────────────────────
    {
        "example_name": "readmission_prevention_plan",
        "intent_context": {
            "matched_node": {"name": "intervention_planning", "label": "Intervention Planning", "level": "specific"},
            "confidence": 0.88,
            "abstract_parents": [{"name": "care_risk_analytics", "label": "Care Risk Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Based on our risk-stratified patient data, design intervention plans for high-risk readmission patients. Generate care plan documents for the nursing team.",
        },
        "selected_plan": "intervention_planning_pipeline",
        "reasoning": "User wants intervention plans from pre-stratified data plus care plan content. intervention_planning_pipeline: AG04 (ingest stratified data) -> AG06 (profile interventions) -> AG36 (analyse effectiveness) -> AG29 (generate intervention recommendations) -> AG30 (generate care plan content). HITL gate before AG30 ensures clinical team reviews content before distribution. Chosen over risk stratification pipeline because input is already stratified.",
        "plan_overrides": None,
    },
    # ── 5. Patient 360 ────────────────────────────────────────────────────
    {
        "example_name": "multi_facility_patient_matching",
        "intent_context": {
            "matched_node": {"name": "patient_entity_resolution", "label": "Patient Entity Resolution", "level": "specific"},
            "confidence": 0.94,
            "abstract_parents": [{"name": "patient_master_data", "label": "Patient Master Data"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "We have patient records from 3 hospital facilities and 2 clinics. Build a unified patient view with a golden record for each unique individual.",
        },
        "selected_plan": "patient_360_pipeline",
        "reasoning": "User needs cross-facility entity resolution and golden records. patient_360_pipeline: AG04 (ingest multi-source) -> AG06 (profile) -> AG08 (recommend FHIR schema) -> AG09 (harmonize) -> AG23 (patient matching ontology) -> AG25 (external enrichment: address validation, death registry) -> AG26 (internal: cross-facility encounter linking) -> AG27 (entity resolution: deterministic + probabilistic) -> AG28 (golden record construction). HITL gate before AG28 for stewardship review of match decisions. 9-step pipeline appropriate for multi-source MDM.",
        "plan_overrides": None,
    },
    # ── 6. Wellness Analytics ─────────────────────────────────────────────
    {
        "example_name": "hedis_compliance_dashboard",
        "intent_context": {
            "matched_node": {"name": "screening_compliance", "label": "Screening Compliance", "level": "specific"},
            "confidence": 0.90,
            "abstract_parents": [{"name": "wellness_analytics", "label": "Wellness Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Generate a HEDIS compliance dashboard showing screening rates for mammography, colonoscopy, and A1C testing. Identify care gaps and prioritise outreach.",
        },
        "selected_plan": "wellness_analytics_pipeline",
        "reasoning": "User wants HEDIS compliance analytics with care gap identification. wellness_analytics_pipeline: AG04 (ingest wellness data) -> AG06 (profile) -> AG07 (full profile: compliance rates) -> AG14 (detect compliance anomalies) -> AG25 (enrich with HEDIS specs and USPSTF guidelines) -> AG36 (produce compliance dashboards and care gap reports) -> AG29 (generate outreach recommendations). Enrichment with HEDIS specifications (AG25) enables accurate gap calculation. Direct match on screening_compliance intent.",
        "plan_overrides": None,
    },
]


# ---------------------------------------------------------------------------
# Main seed function
# ---------------------------------------------------------------------------

def seed() -> None:
    log("Connecting to domain DB...")
    conn = get_execution_conn()

    log("Initializing execution graph schema...")
    init_execution_db(conn)

    # ── Plans ──────────────────────────────────────────────────────────────
    log(f"\nUpserting {len(PLANS)} CareAXL execution plans...")
    for plan in PLANS:
        pid = upsert_execution_plan(conn, **plan)
        log(f"  {plan['name']}: {plan['label']} (id={pid}, app={plan['app_id']}, sol={plan['solution_id']})")
    conn.commit()

    # ── Steps ──────────────────────────────────────────────────────────────
    log(f"\nAdding {len(STEPS)} execution steps...")
    step_count = 0
    for plan_name, step_order, agent_id, purpose, ingress_map, egress_keys, is_optional, condition, hitl_gate, hitl_reason in STEPS:
        try:
            add_execution_step(
                conn, plan_name, step_order, agent_id, purpose,
                ingress_map=ingress_map, egress_keys=egress_keys,
                is_optional=is_optional, condition=condition,
                hitl_gate=hitl_gate, hitl_reason=hitl_reason,
            )
            step_count += 1
        except (ValueError, Exception) as e:
            log(f"  WARN: step ({plan_name}, {agent_id}, order={step_order}): {e}")
    conn.commit()
    log(f"  {step_count} steps inserted.")

    # ── Dependencies ───────────────────────────────────────────────────────
    log(f"\nAdding {len(DEPENDENCIES)} dependencies...")
    dep_count = 0
    for plan_name, from_aid, to_aid, from_order, to_order, dep_type, desc in DEPENDENCIES:
        try:
            add_execution_dependency(
                conn, plan_name, from_aid, to_aid,
                from_step_order=from_order, to_step_order=to_order,
                dep_type=dep_type, description=desc,
            )
            dep_count += 1
        except (ValueError, Exception) as e:
            log(f"  WARN: dep ({plan_name}, {from_aid}->{to_aid}): {e}")
    conn.commit()
    log(f"  {dep_count} dependencies inserted.")

    # ── Examples ───────────────────────────────────────────────────────────
    log(f"\nAdding {len(EXAMPLES)} few-shot examples...")
    ex_count = 0
    for ex in EXAMPLES:
        try:
            add_execution_example(conn, **ex)
            ex_count += 1
        except Exception as e:
            log(f"  WARN: example {ex['example_name']}: {e}")
    conn.commit()
    log(f"  {ex_count} examples inserted.")

    # ── Embeddings ─────────────────────────────────────────────────────────
    log("\nBackfilling embeddings...")
    try:
        count = backfill_execution_embeddings(conn)
        log(f"  Embedded {count} items.")
    except Exception as e:
        log(f"  WARN: embedding backfill: {e}")

    conn.close()
    log("\nCareAXL execution seed complete.")


if __name__ == "__main__":
    seed()
