"""
seed_pipeline_gaps.py — Fill intent graph and execution graph gaps identified
by the 10-scenario DESO pipeline test.

Adds:
  Intent nodes   : 11 new (6 Healthcare, 2 Wellness, 1 Commercial, 2 mid-level)
  Intent triplets: 20+ new connections
  Intent aliases : 40+ new aliases for discoverability
  Exec plans     : 3 new plan templates
  Exec steps     : ~20 new steps
  Exec deps      : ~15 new dependencies
  Few-shot examples: 5 new scenario-matched examples
  Intent pattern updates: on 3 existing plans

Gaps addressed:
  Scenario 1  — Product catalog enrichment from OEM data sheets
  Scenario 2  — Field extraction (add intent pattern to content_generation plan)
  Scenario 4  — Customer DQ + dedup (add intent pattern to mdm_entity_resolution)
  Scenario 6  — Payer defer/delay/deny behavior analysis
  Scenario 7  — Revenue leakage detection (Home Care visits)
  Scenario 8  — Readmission prediction (wire to new predictive plan)
  Scenario 9  — Mortality prediction (wire to new predictive plan)
  Scenario 10 — Wellness customer engagement / fitness suggestions

Run:
    python hooks/seed_pipeline_gaps.py
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intent_graph import (
    get_intent_conn,
    init_intent_db,
    upsert_intent_node,
    add_intent_triplet,
    add_intent_alias,
    embed_intent_nodes,
)
from execution_graph import (
    get_execution_conn,
    init_execution_db,
    upsert_execution_plan,
    add_execution_step,
    add_execution_dependency,
    add_execution_example,
    backfill_execution_embeddings,
)


# ==========================================================================
# PART 1: New Intent Nodes
# ==========================================================================
# format: (name, label, description, level, domain, lifecycle_stage)

NEW_INTENT_NODES = [

    # ── Abstract: Healthcare Operations (new parent for RCM + Home Care) ──
    (
        "healthcare_operations_analytics",
        "Healthcare Operations Analytics",
        "Analyze operational, financial, and compliance data across healthcare "
        "service lines to detect revenue leakage, payer behavior patterns, "
        "billing discrepancies, and operational inefficiencies.",
        "abstract", "Healthcare", "analytics",
    ),

    # ── Mid: Payer Analytics ─────────────────────────────────────────────
    (
        "payer_behavior_analytics",
        "Payer Behavior Analytics",
        "Analyze payer claim processing patterns including denial rates, "
        "appeal outcomes, authorization delays, and payment timeliness to "
        "detect systematic defer-delay-deny strategies.",
        "mid", "Healthcare", "analytics",
    ),

    # ── Mid: Revenue Cycle Analytics ─────────────────────────────────────
    (
        "revenue_cycle_analytics",
        "Revenue Cycle Analytics",
        "Detect and quantify revenue leakage across the care delivery and "
        "billing lifecycle, including missed visits, unbilled services, "
        "documentation gaps, and reimbursement shortfalls.",
        "mid", "Healthcare", "analytics",
    ),

    # ── Mid: Wellness & Engagement ───────────────────────────────────────
    (
        "wellness_engagement",
        "Wellness & Engagement",
        "Drive personalized health and wellness engagement through fitness "
        "recommendations, diet suggestions, adherence tracking, and "
        "behavioral nudging based on user activity and health data.",
        "mid", "Wellness", "recommendation",
    ),

    # ── Mid: Product Catalog Management ──────────────────────────────────
    (
        "product_catalog_management",
        "Product Catalog Management",
        "Improve, enrich, and maintain product catalogs by extracting "
        "attributes from descriptions, sourcing data from OEM/manufacturer "
        "data sheets, and ensuring schema completeness.",
        "mid", "Commercial / Industrial", "enrichment",
    ),

    # ── Specific: Payer deny/defer/delay analysis ────────────────────────
    (
        "payer_deny_defer_analysis",
        "Payer Deny/Defer/Delay Analysis",
        "Identify systematic payer behavior patterns of claim denial, "
        "deferral, and delay. Analyze denial reason codes, turnaround "
        "times, appeal success rates, and payer-specific tendencies.",
        "specific", "Healthcare", "analytics",
    ),
    (
        "claim_appeal_analytics",
        "Claim Appeal Analytics",
        "Analyze claim appeal workflows: success rates by denial reason, "
        "payer, service type; optimal appeal timing; revenue recovered "
        "through appeals vs. write-offs.",
        "specific", "Healthcare", "analytics",
    ),

    # ── Specific: Revenue leakage ────────────────────────────────────────
    (
        "visit_leakage_analysis",
        "Visit Leakage Analysis",
        "Detect revenue leakage from missed or unbilled home care visits, "
        "visit sync failures between scheduling and billing systems, and "
        "Face-to-Face (F2F) encounter documentation gaps.",
        "specific", "Healthcare", "analytics",
    ),
    (
        "service_line_profitability",
        "Service Line Profitability Analysis",
        "Analyze profitability at the service line level (Home Care, Hospice, "
        "PCS, etc.) to identify margin erosion, cost drivers, and "
        "reimbursement gaps by payer mix.",
        "specific", "Healthcare", "analytics",
    ),

    # ── Specific: Wellness engagement ────────────────────────────────────
    (
        "fitness_diet_recommendation",
        "Fitness & Diet Recommendation",
        "Generate personalized daily fitness routines and diet plans based on "
        "user health profile, activity history, goals, and preferences. "
        "Combine recommendation engine with content generation.",
        "specific", "Wellness", "recommendation",
    ),
    (
        "engagement_scoring",
        "Customer Engagement Scoring",
        "Score and predict customer engagement levels using activity data, "
        "interaction history, and behavioral patterns to identify at-risk "
        "users and optimal intervention points.",
        "specific", "Wellness", "analytics",
    ),

    # ── Specific: Product catalog enrichment from OEM ────────────────────
    (
        "product_catalog_enrichment",
        "Product Catalog Enrichment",
        "Enrich product catalog entries (e.g. truck parts, industrial "
        "components) using OEM manufacturer data sheets, technical "
        "specifications, and cross-reference databases.",
        "specific", "Commercial / Industrial", "enrichment",
    ),
]


# ==========================================================================
# PART 2: New Triplets
# ==========================================================================
# format: (subject, predicate, object)

NEW_TRIPLETS = [
    # Hierarchy
    ("healthcare_operations_analytics", "IS_A", "exploratory_analytics"),
    ("payer_behavior_analytics", "IS_A", "healthcare_operations_analytics"),
    ("revenue_cycle_analytics", "IS_A", "healthcare_operations_analytics"),
    ("payer_deny_defer_analysis", "IS_A", "payer_behavior_analytics"),
    ("claim_appeal_analytics", "IS_A", "payer_behavior_analytics"),
    ("visit_leakage_analysis", "IS_A", "revenue_cycle_analytics"),
    ("service_line_profitability", "IS_A", "revenue_cycle_analytics"),

    # Wellness hierarchy
    ("wellness_engagement", "IS_A", "revenue_optimization"),
    ("fitness_diet_recommendation", "IS_A", "wellness_engagement"),
    ("engagement_scoring", "IS_A", "wellness_engagement"),

    # Product catalog hierarchy
    ("product_catalog_management", "IS_A", "parts_and_product_management"),
    ("product_catalog_enrichment", "IS_A", "product_catalog_management"),
    ("product_attribute_extraction", "IS_A", "product_catalog_management"),

    # Cross-relationships
    ("payer_deny_defer_analysis", "REQUIRES", "data_profiling"),
    ("payer_deny_defer_analysis", "ENABLES", "claim_appeal_analytics"),
    ("visit_leakage_analysis", "REQUIRES", "data_profiling"),
    ("visit_leakage_analysis", "ENABLES", "service_line_profitability"),
    ("product_catalog_enrichment", "REQUIRES", "product_attribute_extraction"),
    ("product_catalog_enrichment", "REQUIRES", "data_profiling"),
    ("fitness_diet_recommendation", "REQUIRES", "engagement_scoring"),
    ("engagement_scoring", "REQUIRES", "data_profiling"),

    # Connect existing healthcare intents to new structure
    ("claim_denial_management", "IS_A", "payer_behavior_analytics"),
    ("reimbursement_optimization", "IS_A", "revenue_cycle_analytics"),

    # Readmission/mortality — add REQUIRES for analytics plan bridge
    ("readmission_prediction", "REQUIRES", "data_profiling"),
    ("readmission_prediction", "REQUIRES", "anomaly_detection"),
    ("mortality_prediction", "REQUIRES", "data_profiling"),
    ("mortality_prediction", "REQUIRES", "anomaly_detection"),
]


# ==========================================================================
# PART 3: New Aliases
# ==========================================================================
# format: (intent_name, alias, alias_type)

NEW_ALIASES = [
    # Payer behavior
    ("payer_deny_defer_analysis", "payer behavior", "phrase"),
    ("payer_deny_defer_analysis", "defer delay deny", "phrase"),
    ("payer_deny_defer_analysis", "denial patterns", "phrase"),
    ("payer_deny_defer_analysis", "payer deny", "keyword"),
    ("payer_deny_defer_analysis", "claim deferral", "keyword"),
    ("payer_deny_defer_analysis", "denial tendency", "phrase"),
    ("payer_deny_defer_analysis", "systematic denial", "phrase"),
    ("payer_deny_defer_analysis", "payer turnaround", "phrase"),
    ("payer_behavior_analytics", "payer analytics", "phrase"),
    ("payer_behavior_analytics", "payer performance", "phrase"),
    ("claim_appeal_analytics", "appeal success", "phrase"),
    ("claim_appeal_analytics", "appeal rate", "phrase"),
    ("claim_appeal_analytics", "denial overturn", "phrase"),

    # Revenue leakage
    ("visit_leakage_analysis", "revenue leakage", "phrase"),
    ("visit_leakage_analysis", "visit sync", "keyword"),
    ("visit_leakage_analysis", "F2F", "keyword"),
    ("visit_leakage_analysis", "face to face", "phrase"),
    ("visit_leakage_analysis", "missed visits", "phrase"),
    ("visit_leakage_analysis", "unbilled visits", "phrase"),
    ("visit_leakage_analysis", "home care leakage", "phrase"),
    ("visit_leakage_analysis", "visit reconciliation", "phrase"),
    ("revenue_cycle_analytics", "revenue cycle", "phrase"),
    ("revenue_cycle_analytics", "billing leakage", "phrase"),
    ("service_line_profitability", "service line profit", "phrase"),
    ("service_line_profitability", "margin erosion", "phrase"),

    # Wellness
    ("fitness_diet_recommendation", "fitness suggestion", "phrase"),
    ("fitness_diet_recommendation", "diet recommendation", "phrase"),
    ("fitness_diet_recommendation", "daily fitness", "phrase"),
    ("fitness_diet_recommendation", "daily diet", "phrase"),
    ("fitness_diet_recommendation", "wellness recommendation", "phrase"),
    ("fitness_diet_recommendation", "health suggestion", "phrase"),
    ("wellness_engagement", "customer engagement", "phrase"),
    ("wellness_engagement", "user engagement", "phrase"),
    ("wellness_engagement", "wellness engagement", "phrase"),
    ("engagement_scoring", "engagement score", "phrase"),
    ("engagement_scoring", "engagement prediction", "phrase"),
    ("engagement_scoring", "activity scoring", "phrase"),

    # Product catalog enrichment
    ("product_catalog_enrichment", "catalog enrichment", "phrase"),
    ("product_catalog_enrichment", "OEM data", "keyword"),
    ("product_catalog_enrichment", "manufacturer data sheet", "phrase"),
    ("product_catalog_enrichment", "improve product catalog", "phrase"),
    ("product_catalog_enrichment", "enrich catalog", "phrase"),
    ("product_catalog_enrichment", "truck parts catalog", "phrase"),
    ("product_catalog_enrichment", "technical specifications", "phrase"),
    ("product_catalog_management", "catalog management", "phrase"),
    ("product_catalog_management", "product catalog", "phrase"),

    # Strengthen existing readmission/mortality aliases
    ("readmission_prediction", "PCS readmission", "phrase"),
    ("readmission_prediction", "30 day readmission", "phrase"),
    ("readmission_prediction", "readmitted patients", "phrase"),
    ("readmission_prediction", "identify readmission risk", "phrase"),
    ("mortality_prediction", "mitigate mortality", "phrase"),
    ("mortality_prediction", "prevent mortality", "phrase"),
    ("mortality_prediction", "hospice mortality", "phrase"),
    ("mortality_prediction", "mortality risk hospice", "phrase"),
]


# ==========================================================================
# PART 4: New Execution Plan Templates
# ==========================================================================

NEW_PLANS = [
    # 17. Predictive Analytics Pipeline
    {
        "name": "predictive_analytics_pipeline",
        "label": "Predictive Analytics Pipeline",
        "description": (
            "Build predictive models from profiled data: fetch, profile, "
            "detect anomalies, assess data fitness for ML, then run advanced "
            "analytics (classification, regression, survival). Includes HITL "
            "gate before model deployment."
        ),
        "intent_patterns": [
            "readmission_prediction", "mortality_prediction", "churn_prediction",
            "binary_classification", "regression_modeling", "batch_scoring",
            "sepsis_prediction", "fall_risk_assessment", "length_of_stay_prediction",
        ],
        "lifecycle_stages": ["ml_modeling", "analytics"],
        "domains": ["Healthcare", "Any"],
        "requires_dataset": True,
    },
    # 18. Product Catalog Enrichment
    {
        "name": "product_catalog_enrichment",
        "label": "Product Catalog Enrichment Pipeline",
        "description": (
            "Enrich product catalogs using OEM/manufacturer data sheets and "
            "external reference data. Extracts attributes from unstructured "
            "descriptions, builds domain ontology, enriches via external "
            "sources, generates optimized content, and scores quality."
        ),
        "intent_patterns": [
            "product_catalog_enrichment", "product_attribute_extraction",
            "cross_reference_lookup", "product_description_generation",
            "product_catalog_management",
        ],
        "lifecycle_stages": ["enrichment", "preparation"],
        "domains": ["Commercial / Industrial"],
        "requires_dataset": True,
    },
    # 19. Healthcare Operational Analytics
    {
        "name": "healthcare_operational_analytics",
        "label": "Healthcare Operational Analytics Pipeline",
        "description": (
            "Analyze healthcare operational and financial data to detect "
            "payer behavior patterns (deny/defer/delay), revenue leakage, "
            "billing discrepancies, and service line profitability issues. "
            "Combines profiling, anomaly detection, and advanced analytics."
        ),
        "intent_patterns": [
            "payer_deny_defer_analysis", "claim_appeal_analytics",
            "visit_leakage_analysis", "service_line_profitability",
            "payer_behavior_analytics", "revenue_cycle_analytics",
            "claim_denial_management", "reimbursement_optimization",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Healthcare"],
        "requires_dataset": True,
    },
]


# Steps for new plans
# format: (plan_name, step_order, agent_id, purpose, ingress_map,
#          egress_keys, is_optional, condition, hitl_gate, hitl_reason)

NEW_STEPS = [
    # ── 17. predictive_analytics_pipeline ────────────────────────────────
    # AG04 → AG07 || AG24 → AG14 → AG36 → AG20 → AG01
    ("predictive_analytics_pipeline", 1, "AG04",
     "Fetch patient/entity data from source systems",
     {"source_config": "$input.source_config"},
     ["dataset", "metadata"], False, None, None, None),

    ("predictive_analytics_pipeline", 2, "AG07",
     "Full profiling: distributions, missingness, class balance",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "statistical_summary"],
     False, None, None, None),

    ("predictive_analytics_pipeline", 2, "AG24",
     "Build domain ontology for feature context (e.g. ICD codes, clinical terms)",
     {"domain": "$input.dcp.domain"},
     ["core_graph", "entities", "terminology"],
     True, "When domain graph adds value", None, None),

    ("predictive_analytics_pipeline", 3, "AG14",
     "Detect anomalies and outliers that could bias models",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["anomaly_report", "outlier_rows"],
     False, None, None, None),

    ("predictive_analytics_pipeline", 4, "AG36",
     "Build predictive model: feature engineering, train, evaluate, explain",
     {"dataset": "$step.1.egress.dataset", "profile": "$step.2.egress.profile_report",
      "anomaly_report": "$step.3.egress.anomaly_report",
      "business_objective": "$input.business_objective"},
     ["model", "evaluation_metrics", "feature_importance", "predictions"],
     False, None, None, None),

    ("predictive_analytics_pipeline", 5, "AG20",
     "Assess data fitness for the predictive use case and flag gaps",
     {"dataset": "$step.1.egress.dataset", "profiling": "$step.2.egress",
      "model_metrics": "$step.4.egress.evaluation_metrics",
      "business_objective": "$input.business_objective"},
     ["fit_for_use_report", "readiness_level", "remediation_plan"],
     False, None, None, None),

    ("predictive_analytics_pipeline", 6, "AG01",
     "Evaluate model outputs, explain predictions, validate against domain rules",
     {"model_output": "$step.4.egress", "domain_ctx": "$input.dcp",
      "fitness_report": "$step.5.egress"},
     ["evaluation_report", "recommendations"],
     True, "Final quality gate", None, None),

    # ── 18. product_catalog_enrichment ───────────────────────────────────
    # AG04 → AG07 || AG24 → AG25 → AG30 || AG31 → AG17
    ("product_catalog_enrichment", 1, "AG04",
     "Fetch product catalog data and OEM/manufacturer data sheets",
     {"source_config": "$input.source_config", "reference_sources": "$input.oem_sources"},
     ["dataset", "metadata", "reference_data"], False, None, None, None),

    ("product_catalog_enrichment", 2, "AG07",
     "Profile catalog: identify sparse attributes, data types, completeness gaps",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles"],
     False, None, None, None),

    ("product_catalog_enrichment", 2, "AG24",
     "Build product domain ontology (taxonomy, technical attributes, classification)",
     {"domain": "$input.dcp.domain", "reference_data": "$step.1.egress.reference_data"},
     ["core_graph", "entities", "terminology"],
     False, None, None, None),

    ("product_catalog_enrichment", 3, "AG25",
     "Enrich catalog from OEM data sheets and external APIs (cross-reference, specs)",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp",
      "reference_data": "$step.1.egress.reference_data",
      "enrichment_config": "$input.external_config"},
     ["enriched_dataset", "enrichment_log", "enrichment_rate"],
     False, None, None, None),

    ("product_catalog_enrichment", 4, "AG30",
     "Generate standardized product descriptions from enriched attributes",
     {"dataset": "$step.3.egress.enriched_dataset", "domain_ctx": "$input.dcp",
      "ontology": "$step.2.AG24.egress.core_graph"},
     ["content_dataset", "generation_log"],
     False, None, None, None),

    ("product_catalog_enrichment", 4, "AG31",
     "Optimize descriptions for search, discoverability, and technical accuracy",
     {"content": "$step.4.AG30.egress.content_dataset",
      "optimization_goals": "$input.content_goals"},
     ["optimized_content", "optimization_report"],
     True, "If SEO or search optimization needed", None, None),

    ("product_catalog_enrichment", 5, "AG17",
     "Score final catalog quality: completeness, accuracy, consistency",
     {"dataset": "$step.3.egress.enriched_dataset",
      "profiling_output": "$step.2.AG07.egress"},
     ["overall_score", "dimension_scores", "score_report"],
     False, None, None, None),

    # ── 19. healthcare_operational_analytics ──────────────────────────────
    # AG04 → AG07 → AG14 → AG36 → AG01
    ("healthcare_operational_analytics", 1, "AG04",
     "Fetch claims, billing, visit, and authorization data",
     {"source_config": "$input.source_config"},
     ["dataset", "metadata"], False, None, None, None),

    ("healthcare_operational_analytics", 2, "AG07",
     "Profile operational data: volumes, turnaround distributions, payer mix",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "statistical_summary"],
     False, None, None, None),

    ("healthcare_operational_analytics", 3, "AG14",
     "Detect anomalous patterns: denial spikes, payment delays, visit sync gaps",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp",
      "profile": "$step.2.egress.profile_report"},
     ["anomaly_report", "outlier_rows", "pattern_clusters"],
     False, None, None, None),

    ("healthcare_operational_analytics", 4, "AG36",
     "Analyze payer behavior, revenue leakage, or service line profitability",
     {"dataset": "$step.1.egress.dataset", "profile": "$step.2.egress",
      "anomaly_report": "$step.3.egress", "analysis_focus": "$input.analysis_focus"},
     ["analysis_report", "metrics", "visualizations", "insights"],
     False, None, None, None),

    ("healthcare_operational_analytics", 5, "AG01",
     "Evaluate findings, validate against domain rules, produce actionable report",
     {"analysis_output": "$step.4.egress", "domain_ctx": "$input.dcp"},
     ["evaluation_report", "recommendations", "action_items"],
     True, "Optional final review", None, None),
]


# Dependencies for new plans
# format: (plan_name, from_agent_id, to_agent_id,
#          from_step_order, to_step_order, dep_type, description)

NEW_DEPENDENCIES = [
    # predictive_analytics_pipeline
    ("predictive_analytics_pipeline", "AG04", "AG07", 1, 2, "data", "Profiler needs fetched data"),
    ("predictive_analytics_pipeline", "AG04", "AG24", 1, 2, "data", "Ontology builder needs domain context"),
    ("predictive_analytics_pipeline", "AG07", "AG14", 2, 3, "data", "Anomaly detection needs profiling"),
    ("predictive_analytics_pipeline", "AG14", "AG36", 3, 4, "data", "Analytics needs clean data + anomaly flags"),
    ("predictive_analytics_pipeline", "AG07", "AG36", 2, 4, "data", "Analytics needs profiling for features"),
    ("predictive_analytics_pipeline", "AG36", "AG20", 4, 5, "data", "Fit-for-use needs model metrics"),
    ("predictive_analytics_pipeline", "AG20", "AG01", 5, 6, "sequence", "Evaluator reviews after fitness assessment"),

    # product_catalog_enrichment
    ("product_catalog_enrichment", "AG04", "AG07", 1, 2, "data", "Profiler needs catalog data"),
    ("product_catalog_enrichment", "AG04", "AG24", 1, 2, "data", "Ontology needs reference data"),
    ("product_catalog_enrichment", "AG07", "AG25", 2, 3, "data", "Enricher needs profiling to target gaps"),
    ("product_catalog_enrichment", "AG24", "AG25", 2, 3, "data", "Enricher needs ontology for attribute mapping"),
    ("product_catalog_enrichment", "AG25", "AG30", 3, 4, "data", "Content gen needs enriched data"),
    ("product_catalog_enrichment", "AG24", "AG30", 2, 4, "data", "Content gen needs ontology for descriptions"),
    ("product_catalog_enrichment", "AG30", "AG31", 4, 4, "data", "Optimizer refines generated content"),
    ("product_catalog_enrichment", "AG25", "AG17", 3, 5, "data", "Scorer needs enriched dataset"),

    # healthcare_operational_analytics
    ("healthcare_operational_analytics", "AG04", "AG07", 1, 2, "data", "Profiler needs operational data"),
    ("healthcare_operational_analytics", "AG07", "AG14", 2, 3, "data", "Anomaly detection needs profiling"),
    ("healthcare_operational_analytics", "AG14", "AG36", 3, 4, "data", "Analytics needs anomaly context"),
    ("healthcare_operational_analytics", "AG07", "AG36", 2, 4, "data", "Analytics needs profiling"),
    ("healthcare_operational_analytics", "AG36", "AG01", 4, 5, "sequence", "Evaluator reviews analysis findings"),
]


# ==========================================================================
# PART 5: New Few-Shot Examples for the Execution Graph
# ==========================================================================

NEW_EXAMPLES = [
    # Ex 21: Scenario 1 — Truck parts catalog enrichment from OEM
    {
        "example_name": "truck_parts_catalog_enrichment",
        "intent_context": {
            "matched_node": {
                "name": "product_catalog_enrichment",
                "label": "Product Catalog Enrichment",
                "level": "specific",
                "lifecycle_stage": "enrichment",
            },
            "confidence": 0.72,
            "abstract_parents": ["product_catalog_management", "parts_and_product_management"],
            "specific_children": [],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Commercial / Industrial",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "I want to improve my product catalog on TRUCK PARTS by referring OEM / Manufacturer's data sheets",
        },
        "selected_plan": "product_catalog_enrichment",
        "reasoning": (
            "User wants to enrich an existing truck parts catalog using OEM data sheets. "
            "This is a classic product catalog enrichment workflow: fetch catalog + OEM references, "
            "profile to find attribute gaps, build product ontology for truck parts taxonomy, "
            "enrich from external OEM sources, generate/optimize descriptions, and score quality. "
            "Full product_catalog_enrichment plan applies directly."
        ),
        "plan_overrides": None,
    },

    # Ex 22: Scenario 6 — Payer defer/delay/deny
    {
        "example_name": "payer_deny_defer_delay_analysis",
        "intent_context": {
            "matched_node": {
                "name": "payer_deny_defer_analysis",
                "label": "Payer Deny/Defer/Delay Analysis",
                "level": "specific",
                "lifecycle_stage": "analytics",
            },
            "confidence": 0.75,
            "abstract_parents": ["payer_behavior_analytics", "healthcare_operations_analytics"],
            "specific_children": [],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "I want to understand Payer behavior and see if there are deliberate defer, delay, deny tendencies",
        },
        "selected_plan": "healthcare_operational_analytics",
        "reasoning": (
            "Payer defer/delay/deny analysis is a healthcare operational analytics use case. "
            "The pipeline needs to: (1) fetch claims + authorization data, (2) profile to understand "
            "payer mix, turnaround distributions, and denial rates, (3) detect anomalous patterns "
            "(denial spikes, systematic delays by payer), (4) run focused analytics on payer behavior "
            "trends and denial reason codes. AG14 (AnomalyDetector) is critical for finding "
            "systematic patterns. AG36 (DataAnalysis) provides the trend analysis and payer comparison."
        ),
        "plan_overrides": None,
    },

    # Ex 23: Scenario 7 — Revenue leakage Home Care
    {
        "example_name": "home_care_revenue_leakage",
        "intent_context": {
            "matched_node": {
                "name": "visit_leakage_analysis",
                "label": "Visit Leakage Analysis",
                "level": "specific",
                "lifecycle_stage": "analytics",
            },
            "confidence": 0.73,
            "abstract_parents": ["revenue_cycle_analytics", "healthcare_operations_analytics"],
            "specific_children": [],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "I want to plug revenue leakage on Home Care particularly on Visit Syncs and F2F",
        },
        "selected_plan": "healthcare_operational_analytics",
        "reasoning": (
            "Revenue leakage on visit syncs and F2F is a healthcare operational analytics problem. "
            "The pipeline needs: (1) fetch visit scheduling, billing, and F2F encounter data, "
            "(2) profile to understand visit volumes and billing rates, (3) detect anomalies — "
            "gaps between scheduled and billed visits, missing F2F documentation, sync failures, "
            "(4) quantify leakage by service line, payer, and visit type. AG14 finds the sync "
            "discrepancies; AG36 quantifies the revenue impact."
        ),
        "plan_overrides": None,
    },

    # Ex 24: Scenario 8 — PCS 30-day readmission prediction
    {
        "example_name": "pcs_readmission_prediction_30day",
        "intent_context": {
            "matched_node": {
                "name": "readmission_prediction",
                "label": "Readmission Prediction",
                "level": "specific",
                "lifecycle_stage": "ml_modeling",
            },
            "confidence": 0.78,
            "abstract_parents": ["patient_risk_management", "predictive_analytics"],
            "specific_children": [],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "How can I identify PCS patients most likely to be readmitted in next 30 days",
            "business_objective": "30-day PCS readmission risk scoring",
        },
        "selected_plan": "predictive_analytics_pipeline",
        "reasoning": (
            "This is a classic predictive analytics use case: build a 30-day readmission risk model "
            "for PCS patients. The pipeline: (1) fetch patient demographics, clinical, and encounter "
            "data, (2) profile to understand class balance and feature distributions — profiling and "
            "ontology run parallel (ontology helps with ICD code interpretation), (3) detect anomalies "
            "that could bias the model (outlier LOS, coding errors), (4) AG36 builds the classification "
            "model (binary: readmitted Y/N within 30 days), (5) AG20 assesses whether data quality "
            "supports reliable predictions, (6) AG01 evaluates model output. Clinical domain ontology "
            "(AG24) adds value for ICD mapping."
        ),
        "plan_overrides": None,
    },

    # Ex 25: Scenario 10 — Wellness fitness/diet engagement
    {
        "example_name": "wellness_fitness_diet_engagement",
        "intent_context": {
            "matched_node": {
                "name": "fitness_diet_recommendation",
                "label": "Fitness & Diet Recommendation",
                "level": "specific",
                "lifecycle_stage": "recommendation",
            },
            "confidence": 0.70,
            "abstract_parents": ["wellness_engagement", "revenue_optimization"],
            "specific_children": [],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Wellness",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "I want to drive customer engagement and give daily suggestions on fitness and diet",
        },
        "selected_plan": "recommendation_engine",
        "reasoning": (
            "Wellness fitness/diet daily suggestions maps to the recommendation engine plan. "
            "The pipeline: (1) fetch user health profiles, activity history, and preference data, "
            "(2) profile to understand user segments and data patterns, (3) AG36 analyzes engagement "
            "patterns and segments users, (4) AG29 generates personalized daily recommendations "
            "(fitness routines, diet plans) with confidence scores. Adding AG30 (ContentGenerator) "
            "via override to generate readable daily suggestions from the raw recommendations."
        ),
        "plan_overrides": {
            "add_steps": [
                {"step_order": 4, "agent_id": "AG30",
                 "purpose": "Generate readable daily fitness and diet suggestion content from recommendations",
                 "is_optional": False},
            ],
        },
    },
]


# ==========================================================================
# PART 6: Update intent_patterns on existing plans
# ==========================================================================

PLAN_PATTERN_UPDATES = {
    # Add analytics patterns that were falling through
    "analytics_insights": [
        "payer_behavior_analytics", "revenue_cycle_analytics",
        "healthcare_operations_analytics",
    ],
    # Add wellness engagement to recommendation engine
    "recommendation_engine": [
        "fitness_diet_recommendation", "engagement_scoring",
        "wellness_engagement",
    ],
    # Add attribute extraction to schema + enrichment plans
    "data_enrichment": [
        "product_catalog_enrichment",
    ],
}


# ==========================================================================
# Seed runner
# ==========================================================================

def seed(verbose: bool = True) -> None:
    def log(msg: str) -> None:
        if verbose:
            print(f"[seed_pipeline_gaps] {msg}")

    # ── Intent Graph ──────────────────────────────────────────────────────
    log("Connecting to domain DB (intent graph)...")
    intent_conn = get_intent_conn()
    init_intent_db(intent_conn)

    log(f"Inserting {len(NEW_INTENT_NODES)} intent nodes...")
    for name, label, description, level, domain, lifecycle_stage in NEW_INTENT_NODES:
        nid = upsert_intent_node(intent_conn, name, label, description, level, domain, lifecycle_stage)
        log(f"  {name}: id={nid}")
    intent_conn.commit()

    log(f"\nInserting {len(NEW_TRIPLETS)} triplets...")
    ok, errs = 0, 0
    for subject, predicate, obj in NEW_TRIPLETS:
        try:
            add_intent_triplet(intent_conn, subject, predicate, obj)
            ok += 1
        except (ValueError, Exception) as e:
            log(f"  WARN triplet ({subject} {predicate} {obj}): {e}")
            errs += 1
    intent_conn.commit()
    log(f"  {ok} inserted, {errs} skipped.\n")

    log(f"Inserting {len(NEW_ALIASES)} aliases...")
    alias_ok = 0
    for intent_name, alias, alias_type in NEW_ALIASES:
        try:
            add_intent_alias(intent_conn, intent_name, alias, alias_type)
            alias_ok += 1
        except (ValueError, Exception) as e:
            log(f"  WARN alias ({intent_name}: {alias}): {e}")
    intent_conn.commit()
    log(f"  {alias_ok} inserted.\n")

    log("Embedding all intent nodes...")
    embedded = embed_intent_nodes(intent_conn)
    log(f"  {embedded} nodes embedded.\n")
    intent_conn.close()

    # ── Execution Graph ───────────────────────────────────────────────────
    log("Connecting to domain DB (execution graph)...")
    exec_conn = get_execution_conn()
    init_execution_db(exec_conn)

    # New plans
    log(f"\nInserting {len(NEW_PLANS)} execution plans...")
    for p in NEW_PLANS:
        pid = upsert_execution_plan(
            exec_conn,
            name=p["name"],
            label=p["label"],
            description=p["description"],
            intent_patterns=p["intent_patterns"],
            lifecycle_stages=p["lifecycle_stages"],
            domains=p["domains"],
            requires_dataset=p["requires_dataset"],
        )
        log(f"  {p['name']}: {p['label']} (id={pid})")

    # Update intent_patterns on existing plans
    log("\nUpdating intent_patterns on existing plans...")
    for plan_name, new_patterns in PLAN_PATTERN_UPDATES.items():
        with exec_conn.cursor() as cur:
            # Merge new patterns into existing array
            cur.execute(
                """
                UPDATE execution_plans
                SET intent_patterns = (
                    SELECT array_agg(DISTINCT elem)
                    FROM unnest(intent_patterns || %s::text[]) AS elem
                )
                WHERE name = %s
                RETURNING name, intent_patterns
                """,
                (new_patterns, plan_name),
            )
            row = cur.fetchone()
            if row:
                log(f"  {row[0]}: {len(row[1])} patterns total")
    exec_conn.commit()

    # Steps
    log(f"\nInserting {len(NEW_STEPS)} execution steps...")
    step_count = 0
    for (plan_name, step_order, agent_id, purpose, ingress_map,
         egress_keys, is_optional, condition, hitl_gate, hitl_reason) in NEW_STEPS:
        add_execution_step(
            exec_conn,
            plan_name=plan_name,
            step_order=step_order,
            agent_id=agent_id,
            purpose=purpose,
            ingress_map=ingress_map,
            egress_keys=egress_keys,
            is_optional=is_optional,
            condition=condition,
            hitl_gate=hitl_gate,
            hitl_reason=hitl_reason,
        )
        step_count += 1
    log(f"  {step_count} steps inserted.")

    # Dependencies
    log(f"\nInserting {len(NEW_DEPENDENCIES)} dependencies...")
    dep_count = 0
    for (plan_name, from_id, to_id, from_order, to_order,
         dep_type, description) in NEW_DEPENDENCIES:
        add_execution_dependency(
            exec_conn,
            plan_name=plan_name,
            from_agent_id=from_id,
            to_agent_id=to_id,
            from_step_order=from_order,
            to_step_order=to_order,
            dep_type=dep_type,
            description=description,
        )
        dep_count += 1
    log(f"  {dep_count} dependencies inserted.")

    # Examples
    log(f"\nInserting {len(NEW_EXAMPLES)} few-shot examples...")
    for ex in NEW_EXAMPLES:
        add_execution_example(
            exec_conn,
            example_name=ex["example_name"],
            intent_context=ex["intent_context"],
            domain_context=ex["domain_context"],
            selected_plan=ex["selected_plan"],
            reasoning=ex["reasoning"],
            plan_overrides=ex.get("plan_overrides"),
        )
        log(f"  {ex['example_name']}")

    # Embeddings
    log("\nGenerating BGE-M3 embeddings for new plans and cards...")
    emb_count = backfill_execution_embeddings(exec_conn)
    log(f"  {emb_count} items embedded.")
    exec_conn.close()

    log("\n" + "=" * 60)
    log("Pipeline gaps seeded successfully.")
    log(f"  Intent nodes    : {len(NEW_INTENT_NODES)}")
    log(f"  Triplets        : {ok}")
    log(f"  Aliases         : {alias_ok}")
    log(f"  Exec plans      : {len(NEW_PLANS)}")
    log(f"  Exec steps      : {step_count}")
    log(f"  Exec deps       : {dep_count}")
    log(f"  Examples         : {len(NEW_EXAMPLES)}")
    log(f"  Embeddings      : {emb_count} (intent) + {emb_count} (exec)")
    log("=" * 60)


if __name__ == "__main__":
    seed()
