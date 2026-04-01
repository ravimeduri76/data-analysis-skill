"""
seed_intent_graph.py — Seed the intent knowledge graph with intent nodes, triplets,
and aliases across 6 domains.

Run once (or re-run idempotently — all inserts use ON CONFLICT DO NOTHING / DO UPDATE):

    python hooks/seed_intent_graph.py

After seeding, embeddings are generated automatically via embed_intent_nodes().
Requires BGE-M3 model cached at ~/.cache/huggingface/hub/ — will be downloaded on
first run (~1.5 GB) if not present.

Coverage
--------
  Abstract level  (8 nodes)  : cross-domain goal classes
  Mid level       (12 nodes) : subject areas within a domain
  Specific level  (58 nodes) : concrete operations
  Total           ~78 nodes
  IS_A triplets   ~70
  Other triplets  ~55  (REQUIRES, ENABLES, PRECEDES, COMPETES_WITH, PART_OF)
  Aliases         ~90

Domains seeded
--------------
  Healthcare               — patient risk, clinical coding, RCM
  Healthcare Supply Chain  — GPO, contract, member analysis
  Data Engineering         — quality, enrichment, preparation
  Analytics                — predictive, exploratory, survival, ML
  Commercial / Industrial  — parts, equipment, product lookup
  Sales / Strategy         — revenue, competitive intelligence
  MDM                      — master data, entity resolution
  Knowledge Synthesis      — explanation, process, taxonomy (any domain)
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root or hooks/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
# format: (name, label, description, level, domain, lifecycle_stage)
# ---------------------------------------------------------------------------

NODES = [

    # ── ABSTRACT (cross-domain goal classes) ────────────────────────────────
    (
        "knowledge_synthesis",
        "Knowledge Synthesis",
        "Answer questions, explain processes, look up information, and provide expert guidance without processing a dataset.",
        "abstract", "any", "knowledge_reference",
    ),
    (
        "data_management",
        "Data Management",
        "Govern, prepare, and maintain data assets to ensure quality, consistency, and fitness for downstream use.",
        "abstract", "any", "analytics",
    ),
    (
        "predictive_analytics",
        "Predictive Analytics",
        "Build models and scores that forecast future outcomes from historical data.",
        "abstract", "any", "ml_modeling",
    ),
    (
        "exploratory_analytics",
        "Exploratory Analytics",
        "Profile, describe, and explore data to understand distributions, correlations, and patterns.",
        "abstract", "any", "analytics",
    ),
    (
        "patient_risk_management",
        "Patient Risk Management",
        "Identify, stratify, and intervene on patients at elevated risk of adverse clinical outcomes.",
        "abstract", "Healthcare", "ml_modeling",
    ),
    (
        "parts_and_product_management",
        "Parts and Product Management",
        "Look up, select, verify, and source the correct parts or products for equipment or assemblies.",
        "abstract", "Commercial / Industrial", "product_lookup",
    ),
    (
        "revenue_optimization",
        "Revenue Optimization",
        "Identify and capture incremental revenue through upsell, cross-sell, account strategy, and competitive positioning.",
        "abstract", "any", "sales_strategy",
    ),
    (
        "master_data_management",
        "Master Data Management",
        "Create and maintain a single authoritative version of key business entities across source systems.",
        "abstract", "MDM", "enrichment",
    ),

    # ── MID LEVEL ────────────────────────────────────────────────────────────

    # Data Engineering
    (
        "data_quality_management",
        "Data Quality Management",
        "Assess, report on, and remediate quality issues including nulls, outliers, duplicates, and inconsistencies.",
        "mid", "any", "analytics",
    ),
    (
        "data_enrichment",
        "Data Enrichment",
        "Augment, resolve, and consolidate data to increase completeness, accuracy, and entity coverage.",
        "mid", "any", "enrichment",
    ),
    (
        "data_preparation",
        "Data Preparation",
        "Transform, normalise, and structure raw data into analysis-ready form.",
        "mid", "any", "preparation",
    ),

    # Analytics
    (
        "survival_analysis",
        "Survival Analysis",
        "Model time-to-event outcomes using Kaplan-Meier curves, Cox regression, or parametric survival models.",
        "mid", "any", "survival_analysis",
    ),
    (
        "ml_modeling",
        "Machine Learning Modelling",
        "Build, train, evaluate, and score supervised or unsupervised machine learning models.",
        "mid", "any", "ml_modeling",
    ),

    # Healthcare
    (
        "clinical_coding_and_billing",
        "Clinical Coding and Billing",
        "Assign, look up, and validate ICD-10, CPT, and HCPCS codes for diagnosis, procedure, and billing purposes.",
        "mid", "Healthcare", "knowledge_reference",
    ),
    (
        "healthcare_rcm",
        "Healthcare Revenue Cycle Management",
        "Manage the end-to-end revenue cycle: claim creation, EDI submission, denial handling, and reimbursement.",
        "mid", "Healthcare RCM", "knowledge_reference",
    ),
    (
        "healthcare_supply_chain",
        "Healthcare Supply Chain Analytics",
        "Analyse GPO contract utilisation, member engagement, and category participation to optimise supply spend.",
        "mid", "Healthcare Supply Chain", "analytics",
    ),

    # Commercial / Industrial
    (
        "product_lookup",
        "Product Lookup",
        "Find, identify, and validate a specific product or part by description, application, or cross-reference.",
        "mid", "Commercial / Industrial", "product_lookup",
    ),
    (
        "equipment_maintenance_planning",
        "Equipment Maintenance Planning",
        "Select components, verify fitment, and source replacement parts to restore or maintain equipment operation.",
        "mid", "Commercial / Industrial", "product_lookup",
    ),

    # Sales / Strategy
    (
        "account_growth_strategy",
        "Account Growth Strategy",
        "Develop plans to increase revenue from existing accounts through upsell, cross-sell, and relationship deepening.",
        "mid", "any", "sales_strategy",
    ),
    (
        "competitive_intelligence",
        "Competitive Intelligence",
        "Analyse competitor positioning, market share, and displacement opportunities to inform sales and product strategy.",
        "mid", "any", "competitive_intelligence",
    ),

    # ── SPECIFIC — Data Engineering ──────────────────────────────────────────
    (
        "data_profiling",
        "Data Profiling",
        "Generate statistical summaries of columns: nulls, distributions, cardinality, data types, and quality scores.",
        "specific", "any", "analytics",
    ),
    (
        "data_cleansing",
        "Data Cleansing",
        "Correct, standardise, and remove invalid or inconsistent values in a dataset.",
        "specific", "any", "preparation",
    ),
    (
        "anomaly_detection",
        "Anomaly Detection",
        "Identify statistically unusual records, spikes, or patterns using Z-score, IQR, or model-based methods.",
        "specific", "any", "analytics",
    ),
    (
        "outlier_detection",
        "Outlier Detection",
        "Flag individual data points that deviate significantly from the population distribution.",
        "specific", "any", "analytics",
    ),
    (
        "missing_value_analysis",
        "Missing Value Analysis",
        "Report null rates per column, identify patterns of missingness, and recommend imputation strategies.",
        "specific", "any", "analytics",
    ),
    (
        "entity_resolution",
        "Entity Resolution",
        "Match and cluster records that refer to the same real-world entity across one or more data sources.",
        "specific", "any", "enrichment",
    ),
    (
        "golden_record_creation",
        "Golden Record Creation",
        "Apply survivorship rules to resolved entity clusters to produce a single authoritative master record.",
        "specific", "any", "preparation",
    ),
    (
        "data_quality_improvement",
        "Data Quality Improvement",
        "Systematically improve data quality scores by resolving flagged issues from profiling and DQ assessment.",
        "specific", "any", "enrichment",
    ),
    (
        "data_normalization",
        "Data Normalisation",
        "Rescale or standardise numeric features and enforce consistent formats across categorical fields.",
        "specific", "any", "preparation",
    ),
    (
        "data_transformation",
        "Data Transformation",
        "Reshape, pivot, aggregate, or compute derived fields to produce analysis-ready structure.",
        "specific", "any", "preparation",
    ),
    (
        "data_loading",
        "Data Loading",
        "Load transformed data into a target system, database, or file format.",
        "specific", "any", "preparation",
    ),
    (
        "schema_mapping",
        "Schema Mapping",
        "Map source schema fields to target schema fields, resolving naming and type conflicts.",
        "specific", "any", "preparation",
    ),

    # ── SPECIFIC — Analytics ─────────────────────────────────────────────────
    (
        "dataset_profiling",
        "Dataset Profiling",
        "Profile a full dataset: DQ report, distributions, correlations, and anomaly flags.",
        "specific", "any", "analytics",
    ),
    (
        "correlation_analysis",
        "Correlation Analysis",
        "Compute pairwise correlations between numeric features and identify top correlated pairs.",
        "specific", "any", "analytics",
    ),
    (
        "distribution_analysis",
        "Distribution Analysis",
        "Characterise the statistical distribution of columns including skewness, kurtosis, and histogram shape.",
        "specific", "any", "analytics",
    ),
    (
        "sampling_comparison",
        "Sampling Comparison",
        "Compare two datasets (train/test, before/after) for statistical representativeness and distribution alignment.",
        "specific", "any", "analytics",
    ),
    (
        "kaplan_meier_analysis",
        "Kaplan-Meier Analysis",
        "Generate Kaplan-Meier survival curves and run log-rank tests for group comparison.",
        "specific", "any", "survival_analysis",
    ),
    (
        "cox_regression",
        "Cox Proportional Hazards Regression",
        "Fit a Cox PH model to estimate hazard ratios for covariates in a time-to-event dataset.",
        "specific", "any", "survival_analysis",
    ),
    (
        "churn_prediction",
        "Churn Prediction",
        "Predict which entities (customers, patients, members) will disengage or churn within a time horizon.",
        "specific", "any", "survival_analysis",
    ),
    (
        "binary_classification",
        "Binary Classification",
        "Train and evaluate a supervised model that predicts a binary outcome (positive/negative).",
        "specific", "any", "ml_modeling",
    ),
    (
        "regression_modeling",
        "Regression Modelling",
        "Train and evaluate a supervised model that predicts a continuous numeric outcome.",
        "specific", "any", "ml_modeling",
    ),
    (
        "batch_scoring",
        "Batch Scoring",
        "Apply a trained model to score a large population and produce ranked risk tiers.",
        "specific", "any", "ml_modeling",
    ),

    # ── SPECIFIC — Healthcare ─────────────────────────────────────────────────
    (
        "mortality_prediction",
        "Mortality Prediction",
        "Predict the probability that a patient will die within a specified clinical window.",
        "specific", "Healthcare", "ml_modeling",
    ),
    (
        "readmission_prediction",
        "Readmission Prediction",
        "Predict which patients will be readmitted to hospital within 30 or 90 days of discharge.",
        "specific", "Healthcare", "ml_modeling",
    ),
    (
        "sepsis_prediction",
        "Sepsis Prediction",
        "Identify patients at high risk of developing sepsis based on vitals, labs, and clinical history.",
        "specific", "Healthcare", "ml_modeling",
    ),
    (
        "fall_risk_assessment",
        "Fall Risk Assessment",
        "Score patients on likelihood of in-hospital falls using validated risk factors.",
        "specific", "Healthcare", "ml_modeling",
    ),
    (
        "length_of_stay_prediction",
        "Length of Stay Prediction",
        "Forecast expected hospital length of stay to support capacity planning and discharge optimisation.",
        "specific", "Healthcare", "ml_modeling",
    ),
    (
        "icd_code_lookup",
        "ICD Code Lookup",
        "Find, interpret, or enumerate ICD-10 CM (diagnosis) or PCS (procedure) codes for a clinical condition or surgery.",
        "specific", "Healthcare", "knowledge_reference",
    ),
    (
        "diagnosis_coding",
        "Diagnosis Coding",
        "Assign the correct ICD-10 CM code(s) to a patient diagnosis for billing and reporting.",
        "specific", "Healthcare", "knowledge_reference",
    ),
    (
        "procedure_coding",
        "Procedure Coding",
        "Assign the correct ICD-10 PCS or CPT code(s) to a clinical procedure.",
        "specific", "Healthcare", "knowledge_reference",
    ),
    (
        "claim_processing",
        "Claim Processing",
        "Understand the end-to-end lifecycle of a healthcare insurance claim from submission to adjudication.",
        "specific", "Healthcare", "knowledge_reference",
    ),
    (
        "edi_837_processing",
        "EDI 837 Claim Processing",
        "Explain or design the EDI 837 transaction workflow for submitting professional or institutional claims.",
        "specific", "Healthcare RCM", "knowledge_reference",
    ),
    (
        "claim_denial_management",
        "Claim Denial Management",
        "Identify denial patterns, root causes, and remediation strategies to reduce claim rejection rates.",
        "specific", "Healthcare RCM", "analytics",
    ),
    (
        "reimbursement_optimization",
        "Reimbursement Optimisation",
        "Analyse and improve payer mix, contract rates, and coding accuracy to maximise reimbursement.",
        "specific", "Healthcare RCM", "analytics",
    ),
    (
        "contract_utilization_analysis",
        "Contract Utilisation Analysis",
        "Measure how effectively GPO members are purchasing against contracted items and tiers.",
        "specific", "Healthcare Supply Chain", "analytics",
    ),
    (
        "member_engagement_analysis",
        "Member Engagement Analysis",
        "Analyse GPO member activity, purchasing behaviour, and engagement levels to prevent churn.",
        "specific", "Healthcare Supply Chain", "analytics",
    ),
    (
        "category_participation_analysis",
        "Category Participation Analysis",
        "Measure breadth of category spend under GPO contracts and identify participation gaps.",
        "specific", "Healthcare Supply Chain", "analytics",
    ),

    # ── SPECIFIC — Commercial / Industrial ───────────────────────────────────
    (
        "part_number_lookup",
        "Part Number Lookup",
        "Find the exact OEM or aftermarket part number for a component by description, application, or vehicle/equipment specification.",
        "specific", "Commercial / Industrial", "product_lookup",
    ),
    (
        "compatibility_check",
        "Compatibility Check",
        "Verify that a part or product is compatible with a specific make, model, year, or system configuration.",
        "specific", "Commercial / Industrial", "product_lookup",
    ),
    (
        "cross_reference_lookup",
        "Cross-Reference Lookup",
        "Find equivalent or interchangeable part numbers across OEM and aftermarket suppliers.",
        "specific", "Commercial / Industrial", "product_lookup",
    ),
    (
        "component_selection",
        "Component Selection",
        "Select the correct electrical, mechanical, or electromechanical component for a given application and specification.",
        "specific", "Commercial / Industrial", "product_lookup",
    ),
    (
        "fitment_verification",
        "Fitment Verification",
        "Confirm that a replacement part physically fits and electrically or mechanically interfaces correctly.",
        "specific", "Commercial / Industrial", "product_lookup",
    ),
    (
        "replacement_parts_sourcing",
        "Replacement Parts Sourcing",
        "Identify and evaluate sources for replacement or spare parts, including OEM, aftermarket, and surplus.",
        "specific", "Commercial / Industrial", "product_lookup",
    ),

    # ── SPECIFIC — Sales / Strategy ───────────────────────────────────────────
    (
        "upsell_analysis",
        "Upsell Analysis",
        "Identify accounts with capacity to purchase higher-tier products or expanded service contracts.",
        "specific", "any", "sales_strategy",
    ),
    (
        "cross_sell_analysis",
        "Cross-Sell Analysis",
        "Identify accounts that purchase one product line but not adjacent complementary lines.",
        "specific", "any", "sales_strategy",
    ),
    (
        "account_strategy_planning",
        "Account Strategy Planning",
        "Develop a structured plan for growing revenue and deepening relationships within a key account.",
        "specific", "any", "sales_strategy",
    ),
    (
        "competitive_displacement_analysis",
        "Competitive Displacement Analysis",
        "Identify installed-base accounts using a competitor product that are candidates for displacement.",
        "specific", "any", "competitive_intelligence",
    ),
    (
        "market_share_analysis",
        "Market Share Analysis",
        "Quantify and benchmark a product or brand's share of a defined market segment.",
        "specific", "any", "competitive_intelligence",
    ),
    (
        "installed_base_defense",
        "Installed Base Defense",
        "Develop strategies to retain accounts and prevent displacement by named competitors.",
        "specific", "any", "competitive_intelligence",
    ),

    # ── SPECIFIC — MDM ────────────────────────────────────────────────────────
    (
        "entity_deduplication",
        "Entity Deduplication",
        "Detect and remove or merge duplicate records in a dataset or master data system.",
        "specific", "MDM", "enrichment",
    ),
    (
        "survivorship_rule_application",
        "Survivorship Rule Application",
        "Apply configured survivorship rules to clustered entity matches to determine attribute-level winners.",
        "specific", "MDM", "preparation",
    ),
    (
        "entity_matching",
        "Entity Matching",
        "Score pairs of records for match likelihood using blocking, comparison, and classification steps.",
        "specific", "MDM", "enrichment",
    ),

    # ── MID — Text Analytics (gap exposed by Q1/Q2 test) ────────────────────
    (
        "text_analytics",
        "Text Analytics",
        "Extract, parse, generate, or classify information from unstructured text fields.",
        "mid", "any", "preparation",
    ),

    # ── SPECIFIC — Text Analytics ─────────────────────────────────────────────
    (
        "product_attribute_extraction",
        "Product Attribute Extraction",
        "Parse unstructured product descriptions to extract structured fields such as brand, material, product type, specifications, and features using pattern matching or NLP.",
        "specific", "any", "preparation",
    ),
    (
        "product_description_generation",
        "Product Description Generation",
        "Generate standardised product descriptions (e.g. 80-character short descriptions, long-form copy) from structured product master attributes.",
        "specific", "any", "preparation",
    ),

    # ── SPECIFIC — Schema / Attribute Completeness (gap exposed by Q3 test) ──
    (
        "attribute_gap_analysis",
        "Attribute Gap Analysis",
        "Review a product or entity schema to identify missing, incomplete, or non-standard attributes relative to a target schema or industry standard.",
        "specific", "any", "analytics",
    ),

    # ── SPECIFIC — Knowledge Synthesis ───────────────────────────────────────
    (
        "process_explanation",
        "Process Explanation",
        "Explain how a business process, clinical workflow, or technical system works.",
        "specific", "any", "knowledge_reference",
    ),
    (
        "concept_definition",
        "Concept Definition",
        "Define a domain-specific term, acronym, or concept clearly and accurately.",
        "specific", "any", "knowledge_reference",
    ),
    (
        "procedure_catalogue_lookup",
        "Procedure Catalogue Lookup",
        "Enumerate or categorise a set of procedures, codes, or standards within a defined scope.",
        "specific", "any", "knowledge_reference",
    ),
    (
        "taxonomy_exploration",
        "Taxonomy Exploration",
        "Navigate and explain a classification hierarchy or taxonomy (e.g., ICD chapters, product families, ATC codes).",
        "specific", "any", "knowledge_reference",
    ),
]


# ---------------------------------------------------------------------------
# Triplet definitions
# format: (subject_name, predicate, object_name)
# Predicates: IS_A | REQUIRES | ENABLES | PRECEDES | COMPETES_WITH | PART_OF
# ---------------------------------------------------------------------------

TRIPLETS = [

    # ── IS_A: mid → abstract ─────────────────────────────────────────────────
    ("data_quality_management",     "IS_A", "data_management"),
    ("data_enrichment",             "IS_A", "data_management"),
    ("data_preparation",            "IS_A", "data_management"),
    ("survival_analysis",           "IS_A", "predictive_analytics"),
    ("ml_modeling",                 "IS_A", "predictive_analytics"),
    ("clinical_coding_and_billing", "IS_A", "knowledge_synthesis"),
    ("healthcare_rcm",              "IS_A", "knowledge_synthesis"),
    ("healthcare_supply_chain",     "IS_A", "exploratory_analytics"),
    ("product_lookup",              "IS_A", "parts_and_product_management"),
    ("equipment_maintenance_planning", "IS_A", "parts_and_product_management"),
    ("account_growth_strategy",     "IS_A", "revenue_optimization"),
    ("competitive_intelligence",    "IS_A", "revenue_optimization"),

    # ── IS_A: specific → mid (Data Engineering) ──────────────────────────────
    ("data_profiling",              "IS_A", "data_quality_management"),
    ("data_cleansing",              "IS_A", "data_quality_management"),
    ("anomaly_detection",           "IS_A", "data_quality_management"),
    ("outlier_detection",           "IS_A", "data_quality_management"),
    ("missing_value_analysis",      "IS_A", "data_quality_management"),
    ("entity_resolution",           "IS_A", "data_enrichment"),
    ("golden_record_creation",      "IS_A", "data_enrichment"),
    ("data_quality_improvement",    "IS_A", "data_enrichment"),  # user's example
    ("data_normalization",          "IS_A", "data_preparation"),  # user's example
    ("data_transformation",         "IS_A", "data_preparation"),
    ("data_loading",                "IS_A", "data_preparation"),
    ("schema_mapping",              "IS_A", "data_preparation"),

    # ── IS_A: specific → mid (Analytics) ─────────────────────────────────────
    ("dataset_profiling",           "IS_A", "exploratory_analytics"),
    ("correlation_analysis",        "IS_A", "exploratory_analytics"),
    ("distribution_analysis",       "IS_A", "exploratory_analytics"),
    ("sampling_comparison",         "IS_A", "exploratory_analytics"),
    ("kaplan_meier_analysis",       "IS_A", "survival_analysis"),
    ("cox_regression",              "IS_A", "survival_analysis"),
    ("churn_prediction",            "IS_A", "survival_analysis"),
    ("binary_classification",       "IS_A", "ml_modeling"),
    ("regression_modeling",         "IS_A", "ml_modeling"),
    ("batch_scoring",               "IS_A", "ml_modeling"),

    # ── IS_A: specific → abstract (patient risk — user's example) ────────────
    ("mortality_prediction",        "IS_A", "patient_risk_management"),  # user's example
    ("readmission_prediction",      "IS_A", "patient_risk_management"),
    ("sepsis_prediction",           "IS_A", "patient_risk_management"),
    ("fall_risk_assessment",        "IS_A", "patient_risk_management"),
    ("length_of_stay_prediction",   "IS_A", "patient_risk_management"),

    # patient_risk_management IS_A predictive_analytics (cross-domain IS_A)
    ("patient_risk_management",     "IS_A", "predictive_analytics"),

    # ── IS_A: specific → mid (Healthcare coding/RCM) ─────────────────────────
    ("icd_code_lookup",             "IS_A", "clinical_coding_and_billing"),
    ("diagnosis_coding",            "IS_A", "clinical_coding_and_billing"),
    ("procedure_coding",            "IS_A", "clinical_coding_and_billing"),
    ("claim_processing",            "IS_A", "clinical_coding_and_billing"),
    ("edi_837_processing",          "IS_A", "healthcare_rcm"),
    ("claim_denial_management",     "IS_A", "healthcare_rcm"),
    ("reimbursement_optimization",  "IS_A", "healthcare_rcm"),

    # ── IS_A: specific → mid (HC Supply Chain) ───────────────────────────────
    ("contract_utilization_analysis", "IS_A", "healthcare_supply_chain"),
    ("member_engagement_analysis",    "IS_A", "healthcare_supply_chain"),
    ("category_participation_analysis","IS_A","healthcare_supply_chain"),

    # ── IS_A: specific → mid (Commercial) ────────────────────────────────────
    ("part_number_lookup",          "IS_A", "product_lookup"),
    ("compatibility_check",         "IS_A", "product_lookup"),
    ("cross_reference_lookup",      "IS_A", "product_lookup"),
    ("component_selection",         "IS_A", "equipment_maintenance_planning"),
    ("fitment_verification",        "IS_A", "equipment_maintenance_planning"),
    ("replacement_parts_sourcing",  "IS_A", "equipment_maintenance_planning"),

    # ── IS_A: specific → mid (Sales) ─────────────────────────────────────────
    ("upsell_analysis",             "IS_A", "account_growth_strategy"),
    ("cross_sell_analysis",         "IS_A", "account_growth_strategy"),
    ("account_strategy_planning",   "IS_A", "account_growth_strategy"),
    ("competitive_displacement_analysis", "IS_A", "competitive_intelligence"),
    ("market_share_analysis",       "IS_A", "competitive_intelligence"),
    ("installed_base_defense",      "IS_A", "competitive_intelligence"),

    # ── IS_A: specific → mid (MDM) ───────────────────────────────────────────
    ("entity_deduplication",        "IS_A", "master_data_management"),
    ("survivorship_rule_application","IS_A", "master_data_management"),
    ("entity_matching",             "IS_A", "master_data_management"),

    # MDM IS_A data_enrichment (MDM is a form of data enrichment)
    ("master_data_management",      "IS_A", "data_enrichment"),

    # ── IS_A: text analytics hierarchy (new) ─────────────────────────────────
    ("text_analytics",                  "IS_A", "data_preparation"),
    ("product_attribute_extraction",    "IS_A", "text_analytics"),
    ("product_description_generation",  "IS_A", "text_analytics"),

    # attribute_gap_analysis IS_A data_quality_management (it's a quality assessment)
    ("attribute_gap_analysis",          "IS_A", "data_quality_management"),

    # cross-link: product_attribute_extraction also IS_A data_transformation (it reshapes data)
    ("product_attribute_extraction",    "IS_A", "data_transformation"),

    # REQUIRES / ENABLES for new nodes
    ("product_description_generation",  "REQUIRES", "product_attribute_extraction"),
    ("product_attribute_extraction",    "ENABLES",  "product_description_generation"),
    ("attribute_gap_analysis",          "REQUIRES", "data_profiling"),
    ("attribute_gap_analysis",          "ENABLES",  "data_quality_improvement"),

    # ── IS_A: specific → abstract (Knowledge Synthesis) ──────────────────────
    ("process_explanation",         "IS_A", "knowledge_synthesis"),
    ("concept_definition",          "IS_A", "knowledge_synthesis"),
    ("procedure_catalogue_lookup",  "IS_A", "knowledge_synthesis"),
    ("taxonomy_exploration",        "IS_A", "knowledge_synthesis"),

    # icd_code_lookup is also a form of taxonomy/procedure exploration
    ("icd_code_lookup",             "IS_A", "taxonomy_exploration"),

    # ── REQUIRES ─────────────────────────────────────────────────────────────
    ("binary_classification",       "REQUIRES", "data_profiling"),
    ("cox_regression",              "REQUIRES", "kaplan_meier_analysis"),
    ("golden_record_creation",      "REQUIRES", "entity_resolution"),
    ("entity_resolution",           "REQUIRES", "data_profiling"),
    ("batch_scoring",               "REQUIRES", "binary_classification"),
    ("diagnosis_coding",            "REQUIRES", "icd_code_lookup"),
    ("survivorship_rule_application","REQUIRES", "entity_matching"),
    ("churn_prediction",            "REQUIRES", "data_profiling"),
    ("data_cleansing",              "REQUIRES", "data_profiling"),
    ("data_normalization",          "REQUIRES", "data_profiling"),
    ("fitment_verification",        "REQUIRES", "compatibility_check"),
    ("reimbursement_optimization",  "REQUIRES", "claim_denial_management"),

    # ── ENABLES ──────────────────────────────────────────────────────────────
    ("data_normalization",          "ENABLES", "binary_classification"),
    ("data_normalization",          "ENABLES", "regression_modeling"),
    ("data_profiling",              "ENABLES", "anomaly_detection"),
    ("data_profiling",              "ENABLES", "data_cleansing"),
    ("entity_resolution",           "ENABLES", "golden_record_creation"),
    ("kaplan_meier_analysis",       "ENABLES", "cox_regression"),
    ("entity_matching",             "ENABLES", "survivorship_rule_application"),
    ("binary_classification",       "ENABLES", "batch_scoring"),
    ("data_quality_improvement",    "ENABLES", "binary_classification"),
    ("contract_utilization_analysis","ENABLES", "account_strategy_planning"),
    ("member_engagement_analysis",  "ENABLES", "installed_base_defense"),

    # ── PRECEDES ─────────────────────────────────────────────────────────────
    ("data_profiling",              "PRECEDES", "data_cleansing"),
    ("data_profiling",              "PRECEDES", "anomaly_detection"),
    ("data_profiling",              "PRECEDES", "missing_value_analysis"),
    ("kaplan_meier_analysis",       "PRECEDES", "cox_regression"),
    ("entity_resolution",           "PRECEDES", "golden_record_creation"),
    ("entity_matching",             "PRECEDES", "entity_resolution"),
    ("data_transformation",         "PRECEDES", "data_loading"),
    ("schema_mapping",              "PRECEDES", "data_transformation"),
    ("compatibility_check",         "PRECEDES", "fitment_verification"),

    # ── PART_OF ───────────────────────────────────────────────────────────────
    ("golden_record_creation",      "PART_OF", "master_data_management"),
    ("entity_matching",             "PART_OF", "entity_resolution"),
    ("kaplan_meier_analysis",       "PART_OF", "survival_analysis"),
    ("data_profiling",              "PART_OF", "dataset_profiling"),

    # ── COMPETES_WITH ─────────────────────────────────────────────────────────
    ("cox_regression",              "COMPETES_WITH", "binary_classification"),   # both predict events
    ("churn_prediction",            "COMPETES_WITH", "binary_classification"),   # churn is binary too
    ("entity_deduplication",        "COMPETES_WITH", "entity_matching"),         # simpler vs full MDM
    ("market_share_analysis",       "COMPETES_WITH", "competitive_displacement_analysis"),
]


# ---------------------------------------------------------------------------
# Alias definitions
# format: (intent_name, alias, alias_type)
# alias_type: 'keyword' | 'phrase' | 'synonym'
# ---------------------------------------------------------------------------

ALIASES = [
    # Data Engineering
    ("data_profiling",           "profile data",             "phrase"),
    ("data_profiling",           "data quality",             "phrase"),
    ("data_profiling",           "DQ report",                "keyword"),
    ("data_profiling",           "quality report",           "phrase"),
    ("data_cleansing",           "clean data",               "phrase"),
    ("data_cleansing",           "fix data",                 "phrase"),
    ("data_cleansing",           "data cleaning",            "synonym"),
    ("anomaly_detection",        "find anomalies",           "phrase"),
    ("anomaly_detection",        "detect anomalies",         "phrase"),
    ("anomaly_detection",        "unusual patterns",         "phrase"),
    ("anomaly_detection",        "fraud detection",          "phrase"),
    ("outlier_detection",        "find outliers",            "phrase"),
    ("outlier_detection",        "outlier analysis",         "phrase"),
    ("missing_value_analysis",   "missing values",           "phrase"),
    ("missing_value_analysis",   "null analysis",            "phrase"),
    ("missing_value_analysis",   "null rate",                "phrase"),
    ("entity_resolution",        "deduplication",            "synonym"),
    ("entity_resolution",        "dedup",                    "keyword"),
    ("entity_resolution",        "record linkage",           "phrase"),
    ("entity_resolution",        "match records",            "phrase"),
    ("golden_record_creation",   "golden record",            "phrase"),
    ("golden_record_creation",   "survivorship",             "keyword"),
    ("golden_record_creation",   "merge records",            "phrase"),
    ("data_normalization",       "normalise data",           "phrase"),
    ("data_normalization",       "normalize data",           "phrase"),
    ("data_normalization",       "scale features",           "phrase"),
    ("data_transformation",      "transform data",           "phrase"),
    ("data_transformation",      "reshape data",             "phrase"),
    ("data_transformation",      "compute features",         "phrase"),

    # Analytics
    ("dataset_profiling",        "analyse dataset",          "phrase"),
    ("dataset_profiling",        "analyze data",             "phrase"),
    ("dataset_profiling",        "explore data",             "phrase"),
    ("correlation_analysis",     "correlations",             "keyword"),
    ("correlation_analysis",     "feature correlation",      "phrase"),
    ("sampling_comparison",      "compare datasets",         "phrase"),
    ("sampling_comparison",      "train test comparison",    "phrase"),
    ("kaplan_meier_analysis",    "KM curve",                 "keyword"),
    ("kaplan_meier_analysis",    "Kaplan Meier",             "phrase"),
    ("kaplan_meier_analysis",    "survival curve",           "phrase"),
    ("cox_regression",           "Cox model",                "phrase"),
    ("cox_regression",           "Cox regression",           "phrase"),
    ("cox_regression",           "hazard ratio",             "phrase"),
    ("churn_prediction",         "churn model",              "phrase"),
    ("churn_prediction",         "churn rate",               "phrase"),
    ("churn_prediction",         "predict churn",            "phrase"),
    ("binary_classification",    "classification model",     "phrase"),
    ("binary_classification",    "predict outcome",          "phrase"),
    ("binary_classification",    "predict flag",             "phrase"),
    ("batch_scoring",            "score population",         "phrase"),
    ("batch_scoring",            "risk scores",              "phrase"),
    ("batch_scoring",            "risk tiers",               "phrase"),

    # Healthcare
    ("mortality_prediction",     "predict mortality",        "phrase"),
    ("mortality_prediction",     "death risk",               "phrase"),
    ("mortality_prediction",     "mortality risk",           "phrase"),
    ("readmission_prediction",   "readmission risk",         "phrase"),
    ("readmission_prediction",   "predict readmission",      "phrase"),
    ("readmission_prediction",   "30 day readmission",       "phrase"),
    ("icd_code_lookup",          "ICD codes",                "phrase"),
    ("icd_code_lookup",          "ICD-10",                   "keyword"),
    ("icd_code_lookup",          "diagnosis codes",          "phrase"),
    ("icd_code_lookup",          "procedure codes",          "phrase"),
    ("edi_837_processing",       "EDI 837",                  "keyword"),
    ("edi_837_processing",       "837 claim",                "phrase"),
    ("edi_837_processing",       "claim submission",         "phrase"),
    ("claim_denial_management",  "denial management",        "phrase"),
    ("claim_denial_management",  "claim denials",            "phrase"),
    ("claim_denial_management",  "denial rate",              "phrase"),

    # Commercial / Industrial
    ("part_number_lookup",       "find part",                "phrase"),
    ("part_number_lookup",       "part number",              "phrase"),
    ("part_number_lookup",       "headlight",                "keyword"),
    ("compatibility_check",      "compatible",               "keyword"),
    ("compatibility_check",      "will it fit",              "phrase"),
    ("cross_reference_lookup",   "cross reference",          "phrase"),
    ("cross_reference_lookup",   "interchangeable",          "keyword"),
    ("component_selection",      "limit switch",             "phrase"),
    ("component_selection",      "relay types",              "phrase"),
    ("component_selection",      "select component",         "phrase"),

    # Sales / Strategy
    ("upsell_analysis",          "upsell",                   "keyword"),
    ("upsell_analysis",          "upgrade opportunity",      "phrase"),
    ("cross_sell_analysis",      "cross sell",               "phrase"),
    ("cross_sell_analysis",      "adjacent products",        "phrase"),
    ("account_strategy_planning","account strategy",         "phrase"),
    ("account_strategy_planning","revenue synergy",          "phrase"),
    ("competitive_displacement_analysis", "displace competitor", "phrase"),
    ("competitive_displacement_analysis", "win against",     "phrase"),
    ("installed_base_defense",   "defend account",           "phrase"),
    ("installed_base_defense",   "prevent displacement",     "phrase"),

    # Text Analytics / Product Data (new)
    ("product_attribute_extraction", "extract fields",           "phrase"),
    ("product_attribute_extraction", "parse description",        "phrase"),
    ("product_attribute_extraction", "extract attributes",       "phrase"),
    ("product_attribute_extraction", "NLP extraction",           "phrase"),
    ("product_attribute_extraction", "parse product text",       "phrase"),
    ("product_attribute_extraction", "field extraction",         "phrase"),
    ("product_description_generation","generate description",    "phrase"),
    ("product_description_generation","create description",      "phrase"),
    ("product_description_generation","80 character description","phrase"),
    ("product_description_generation","short description",       "phrase"),
    ("product_description_generation","description generation",  "phrase"),
    ("attribute_gap_analysis",       "missing attributes",       "phrase"),
    ("attribute_gap_analysis",       "schema review",            "phrase"),
    ("attribute_gap_analysis",       "critical attributes",      "phrase"),
    ("attribute_gap_analysis",       "attribute completeness",   "phrase"),
    ("attribute_gap_analysis",       "schema completeness",      "phrase"),
    ("attribute_gap_analysis",       "missing fields",           "phrase"),

    # MDM
    ("entity_deduplication",     "remove duplicates",        "phrase"),
    ("entity_deduplication",     "duplicate records",        "phrase"),
    ("entity_matching",          "match entities",           "phrase"),
    ("entity_matching",          "fuzzy match",              "phrase"),

    # Knowledge Synthesis
    ("process_explanation",      "how does",                 "phrase"),
    ("process_explanation",      "explain process",          "phrase"),
    ("process_explanation",      "how it works",             "phrase"),
    ("concept_definition",       "what is",                  "phrase"),
    ("concept_definition",       "define",                   "keyword"),
    ("procedure_catalogue_lookup","what are the",            "phrase"),
    ("procedure_catalogue_lookup","list of",                 "phrase"),
    ("taxonomy_exploration",     "types of",                 "phrase"),
    ("taxonomy_exploration",     "categories of",            "phrase"),
    ("taxonomy_exploration",     "classification",           "keyword"),
]


# ---------------------------------------------------------------------------
# Seed runner
# ---------------------------------------------------------------------------

def seed(verbose: bool = True) -> None:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("Connecting to domain DB...")
    conn = get_intent_conn()
    init_intent_db(conn)
    log("Schema ready.\n")

    # ── nodes ────────────────────────────────────────────────────────────────
    log(f"Inserting {len(NODES)} intent nodes...")
    node_ids: dict[str, int] = {}
    for name, label, description, level, domain, lifecycle_stage in NODES:
        node_id = upsert_intent_node(conn, name, label, description, level, domain, lifecycle_stage)
        node_ids[name] = node_id
    conn.commit()
    log(f"  {len(node_ids)} nodes upserted.\n")

    # ── triplets ──────────────────────────────────────────────────────────────
    log(f"Inserting {len(TRIPLETS)} triplets...")
    triplet_count = 0
    errors = 0
    for subject, predicate, obj in TRIPLETS:
        try:
            add_intent_triplet(conn, subject, predicate, obj)
            triplet_count += 1
        except ValueError as e:
            log(f"  WARN: {e}")
            errors += 1
    conn.commit()
    log(f"  {triplet_count} triplets inserted, {errors} errors.\n")

    # ── aliases ───────────────────────────────────────────────────────────────
    log(f"Inserting {len(ALIASES)} aliases...")
    alias_count = 0
    for intent_name, alias, alias_type in ALIASES:
        try:
            add_intent_alias(conn, intent_name, alias, alias_type)
            alias_count += 1
        except ValueError as e:
            log(f"  WARN: {e}")
    conn.commit()
    log(f"  {alias_count} aliases inserted.\n")

    # ── embeddings ────────────────────────────────────────────────────────────
    log("Generating BGE-M3 embeddings for all nodes (this may take 30-120s on CPU)...")
    embedded = embed_intent_nodes(conn)
    log(f"  {embedded} nodes embedded.\n")

    conn.close()

    log("=" * 60)
    log("Intent graph seeded successfully.")
    log(f"  Nodes    : {len(node_ids)}")
    log(f"  Triplets : {triplet_count}")
    log(f"  Aliases  : {alias_count}")
    log(f"  Embedded : {embedded}")
    log("=" * 60)


if __name__ == "__main__":
    seed()
