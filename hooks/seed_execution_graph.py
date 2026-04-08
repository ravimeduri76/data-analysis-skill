"""
seed_execution_graph.py — Seed the execution graph with agent cards (AG01-AG40),
plan templates (~17), steps, dependencies, and few-shot examples (~21).

Run once (or re-run idempotently — all inserts use ON CONFLICT DO UPDATE):

    python hooks/seed_execution_graph.py

After seeding, embeddings are generated automatically via backfill_execution_embeddings().

Coverage
--------
  Agent cards      : 40  (AG01-AG40)
  Plan templates   : 17
  Few-shot examples: 21

Agent categories
----------------
  Orchestration     : AG01 (Evaluator), AG02 (DESO Planner), AG03 (ESA/HITL)
  Data Access       : AG04 (Data Pipeline), AG05 (Sampler)
  Profiling         : AG06 (Initial Profiler), AG07 (Full Profiler)
  Schema            : AG08 (Schema Recommender), AG09 (Schema Harmonizer), AG10 (Schema Enforcer)
  Data Quality      : AG11 (DQ Rules), AG12 (DQ Completeness), AG13 (DQ Correctness),
                      AG14 (Anomaly Detector), AG15 (Recency Assessor),
                      AG16 (DQ Fixer), AG17 (DQ Scorer)
  Assessment        : AG18 (Discoverability), AG19 (Catalog Quality), AG20 (Fit for Use)
  Governance        : AG21 (Lineage), AG22 (Data Steward)
  Knowledge         : AG23 (Customer Ontology), AG24 (Core Ontology)
  Enrichment        : AG25 (External Enricher), AG26 (Internal Enricher)
  MDM / 360         : AG27 (X360), AG28 (Golden Data Manager)
  Intelligence      : AG29 (NBA Recommender), AG30 (Content Generator), AG31 (Content Optimizer)
  Security          : AG32 (Sensitive Data Analyzer), AG33 (Data Loss Protector)
  Publishing        : AG34 (Catalog Manager), AG35 (Syndicator)
  Analytics         : AG36 (Data Analysis), AG37 (RAG Agent), AG38 (Search)
  Operations        : AG39 (FinOps)
  Research          : AG40 (ResearchOrchestrator)
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root or hooks/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution_graph import (
    get_execution_conn,
    init_execution_db,
    upsert_agent_card,
    upsert_execution_plan,
    add_execution_step,
    add_execution_dependency,
    add_execution_example,
    backfill_execution_embeddings,
)

# ---------------------------------------------------------------------------
# Agent Cards: AG01–AG40
# ---------------------------------------------------------------------------

AGENT_CARDS = [
    # ── Orchestration ─────────────────────────────────────────────────────
    {
        "agent_id": "AG01",
        "name": "Evaluator",
        "type_": "cognitive",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Assess the completion of a workflow or task by another agent using general intelligence or tools. Generic evaluator and reviewer.",
        "skill_workflow": None,
        "ingress_schema": {"task_output": "any", "evaluation_criteria": "dict", "tools": "list (optional)"},
        "egress_schema": {"evaluation_result": "dict", "pass_fail": "bool", "findings": "list", "recommendations": "list"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Runs after any agent to validate output quality. Can use tools for assessment.",
    },
    {
        "agent_id": "AG02",
        "name": "DESO.Planner",
        "type_": "cognitive",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Plan a workflow by reasoning the various steps with available data and tools. Generic or specialized planner that outputs multi-step execution plans. Seeks more data or tools as required for efficient execution.",
        "skill_workflow": "DESO Skill orchestrator",
        "ingress_schema": {"objective": "str", "available_data": "list", "available_tools": "list", "domain_ctx": "DCP (optional)"},
        "egress_schema": {"execution_plan": "dict", "steps": "list", "resource_requests": "list", "risks": "list"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "This is the DESO orchestrator itself. Referenced for plan-within-plan composition.",
    },
    {
        "agent_id": "AG03",
        "name": "ESA.HumanInTheLoop",
        "type_": "cognitive",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Plan inputs required from human actors across agents. Consolidate and aggregate required inputs and present in a way that minimizes human effort while seeking the right inputs.",
        "skill_workflow": None,
        "ingress_schema": {"hitl_requests": "list[dict]", "agent_outputs": "dict", "priority": "str"},
        "egress_schema": {"consolidated_questions": "list", "human_responses": "dict", "approval_status": "dict"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "Aggregates HITL needs from multiple agents into a single consolidated review. Minimizes user interruptions.",
    },

    # ── Data Access ───────────────────────────────────────────────────────
    {
        "agent_id": "AG04",
        "name": "DataPipeline",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Connect to data sources (files, directories, databases) and fetch or push both metadata and actual data. Uses MCP where available, else native connectors or drivers.",
        "skill_workflow": None,
        "ingress_schema": {"source_config": "dict (type, path/connection_string, credentials)", "query": "str (optional)", "push_target": "dict (optional)"},
        "egress_schema": {"dataset": "DataFrame or file path", "metadata": "dict (schema, row_count, col_count)", "connection_status": "str"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Phase 1: file/directory and DB. MCP integration where available. Handles both read and write.",
    },
    {
        "agent_id": "AG05",
        "name": "Sampler",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Scan all available data and create a representative sample of desired size preserving patterns and inter-relationships in the data.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame or path", "sample_size": "int or float (fraction)", "stratify_cols": "list (optional)", "preserve_relations": "bool"},
        "egress_schema": {"sample_df": "DataFrame", "sampling_report": "dict (method, coverage, representativeness_score)"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Use before profiling on large datasets. Preserves statistical distributions and key relationships.",
    },

    # ── Profiling ─────────────────────────────────────────────────────────
    {
        "agent_id": "AG06",
        "name": "InitialProfiler",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "High-level initial analysis: use an LLM to recommend which columns can be profiled by LLM with confidence. Split columns into LLM-profileable and program-profileable sets.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame or sample", "domain_ctx": "DCP (optional)"},
        "egress_schema": {"llm_columns": "list", "program_columns": "list", "initial_summary": "dict", "column_type_map": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Runs before Full Profiler. Uses LLM intelligence to route columns to the right profiling method.",
    },
    {
        "agent_id": "AG07",
        "name": "FullProfiler",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Full profiling at schema, table, and column levels along basic, statistical, functional, non-functional, and business dimensions. Federates tasks to multiple tools/LLMs. Produces consumable profiling report with HITL input.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame or path", "column_routing": "from AG06 (optional)", "domain_ctx": "DCP", "business_rules": "dict (optional)"},
        "egress_schema": {"profile_report": "dict", "column_profiles": "list[dict]", "statistical_summary": "DataFrame", "business_dimension_scores": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Comprehensive profiler. Can use multiple tools, LLMs, and foundation models. Produces the main profiling report.",
    },

    # ── Schema Management ─────────────────────────────────────────────────
    {
        "agent_id": "AG08",
        "name": "SchemaRecommender",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Compare customer data schema to an industry standard (HL7/FHIR, ACORD, etc.) or domain model (ICD-10, drug ontologies) or EDB Knowledge Graph. Recommend schema differences: entities, attributes, relations, views to incorporate.",
        "skill_workflow": None,
        "ingress_schema": {"customer_schema": "dict", "reference_model": "str or dict (standard name or custom)", "domain_ctx": "DCP"},
        "egress_schema": {"gap_analysis": "dict", "recommended_entities": "list", "recommended_attributes": "list", "recommended_relations": "list", "conformance_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Compares to HL7/FHIR for EMR, ACORD for Insurance, ICD-10 for Claims, etc. Uses Knowledge Graph for domain models.",
    },
    {
        "agent_id": "AG09",
        "name": "SchemaHarmonizer",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Implement data quality policy and rule engine with industry standard and normative rules at table, column, and row level based on industry or domain standard.",
        "skill_workflow": None,
        "ingress_schema": {"schema": "dict", "industry_standard": "str", "gap_analysis": "from AG08 (optional)"},
        "egress_schema": {"harmonized_schema": "dict", "rules_applied": "list", "rule_engine_config": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Inherits normative rules from industry/domain standards. Works with AG08 recommendations.",
    },
    {
        "agent_id": "AG10",
        "name": "SchemaEnforcer",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Use domain or industry model through the Knowledge Graph and AI to enforce schema fixes. Create new tables, attributes, relations based on profiling and transaction patterns.",
        "skill_workflow": None,
        "ingress_schema": {"current_schema": "dict", "target_schema": "dict", "graph_ctx": "from AG24 (optional)", "profiling_output": "from AG07"},
        "egress_schema": {"schema_changes": "list[dict]", "ddl_statements": "list[str]", "enforcement_report": "dict"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "Schema changes are structural — always requires HITL approval before applying DDL.",
    },

    # ── Data Quality ──────────────────────────────────────────────────────
    {
        "agent_id": "AG11",
        "name": "DQManager.Rules",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Allow customers to create custom data quality rules based on identified patterns and data relations discovered through AI with HITL workflow.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "existing_rules": "dict (optional)", "profiling_output": "from AG07", "domain_ctx": "DCP"},
        "egress_schema": {"rules": "list[dict]", "rule_validation_results": "dict", "suggested_rules": "list[dict]"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "AI discovers patterns, suggests rules, human confirms. Rules stored for reuse.",
    },
    {
        "agent_id": "AG12",
        "name": "DQManager.Completeness",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Assess data completeness based on profiling: percentage of NULLs, sparsity, and other attributes. Give a completeness score.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "profiling_output": "from AG07", "domain_ctx": "DCP"},
        "egress_schema": {"completeness_report": "dict", "null_analysis": "DataFrame", "sparsity_analysis": "DataFrame", "completeness_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Part of the quality dimension assessment. Feeds into AG17 for overall scoring.",
    },
    {
        "agent_id": "AG13",
        "name": "DQManager.Correctness",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Assess data correctness along precision (e.g. significant digits), consistency (logical: men cannot have breast cancer; physical: zip/state mismatches), and accuracy (e.g. product attribute verification).",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "profiling_output": "from AG07", "domain_ctx": "DCP", "catalog": "dict (optional)", "policies": "dict (optional)"},
        "egress_schema": {"correctness_report": "dict", "precision_issues": "list", "consistency_issues": "list", "accuracy_issues": "list", "correctness_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Uses catalog, policies, and industry model for validation. Detects logical and physical inconsistencies.",
    },
    {
        "agent_id": "AG14",
        "name": "AnomalyDetector",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Identify outliers and anomalies at column, row, and multivariate levels using rules, statistical tools (Monte Carlo, SODA), and advanced AI/ML techniques (mixture models, distance-based, entropy-based).",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "domain_ctx": "DCP", "methods": "list (optional: rules, statistical, ml)"},
        "egress_schema": {"anomaly_report": "DataFrame", "outlier_rows": "DataFrame", "anomaly_summary": "dict", "method_results": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Multi-method anomaly detection. Column-level, row-level, and multivariate analysis.",
    },
    {
        "agent_id": "AG15",
        "name": "RecencyAssessor",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Assess if data is current and up-to-date. Check whether data represents current time or a specific period/point in the past.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "reference_date": "date (optional)", "date_columns": "list"},
        "egress_schema": {"recency_report": "dict", "staleness_map": "dict", "recency_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Feeds into Fit for Use assessment. Important for time-sensitive domains.",
    },
    {
        "agent_id": "AG16",
        "name": "DQFixer",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Fix data quality issues using LLM, programming, rules, SODA, or Monte Carlo: missing values, inconsistencies, precision, accuracy, deduplication, outlier removal, normalization, catalog adherence.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "quality_issues": "from AG12/AG13/AG14", "rules": "from AG11", "fix_strategy": "dict"},
        "egress_schema": {"fixed_dataset": "DataFrame", "fix_log": "list[dict]", "before_after_metrics": "dict"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "Modifies data — always requires HITL approval. Fix log tracks every change for auditability.",
    },
    {
        "agent_id": "AG17",
        "name": "DQScorer",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Score data for quality both for a stated objective and generally on a unified scale. Combine profiling, quality assessment, and fit-for-use into a standardized score of 100 or customer-chosen scale.",
        "skill_workflow": None,
        "ingress_schema": {"profiling_output": "from AG07", "completeness": "from AG12", "correctness": "from AG13", "anomaly_report": "from AG14", "recency": "from AG15 (optional)", "objective": "str (optional)", "scale": "int (default 100)"},
        "egress_schema": {"overall_score": "float", "dimension_scores": "dict", "score_report": "dict", "fit_for_use_flag": "bool"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Aggregates all quality dimensions into a single score. Customizable scale and weighting.",
    },

    # ── Assessment ────────────────────────────────────────────────────────
    {
        "agent_id": "AG18",
        "name": "DiscoverabilityAssessor",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Measure discoverability of data from catalog: try retrieving catalog objects through queries, keyword searches, full indexing, and semantic information retrieval.",
        "skill_workflow": None,
        "ingress_schema": {"catalog": "dict", "data_sources": "list", "test_queries": "list (optional)"},
        "egress_schema": {"discoverability_report": "dict", "retrieval_scores": "dict", "gaps": "list", "discoverability_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Tests catalog from user perspective: can data be found?",
    },
    {
        "agent_id": "AG19",
        "name": "CatalogQualityAssessor",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Measure catalog quality and accuracy by comparing to industry standard and actual data. Check if catalog is complete and consistent with real data.",
        "skill_workflow": None,
        "ingress_schema": {"catalog": "dict", "industry_standard": "str or dict", "actual_data_schemas": "list[dict]"},
        "egress_schema": {"catalog_quality_report": "dict", "completeness_gaps": "list", "consistency_issues": "list", "catalog_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Compares catalog metadata to both industry standard and actual data.",
    },
    {
        "agent_id": "AG20",
        "name": "FitForUse",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Top-down assessment of whether data is fit for a declared use case or business goal across dimensions: quality, readiness, feature relevancy, label quality, feature quality, class balance, ease of pre-processing, bias/representativeness, noise, data sufficiency.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "business_objective": "str", "quality_scores": "from AG17", "profiling": "from AG07", "domain_ctx": "DCP"},
        "egress_schema": {"fit_for_use_report": "dict", "dimension_scores": "dict", "readiness_level": "str (ready/conditional/not_ready)", "remediation_plan": "list (if not ready)"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "10-dimension fitness assessment. Produces actionable remediation plan when data is not ready.",
    },

    # ── Governance ────────────────────────────────────────────────────────
    {
        "agent_id": "AG21",
        "name": "Lineage",
        "type_": "worker",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Establish data lineage and provenance through journey tracing. Trace data back to source with transformation details. Ensure data integrity throughout the journey.",
        "skill_workflow": None,
        "ingress_schema": {"data_entity": "str", "source_registry": "dict", "transformation_log": "list (optional)"},
        "egress_schema": {"lineage_graph": "dict", "provenance_report": "dict", "integrity_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Critical for compliance and audit. Tracks every transformation from source to consumption.",
    },
    {
        "agent_id": "AG22",
        "name": "DataSteward",
        "type_": "worker",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Define data stewardship: reviews, audits, maker-checker pairs, and clear ownership at function, group, or individual level.",
        "skill_workflow": None,
        "ingress_schema": {"data_entities": "list", "org_structure": "dict (optional)", "existing_policies": "dict (optional)"},
        "egress_schema": {"stewardship_map": "dict", "review_workflows": "list", "ownership_assignments": "dict", "audit_schedule": "dict"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "Defines ownership and review processes. Requires HITL to confirm organizational assignments.",
    },

    # ── Knowledge / Ontology ──────────────────────────────────────────────
    {
        "agent_id": "AG23",
        "name": "CustomerOntologyBuilder",
        "type_": "worker",
        "lifecycle_stage": "enrichment",
        "domain": "Any",
        "capability": "Build, update, and maintain customer knowledge graph representing customer data in terms of entities, relationships, and properties for richer traversals, insights, and standardization.",
        "skill_workflow": "Uses hooks/domain_graph.py",
        "ingress_schema": {"customer_data": "DataFrame or schema", "domain_ctx": "DCP", "core_ontology": "from AG24 (optional)"},
        "egress_schema": {"customer_graph": "dict", "entities": "list", "relationships": "list", "properties": "list", "graph_stats": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Customer-specific ontology. Builds on top of core domain ontology from AG24 when available.",
    },
    {
        "agent_id": "AG24",
        "name": "CoreOntologyBuilder",
        "type_": "worker",
        "lifecycle_stage": "enrichment",
        "domain": "Any",
        "capability": "Build, update, and maintain domain or industry knowledge graph representing industry entities, relationships, and properties for standardization and insight traversals.",
        "skill_workflow": "Uses hooks/domain_graph.py",
        "ingress_schema": {"domain": "str", "industry_standard": "str or dict (optional)", "seed_data": "dict (optional)"},
        "egress_schema": {"core_graph": "dict", "entities": "list", "relationships": "list", "terminology": "list", "invariants": "list"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Replaces WA7 (DomainGraph.Traversal). Builds and retrieves from core domain knowledge graph.",
    },

    # ── Enrichment ────────────────────────────────────────────────────────
    {
        "agent_id": "AG25",
        "name": "DataEnricher.External",
        "type_": "worker",
        "lifecycle_stage": "enrichment",
        "domain": "Any",
        "capability": "Enrich business data by calling external APIs: Precisely, Google Maps, Address Doctor, D&B, etc. Resolve inconsistencies and complete data for entities.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "enrichment_config": "dict (apis, columns, strategies)", "domain_ctx": "DCP"},
        "egress_schema": {"enriched_dataset": "DataFrame", "enrichment_log": "list[dict]", "api_call_summary": "dict", "enrichment_rate": "float"},
        "runtime_hint": "async",
        "hitl_required": False,
        "notes": "Calls external APIs — may have cost implications. Runs async for large datasets. Track API costs via AG39.",
    },
    {
        "agent_id": "AG26",
        "name": "DataEnricher.Internal",
        "type_": "worker",
        "lifecycle_stage": "enrichment",
        "domain": "Any",
        "capability": "Enrich data through internal techniques: knowledge graph, domain model, geo-coding, geo-fencing through lat-long boundaries, and other internal transformations.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "graph_ctx": "from AG24/AG23 (optional)", "enrichment_rules": "dict"},
        "egress_schema": {"enriched_dataset": "DataFrame", "enrichment_log": "list[dict]", "enrichment_rate": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "No external API calls — uses internal knowledge. Geo-coding, graph-based inference, model-based completion.",
    },

    # ── MDM / 360 ─────────────────────────────────────────────────────────
    {
        "agent_id": "AG27",
        "name": "X360",
        "type_": "worker",
        "lifecycle_stage": "enrichment",
        "domain": "Any",
        "capability": "Provide a 360-degree view of an entity: unify master, transactional, and activity/events data into a comprehensive entity profile.",
        "skill_workflow": None,
        "ingress_schema": {"entity_type": "str", "master_data": "DataFrame", "transactional_data": "DataFrame (optional)", "events_data": "DataFrame (optional)", "graph_ctx": "dict (optional)"},
        "egress_schema": {"entity_360": "dict", "unified_profile": "DataFrame", "data_source_map": "dict", "completeness_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Combines multiple data sources into a single 360 view per entity.",
    },
    {
        "agent_id": "AG28",
        "name": "GoldenDataManager",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Export and maintain golden data with a store. Manage cleaned and enriched data as golden records that are referenceable and ensure regulatory compliance.",
        "skill_workflow": None,
        "ingress_schema": {"cleaned_data": "DataFrame", "enrichment_log": "dict (optional)", "compliance_rules": "dict (optional)", "golden_store_config": "dict"},
        "egress_schema": {"golden_records": "DataFrame", "golden_store_status": "dict", "compliance_report": "dict", "version_info": "dict"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "Writes golden data — always requires HITL approval. Ensures compliance to regulations.",
    },

    # ── Intelligence ──────────────────────────────────────────────────────
    {
        "agent_id": "AG29",
        "name": "NBARecommender",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Recommend next best action or offer using rules, heuristics, filtering, or AI/ML. Provides reason codes, confidence, memory, and learning from feedback loops.",
        "skill_workflow": None,
        "ingress_schema": {"entity_data": "dict or DataFrame", "context": "dict", "rules": "dict (optional)", "model_config": "dict (optional)"},
        "egress_schema": {"recommendations": "list[dict]", "reason_codes": "list", "confidence_scores": "list[float]", "model_used": "str"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Supports rule-based, heuristic, collaborative filtering, and ML-based recommendations. Learns from feedback.",
    },
    {
        "agent_id": "AG30",
        "name": "ContentGenerator",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Generate content: product/customer/entity descriptions to various lengths and detail levels (one-liner, title, summary, detailed). Ensure completeness, accuracy, and reliability of generated content.",
        "skill_workflow": None,
        "ingress_schema": {"entity_data": "dict or DataFrame", "content_type": "str (title/summary/detailed)", "target_audience": "str (optional)", "media_inputs": "list (optional: text, data, images)"},
        "egress_schema": {"generated_content": "list[dict]", "content_quality_score": "float", "factuality_check": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Multi-modal input support (text, data, images). Includes factuality validation.",
    },
    {
        "agent_id": "AG31",
        "name": "ContentOptimizer",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Optimize content for business objectives: SEO, customer ratings, referrals, funnel improvement. Optimize for impressions, visits, time spent, and customer engagement through configurability and personalization.",
        "skill_workflow": None,
        "ingress_schema": {"content": "from AG30", "optimization_goal": "str (seo/engagement/conversion)", "target_platform": "str (optional)"},
        "egress_schema": {"optimized_content": "list[dict]", "optimization_report": "dict", "predicted_impact": "dict"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Works after ContentGenerator. Optimizes for discoverability, engagement, and business KPIs.",
    },

    # ── Security ──────────────────────────────────────────────────────────
    {
        "agent_id": "AG32",
        "name": "SensitiveDataAnalyzer",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Identify sensitive and protectable data including PII from regulatory, information security, compliance, and other points of view. Mark importance and data loss risk.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "regulations": "list (optional: GDPR, HIPAA, CCPA)", "custom_rules": "dict (optional)"},
        "egress_schema": {"sensitivity_report": "dict", "pii_columns": "list", "risk_classification": "dict", "data_loss_risk_score": "float"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Must run before any data sharing or syndication. Regulatory awareness (GDPR, HIPAA, CCPA).",
    },
    {
        "agent_id": "AG33",
        "name": "DataLossProtector",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Protect sensitive data through various strategies: masking, obfuscation, encryption, and differential privacy.",
        "skill_workflow": None,
        "ingress_schema": {"dataset": "DataFrame", "sensitivity_report": "from AG32", "protection_strategy": "dict (masking/obfuscation/encryption/differential_privacy)"},
        "egress_schema": {"protected_dataset": "DataFrame", "protection_log": "list[dict]", "reversibility_info": "dict"},
        "runtime_hint": "sync",
        "hitl_required": True,
        "notes": "Modifies data for privacy — requires HITL approval. Must preserve data utility for downstream use.",
    },

    # ── Publishing ────────────────────────────────────────────────────────
    {
        "agent_id": "AG34",
        "name": "CatalogManager",
        "type_": "worker",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Create, maintain, and update catalog and documentation of all data. Provide reference and help users understand data better.",
        "skill_workflow": None,
        "ingress_schema": {"data_assets": "list[dict]", "profiling_output": "from AG07 (optional)", "existing_catalog": "dict (optional)"},
        "egress_schema": {"catalog": "dict", "documentation": "dict", "catalog_diff": "dict (if updating)"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Central catalog management. Integrates profiling output for auto-documentation.",
    },
    {
        "agent_id": "AG35",
        "name": "Syndicator",
        "type_": "worker",
        "lifecycle_stage": "preparation",
        "domain": "Any",
        "capability": "Syndicate clean and trustworthy data to downstream consumers through entitlements, policies, and commercial models.",
        "skill_workflow": None,
        "ingress_schema": {"golden_data": "from AG28", "target_consumers": "list[dict]", "entitlements": "dict", "commercial_model": "dict (optional)"},
        "egress_schema": {"syndication_status": "dict", "delivery_log": "list[dict]", "consumer_ack": "dict"},
        "runtime_hint": "async",
        "hitl_required": True,
        "notes": "Pushes data externally — always requires HITL approval. Enforces entitlements and policies.",
    },

    # ── Analytics ─────────────────────────────────────────────────────────
    {
        "agent_id": "AG36",
        "name": "DataAnalysis",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Generate hidden patterns, relations, and insights from data. Report in multi-modal way through text, charts, tables, and graphics via UI.",
        "skill_workflow": "data_analysis workflows A-D",
        "ingress_schema": {"dataset": "DataFrame or path", "analysis_type": "str (profile/anomaly/survival/custom)", "domain_ctx": "DCP"},
        "egress_schema": {"insights": "list[dict]", "charts": "list[str (paths)]", "summary_report": "str", "correlation_report": "DataFrame (optional)"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "General-purpose analysis agent. Bridges to data_analysis skill workflows.",
    },
    {
        "agent_id": "AG37",
        "name": "RAGAgent",
        "type_": "worker",
        "lifecycle_stage": "enrichment",
        "domain": "Any",
        "capability": "Build a RAG with vector database and chunking from various input forms (files, databases, documents) for semantic retrieval and LLM-powered question answering.",
        "skill_workflow": None,
        "ingress_schema": {"data_sources": "list[dict (type, path)]", "chunk_config": "dict (optional)", "vector_db_config": "dict (optional)"},
        "egress_schema": {"rag_index": "dict (index_id, doc_count, chunk_count)", "query_interface": "callable", "build_report": "dict"},
        "runtime_hint": "async",
        "hitl_required": False,
        "notes": "Builds RAG index for downstream Search and Q&A. Supports multiple input formats.",
    },
    {
        "agent_id": "AG38",
        "name": "Search",
        "type_": "worker",
        "lifecycle_stage": "analytics",
        "domain": "Any",
        "capability": "Enable semantic and cognitive search on all data artifacts using mixed-mode search (hybrid, filtered, semantic). Allow users to search and retrieve all information.",
        "skill_workflow": None,
        "ingress_schema": {"query": "str", "search_mode": "str (hybrid/semantic/filtered)", "rag_index": "from AG37 (optional)", "data_sources": "list (optional)"},
        "egress_schema": {"results": "list[dict]", "relevance_scores": "list[float]", "facets": "dict (optional)"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Multi-mode search across all data. Uses RAG index when available.",
    },

    # ── Operations ────────────────────────────────────────────────────────
    {
        "agent_id": "AG39",
        "name": "FinOps",
        "type_": "worker",
        "lifecycle_stage": "any",
        "domain": "Any",
        "capability": "Provide cost of processing (compute, storage, I/O, network, human oversight) to all other agents, tools, and processors/executors.",
        "skill_workflow": None,
        "ingress_schema": {"agents_involved": "list[str]", "resource_usage": "dict (optional)", "time_window": "str (optional)"},
        "egress_schema": {"cost_report": "dict", "resource_breakdown": "dict", "optimization_suggestions": "list"},
        "runtime_hint": "sync",
        "hitl_required": False,
        "notes": "Cross-cutting agent. Can be added to any plan for cost tracking and optimization.",
    },
    # ── Research ──────────────────────────────────────────────────────────
    {
        "agent_id": "AG40",
        "name": "ResearchOrchestrator",
        "type_": "cognitive",
        "lifecycle_stage": "ml_modeling",
        "domain": "Any",
        "capability": "Structured 4-phase ML research (Understand, Design, Prioritize, Execute) across 7 improvement surfaces: Data, Features, Architectures, Models, Parameters, Validation, Interpretation. Multi-iteration self-stopping. Produces research_project.md and populates experiment ledger.",
        "skill_workflow": "research_orchestrator SKILL.md (6-step protocol)",
        "ingress_schema": {"dataset": "DataFrame or path", "domain_ctx": "DCP", "intent_spec": "IS", "target_col": "str (optional)", "budget": "dict (optional)"},
        "egress_schema": {"research_result": "dict", "stop_reason": "str", "research_project_md": "str path", "research_summary": "str"},
        "runtime_hint": "async",
        "hitl_required": False,
        "notes": "Has 4 sub-agents (RA1-RA4) and L3 experiment ledger. Invocable as /research skill or by SA3. Wraps hooks/research_orchestrator.py.",
    },
]

# ---------------------------------------------------------------------------
# Plan Templates (~16)
# ---------------------------------------------------------------------------

PLANS = [
    # 1. Data Profiling Pipeline
    {
        "name": "data_profiling_pipeline",
        "label": "Data Profiling Pipeline",
        "description": "Connect to data source, sample, perform initial and full profiling. Produces comprehensive profiling report.",
        "intent_patterns": [
            "dataset_profiling", "data_quality_assessment", "data_profiling",
            "exploratory_data_analysis", "data_understanding",
        ],
        "lifecycle_stages": ["analytics", "preparation"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 2. Data Quality Assessment
    {
        "name": "data_quality_assessment",
        "label": "Data Quality Assessment",
        "description": "Comprehensive quality assessment: completeness, correctness, anomalies, recency, and unified quality score.",
        "intent_patterns": [
            "data_quality_assessment", "quality_audit", "data_quality_check",
            "completeness_check", "correctness_check",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 3. Schema Conformance & Harmonization
    {
        "name": "schema_conformance",
        "label": "Schema Conformance & Harmonization",
        "description": "Compare customer schema to industry standard or domain model, recommend and enforce schema changes.",
        "intent_patterns": [
            "schema_validation", "schema_conformance", "schema_harmonization",
            "data_model_alignment", "standard_compliance",
        ],
        "lifecycle_stages": ["preparation"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 4. Data Quality Remediation
    {
        "name": "data_quality_remediation",
        "label": "Data Quality Remediation",
        "description": "Fix data quality issues using rules and AI, re-score after fixes. Full remediation cycle.",
        "intent_patterns": [
            "data_quality_fix", "data_cleansing", "data_remediation",
            "missing_value_treatment", "deduplication",
        ],
        "lifecycle_stages": ["preparation"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 5. MDM Entity Resolution & Golden Record
    {
        "name": "mdm_entity_resolution",
        "label": "MDM Entity Resolution & Golden Record",
        "description": "Full MDM lifecycle: ontology-grounded entity matching, HITL review, 360-degree view, golden record creation.",
        "intent_patterns": [
            "entity_resolution", "record_deduplication", "data_deduplication",
            "master_data_consolidation", "golden_record_creation",
        ],
        "lifecycle_stages": ["enrichment", "preparation"],
        "domains": ["Master Data Management", "Any"],
        "requires_dataset": True,
    },
    # 6. Data Enrichment Pipeline
    {
        "name": "data_enrichment",
        "label": "Data Enrichment Pipeline",
        "description": "Enrich data through external APIs and internal techniques, then re-score quality.",
        "intent_patterns": [
            "data_enrichment", "data_augmentation", "address_validation",
            "geo_enrichment", "entity_enrichment",
        ],
        "lifecycle_stages": ["enrichment"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 7. Knowledge Graph Setup
    {
        "name": "knowledge_graph_setup",
        "label": "Knowledge Graph Setup",
        "description": "Build core domain ontology and customer-specific ontology for data representation and traversal.",
        "intent_patterns": [
            "knowledge_graph_build", "ontology_creation", "domain_modeling",
            "taxonomy_exploration", "knowledge_synthesis",
        ],
        "lifecycle_stages": ["enrichment", "knowledge_reference"],
        "domains": ["Any"],
        "requires_dataset": False,
    },
    # 8. Fit for Use Assessment
    {
        "name": "fit_for_use_assessment",
        "label": "Fit for Use Assessment",
        "description": "End-to-end data fitness assessment: profile, quality check across all dimensions, and fitness verdict with remediation plan.",
        "intent_patterns": [
            "fit_for_use", "data_readiness", "ml_readiness",
            "data_fitness", "data_sufficiency",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 9. Data Governance Setup
    {
        "name": "data_governance",
        "label": "Data Governance Setup",
        "description": "Establish lineage, stewardship, ownership, and catalog for effective data governance.",
        "intent_patterns": [
            "data_governance", "data_lineage", "data_stewardship",
            "data_ownership", "audit_trail",
        ],
        "lifecycle_stages": ["any"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 10. Content Generation & Optimization
    {
        "name": "content_generation",
        "label": "Content Generation & Optimization",
        "description": "Generate entity descriptions and optimize content for business objectives (SEO, engagement, conversion).",
        "intent_patterns": [
            "content_generation", "product_description", "entity_description",
            "seo_optimization", "content_optimization",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 11. Data Security & Privacy
    {
        "name": "data_security",
        "label": "Data Security & Privacy",
        "description": "Identify sensitive data (PII), assess risk, and apply protection strategies (masking, encryption, differential privacy).",
        "intent_patterns": [
            "pii_detection", "sensitive_data_scan", "data_privacy",
            "data_masking", "data_protection",
        ],
        "lifecycle_stages": ["analytics", "preparation"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 12. Data Syndication
    {
        "name": "data_syndication",
        "label": "Data Syndication Pipeline",
        "description": "Publish golden data to downstream consumers with entitlements, cataloging, and compliance.",
        "intent_patterns": [
            "data_syndication", "data_publishing", "data_distribution",
            "downstream_delivery",
        ],
        "lifecycle_stages": ["preparation"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 13. Analytics & Insights
    {
        "name": "analytics_insights",
        "label": "Analytics & Insights",
        "description": "Analyze data for hidden patterns, generate insights, build RAG for Q&A, and enable search.",
        "intent_patterns": [
            "data_analysis", "exploratory_data_analysis", "pattern_detection",
            "insight_generation", "comprehensive_analysis",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 14. Recommendation Engine
    {
        "name": "recommendation_engine",
        "label": "Recommendation Engine",
        "description": "Build and serve next-best-action or next-best-offer recommendations using data analysis and AI/ML.",
        "intent_patterns": [
            "recommendation", "next_best_action", "next_best_offer",
            "product_recommendation", "cross_sell", "upsell",
        ],
        "lifecycle_stages": ["analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 15. Full Data Quality Lifecycle
    {
        "name": "full_data_quality_lifecycle",
        "label": "Full Data Quality Lifecycle",
        "description": "End-to-end: ingest, sample, profile, assess quality across all dimensions, fix issues, re-score, and assess fitness.",
        "intent_patterns": [
            "full_data_quality_lifecycle", "end_to_end_quality",
            "data_quality_pipeline", "comprehensive_quality",
        ],
        "lifecycle_stages": ["preparation", "analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 16. Semantic Search Setup
    {
        "name": "semantic_search_setup",
        "label": "Semantic Search Setup",
        "description": "Build RAG index from data sources and enable multi-mode search (hybrid, semantic, filtered).",
        "intent_patterns": [
            "semantic_search", "search_setup", "rag_build",
            "knowledge_base", "question_answering",
        ],
        "lifecycle_stages": ["enrichment", "analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
    # 17. Research Experiment Pipeline
    {
        "name": "research_experiment_pipeline",
        "label": "Research Experiment Pipeline",
        "description": "Structured ML research: fetch data, profile, then run 4-phase research orchestrator (Understand, Design across 7 surfaces, Prioritize, Execute with stopping). Produces research_project.md and experiment ledger.",
        "intent_patterns": [
            "ml_research", "experiment_protocol", "hypothesis_testing",
            "optimize_model", "systematic_tuning", "research_project",
            "iterative_improvement", "multi_surface_optimization",
        ],
        "lifecycle_stages": ["ml_modeling", "analytics"],
        "domains": ["Any"],
        "requires_dataset": True,
    },
]

# ---------------------------------------------------------------------------
# Steps per plan — (plan_name, step_order, agent_id, purpose, ingress_map,
#                    egress_keys, is_optional, condition, hitl_gate, hitl_reason)
# ---------------------------------------------------------------------------

STEPS = [
    # ── 1. data_profiling_pipeline: AG04 → AG05 → AG06 → AG07
    ("data_profiling_pipeline", 1, "AG04", "Connect to data source and fetch data + metadata",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("data_profiling_pipeline", 2, "AG05", "Sample data if large, preserving distributions",
     {"dataset": "$step.1.egress.dataset", "sample_size": "$input.sample_size"},
     ["sample_df", "sampling_report"], True, "Only if dataset > 100K rows", None, None),
    ("data_profiling_pipeline", 3, "AG06", "Initial profiling: route columns to LLM vs program",
     {"dataset": "$step.2.egress.sample_df", "domain_ctx": "$input.dcp"},
     ["llm_columns", "program_columns", "initial_summary"], False, None, None, None),
    ("data_profiling_pipeline", 4, "AG07", "Full profiling at schema, table, column levels",
     {"dataset": "$step.1.egress.dataset", "column_routing": "$step.3.egress", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "statistical_summary"], False, None, None, None),

    # ── 2. data_quality_assessment: AG04 → AG07 → AG12 || AG13 || AG14 || AG15 → AG17
    ("data_quality_assessment", 1, "AG04", "Fetch data from source",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("data_quality_assessment", 2, "AG07", "Full profiling as quality baseline",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles"], False, None, None, None),
    ("data_quality_assessment", 3, "AG12", "Assess completeness: NULLs, sparsity",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.2.egress"},
     ["completeness_report", "completeness_score"], False, None, None, None),
    ("data_quality_assessment", 3, "AG13", "Assess correctness: precision, consistency, accuracy",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.2.egress", "domain_ctx": "$input.dcp"},
     ["correctness_report", "correctness_score"], False, None, None, None),
    ("data_quality_assessment", 3, "AG14", "Detect anomalies and outliers",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["anomaly_report", "outlier_rows"], False, None, None, None),
    ("data_quality_assessment", 3, "AG15", "Assess data recency and currency",
     {"dataset": "$step.1.egress.dataset", "date_columns": "$step.2.egress.column_profiles"},
     ["recency_report", "recency_score"], True, "Only if date columns present", None, None),
    ("data_quality_assessment", 4, "AG17", "Compute unified quality score",
     {"profiling_output": "$step.2.egress", "completeness": "$step.3.AG12.egress", "correctness": "$step.3.AG13.egress", "anomaly_report": "$step.3.AG14.egress"},
     ["overall_score", "dimension_scores", "score_report"], False, None, None, None),

    # ── 3. schema_conformance: AG04 → AG07 → AG24 → AG08 → AG09 → AG10
    ("schema_conformance", 1, "AG04", "Fetch data and schema",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("schema_conformance", 2, "AG07", "Profile to understand current schema",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("schema_conformance", 2, "AG24", "Build/retrieve core domain ontology",
     {"domain": "$input.dcp.domain", "industry_standard": "$input.industry_standard"},
     ["core_graph", "entities", "terminology"], False, None, None, None),
    ("schema_conformance", 3, "AG08", "Recommend schema changes vs standard",
     {"customer_schema": "$step.1.egress.metadata", "reference_model": "$step.2.AG24.egress.core_graph", "domain_ctx": "$input.dcp"},
     ["gap_analysis", "recommended_entities", "conformance_score"], False, None, None, None),
    ("schema_conformance", 4, "AG09", "Harmonize: apply normative rules",
     {"schema": "$step.1.egress.metadata", "gap_analysis": "$step.3.egress"},
     ["harmonized_schema", "rules_applied"], False, None, None, None),
    ("schema_conformance", 5, "AG10", "Enforce schema fixes via DDL",
     {"current_schema": "$step.1.egress.metadata", "target_schema": "$step.4.egress.harmonized_schema"},
     ["schema_changes", "ddl_statements"], False, None,
     "before", "Review and approve schema DDL changes before execution"),

    # ── 4. data_quality_remediation: AG07 → AG11 → AG16 → AG17
    ("data_quality_remediation", 1, "AG07", "Profile to identify quality issues",
     {"dataset": "$input.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("data_quality_remediation", 2, "AG11", "Create or apply quality rules",
     {"dataset": "$input.dataset", "profiling_output": "$step.1.egress", "domain_ctx": "$input.dcp"},
     ["rules", "rule_validation_results"], False, None,
     "after", "Review AI-suggested rules before applying fixes"),
    ("data_quality_remediation", 3, "AG16", "Fix quality issues using rules and AI",
     {"dataset": "$input.dataset", "quality_issues": "$step.1.egress", "rules": "$step.2.egress.rules"},
     ["fixed_dataset", "fix_log", "before_after_metrics"], False, None,
     "before", "Approve fix strategy before modifying data"),
    ("data_quality_remediation", 4, "AG17", "Re-score quality after fixes",
     {"profiling_output": "$step.1.egress", "completeness": "recomputed", "correctness": "recomputed"},
     ["overall_score", "dimension_scores"], False, None, None, None),

    # ── 5. mdm_entity_resolution: AG04 → AG24 → AG23 → AG27 → AG03 → AG28
    ("mdm_entity_resolution", 1, "AG04", "Fetch records from sources",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("mdm_entity_resolution", 2, "AG24", "Build/retrieve core domain ontology for entity schema",
     {"domain": "$input.dcp.domain"}, ["core_graph", "entities"], False, None, None, None),
    ("mdm_entity_resolution", 3, "AG23", "Build customer-specific ontology for matching",
     {"customer_data": "$step.1.egress.dataset", "core_ontology": "$step.2.egress.core_graph"},
     ["customer_graph", "entities", "relationships"], False, None, None, None),
    ("mdm_entity_resolution", 4, "AG27", "Build 360 view for entity resolution",
     {"entity_type": "$input.entity_type", "master_data": "$step.1.egress.dataset", "graph_ctx": "$step.3.egress.customer_graph"},
     ["entity_360", "unified_profile"], False, None, None, None),
    ("mdm_entity_resolution", 5, "AG03", "HITL review of match clusters and 360 profiles",
     {"hitl_requests": ["Review entity matches", "Confirm survivorship rules"], "agent_outputs": "$step.4.egress"},
     ["human_responses", "approval_status"], False, None, None, None),
    ("mdm_entity_resolution", 6, "AG28", "Create and store golden records",
     {"cleaned_data": "$step.4.egress.unified_profile", "compliance_rules": "$input.compliance_rules"},
     ["golden_records", "golden_store_status", "compliance_report"], False, None,
     "before", "Final approval before writing golden records to store"),

    # ── 6. data_enrichment: AG04 → AG07 → AG25 || AG26 → AG17
    ("data_enrichment", 1, "AG04", "Fetch data to enrich",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("data_enrichment", 2, "AG07", "Profile to understand enrichment opportunities",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("data_enrichment", 3, "AG25", "Enrich via external APIs (Precisely, D&B, Google Maps, etc.)",
     {"dataset": "$step.1.egress.dataset", "enrichment_config": "$input.external_config", "domain_ctx": "$input.dcp"},
     ["enriched_dataset", "enrichment_log", "enrichment_rate"], True, "Only if external APIs configured", None, None),
    ("data_enrichment", 3, "AG26", "Enrich via internal techniques (graph, geo-coding, model)",
     {"dataset": "$step.1.egress.dataset", "graph_ctx": "$input.graph_ctx"},
     ["enriched_dataset", "enrichment_log", "enrichment_rate"], True, "Only if internal enrichment rules available", None, None),
    ("data_enrichment", 4, "AG17", "Re-score quality after enrichment",
     {"profiling_output": "$step.2.egress"},
     ["overall_score", "dimension_scores"], False, None, None, None),

    # ── 7. knowledge_graph_setup: AG24 → AG23
    ("knowledge_graph_setup", 1, "AG24", "Build core domain/industry ontology",
     {"domain": "$input.dcp.domain", "industry_standard": "$input.industry_standard", "seed_data": "$input.seed_data"},
     ["core_graph", "entities", "relationships", "terminology", "invariants"], False, None, None, None),
    ("knowledge_graph_setup", 2, "AG23", "Build customer-specific ontology on top of core",
     {"customer_data": "$input.customer_data", "core_ontology": "$step.1.egress.core_graph"},
     ["customer_graph", "entities", "relationships", "properties"], True, "Only if customer data provided", None, None),

    # ── 8. fit_for_use_assessment: AG04 → AG05 → AG07 → AG12||AG13||AG14 → AG17 → AG20
    ("fit_for_use_assessment", 1, "AG04", "Fetch data",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("fit_for_use_assessment", 2, "AG05", "Sample for efficient assessment",
     {"dataset": "$step.1.egress.dataset"}, ["sample_df"], True, "Sample if > 100K rows", None, None),
    ("fit_for_use_assessment", 3, "AG07", "Full profile",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("fit_for_use_assessment", 4, "AG12", "Completeness assessment",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.3.egress"},
     ["completeness_score"], False, None, None, None),
    ("fit_for_use_assessment", 4, "AG13", "Correctness assessment",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.3.egress"},
     ["correctness_score"], False, None, None, None),
    ("fit_for_use_assessment", 4, "AG14", "Anomaly detection",
     {"dataset": "$step.1.egress.dataset"},
     ["anomaly_report"], False, None, None, None),
    ("fit_for_use_assessment", 5, "AG17", "Compute unified quality score",
     {"profiling_output": "$step.3.egress", "completeness": "$step.4.AG12.egress", "correctness": "$step.4.AG13.egress"},
     ["overall_score", "dimension_scores"], False, None, None, None),
    ("fit_for_use_assessment", 6, "AG20", "Fit for use verdict and remediation plan",
     {"dataset": "$step.1.egress.dataset", "business_objective": "$input.business_objective", "quality_scores": "$step.5.egress", "profiling": "$step.3.egress"},
     ["fit_for_use_report", "readiness_level", "remediation_plan"], False, None, None, None),

    # ── 9. data_governance: AG07 → AG21 → AG22 → AG34
    ("data_governance", 1, "AG07", "Profile data to understand governance scope",
     {"dataset": "$input.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("data_governance", 2, "AG21", "Establish data lineage and provenance",
     {"data_entity": "$input.data_entity", "source_registry": "$input.source_registry"},
     ["lineage_graph", "provenance_report", "integrity_score"], False, None, None, None),
    ("data_governance", 3, "AG22", "Define stewardship, ownership, and review workflows",
     {"data_entities": "$input.data_entities", "org_structure": "$input.org_structure"},
     ["stewardship_map", "ownership_assignments"], False, None,
     "after", "Confirm ownership assignments and review workflows"),
    ("data_governance", 4, "AG34", "Create and update data catalog",
     {"data_assets": "$input.data_assets", "profiling_output": "$step.1.egress"},
     ["catalog", "documentation"], False, None, None, None),

    # ── 10. content_generation: AG04 → AG07 → AG30 → AG31
    ("content_generation", 1, "AG04", "Fetch entity data for content generation",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("content_generation", 2, "AG07", "Profile entities to understand data",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("content_generation", 3, "AG30", "Generate content (descriptions, titles, summaries)",
     {"entity_data": "$step.1.egress.dataset", "content_type": "$input.content_type", "target_audience": "$input.target_audience"},
     ["generated_content", "content_quality_score"], False, None, None, None),
    ("content_generation", 4, "AG31", "Optimize content for business objective",
     {"content": "$step.3.egress.generated_content", "optimization_goal": "$input.optimization_goal"},
     ["optimized_content", "optimization_report", "predicted_impact"], True, "Only if optimization goal specified", None, None),

    # ── 11. data_security: AG04 → AG32 → AG33
    ("data_security", 1, "AG04", "Fetch data for security analysis",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("data_security", 2, "AG32", "Identify sensitive data and PII",
     {"dataset": "$step.1.egress.dataset", "regulations": "$input.regulations"},
     ["sensitivity_report", "pii_columns", "risk_classification", "data_loss_risk_score"], False, None, None, None),
    ("data_security", 3, "AG33", "Apply data protection (masking, encryption, etc.)",
     {"dataset": "$step.1.egress.dataset", "sensitivity_report": "$step.2.egress", "protection_strategy": "$input.protection_strategy"},
     ["protected_dataset", "protection_log"], False, None,
     "before", "Approve protection strategy before modifying data"),

    # ── 12. data_syndication: AG28 → AG34 → AG35
    ("data_syndication", 1, "AG28", "Prepare golden data for syndication",
     {"cleaned_data": "$input.golden_data", "golden_store_config": "$input.store_config"},
     ["golden_records", "compliance_report"], False, None, None, None),
    ("data_syndication", 2, "AG34", "Update catalog for published data",
     {"data_assets": "$step.1.egress.golden_records"},
     ["catalog"], False, None, None, None),
    ("data_syndication", 3, "AG35", "Syndicate to downstream consumers",
     {"golden_data": "$step.1.egress.golden_records", "target_consumers": "$input.target_consumers", "entitlements": "$input.entitlements"},
     ["syndication_status", "delivery_log"], False, None,
     "before", "Approve syndication targets and entitlements before publishing"),

    # ── 13. analytics_insights: AG04 → AG05 → AG36 → AG37
    ("analytics_insights", 1, "AG04", "Fetch data for analysis",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("analytics_insights", 2, "AG05", "Sample for efficient analysis",
     {"dataset": "$step.1.egress.dataset"}, ["sample_df"], True, "Sample if large dataset", None, None),
    ("analytics_insights", 3, "AG36", "Analyze data: patterns, correlations, insights",
     {"dataset": "$step.1.egress.dataset", "analysis_type": "$input.analysis_type", "domain_ctx": "$input.dcp"},
     ["insights", "charts", "summary_report"], False, None, None, None),
    ("analytics_insights", 4, "AG37", "Build RAG for follow-up Q&A",
     {"data_sources": ["$step.1.egress.dataset", "$step.3.egress.summary_report"]},
     ["rag_index", "build_report"], True, "Only if Q&A capability requested", None, None),

    # ── 14. recommendation_engine: AG04 → AG07 → AG36 → AG29
    ("recommendation_engine", 1, "AG04", "Fetch entity and context data",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("recommendation_engine", 2, "AG07", "Profile data to understand features",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report"], False, None, None, None),
    ("recommendation_engine", 3, "AG36", "Analyze data for recommendation signals",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["insights", "correlation_report"], False, None, None, None),
    ("recommendation_engine", 4, "AG29", "Generate recommendations",
     {"entity_data": "$step.1.egress.dataset", "context": "$step.3.egress"},
     ["recommendations", "reason_codes", "confidence_scores"], False, None, None, None),

    # ── 15. full_data_quality_lifecycle: AG04 → AG05 → AG06 → AG07 → AG12||AG13||AG14 → AG16 → AG17 → AG20
    ("full_data_quality_lifecycle", 1, "AG04", "Fetch data",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("full_data_quality_lifecycle", 2, "AG05", "Sample for initial analysis",
     {"dataset": "$step.1.egress.dataset"}, ["sample_df"], True, "Sample if > 100K rows", None, None),
    ("full_data_quality_lifecycle", 3, "AG06", "Initial profiling and column routing",
     {"dataset": "$step.2.egress.sample_df"}, ["llm_columns", "program_columns"], False, None, None, None),
    ("full_data_quality_lifecycle", 4, "AG07", "Full profiling",
     {"dataset": "$step.1.egress.dataset", "column_routing": "$step.3.egress"},
     ["profile_report"], False, None, None, None),
    ("full_data_quality_lifecycle", 5, "AG12", "Completeness assessment",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.4.egress"},
     ["completeness_score"], False, None, None, None),
    ("full_data_quality_lifecycle", 5, "AG13", "Correctness assessment",
     {"dataset": "$step.1.egress.dataset", "profiling_output": "$step.4.egress"},
     ["correctness_score"], False, None, None, None),
    ("full_data_quality_lifecycle", 5, "AG14", "Anomaly detection",
     {"dataset": "$step.1.egress.dataset"},
     ["anomaly_report"], False, None, None, None),
    ("full_data_quality_lifecycle", 6, "AG16", "Fix quality issues",
     {"dataset": "$step.1.egress.dataset", "quality_issues": "$step.5.egress"},
     ["fixed_dataset", "fix_log"], False, None,
     "before", "Approve fix strategy before modifying data"),
    ("full_data_quality_lifecycle", 7, "AG17", "Re-score quality after fixes",
     {"profiling_output": "$step.4.egress"},
     ["overall_score", "dimension_scores"], False, None, None, None),
    ("full_data_quality_lifecycle", 8, "AG20", "Fit for use assessment",
     {"business_objective": "$input.business_objective", "quality_scores": "$step.7.egress"},
     ["fit_for_use_report", "readiness_level", "remediation_plan"], False, None, None, None),

    # ── 16. semantic_search_setup: AG04 → AG37 → AG38
    ("semantic_search_setup", 1, "AG04", "Fetch data sources for indexing",
     {"source_config": "$input.source_config"}, ["dataset"], False, None, None, None),
    ("semantic_search_setup", 2, "AG37", "Build RAG index with vector DB and chunking",
     {"data_sources": "$step.1.egress.dataset", "chunk_config": "$input.chunk_config"},
     ["rag_index", "build_report"], False, None, None, None),
    ("semantic_search_setup", 3, "AG38", "Enable search interface",
     {"rag_index": "$step.2.egress.rag_index"},
     ["results", "relevance_scores"], False, None, None, None),

    # ── 17. research_experiment_pipeline: AG04 → AG07 → AG40
    ("research_experiment_pipeline", 1, "AG04", "Fetch dataset for research",
     {"source_config": "$input.source_config"}, ["dataset", "metadata"], False, None, None, None),
    ("research_experiment_pipeline", 2, "AG07", "Full profiling to understand data landscape",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp"},
     ["profile_report", "column_profiles", "col_classes"], False, None, None, None),
    ("research_experiment_pipeline", 3, "AG40", "Run 4-phase research orchestrator (multi-iteration, self-stopping)",
     {"dataset": "$step.1.egress.dataset", "domain_ctx": "$input.dcp", "intent_spec": "$input.is",
      "col_classes": "$step.2.egress.col_classes", "budget": "$input.budget"},
     ["research_result", "stop_reason", "research_project_md", "research_summary"], False, None, None, None),
]

# ---------------------------------------------------------------------------
# Dependencies — (plan_name, from_agent_id, to_agent_id,
#                  from_step_order, to_step_order, dep_type, description)
# ---------------------------------------------------------------------------

DEPENDENCIES = [
    # data_profiling_pipeline
    ("data_profiling_pipeline", "AG04", "AG05", 1, 2, "data", "Sampler needs fetched dataset"),
    ("data_profiling_pipeline", "AG05", "AG06", 2, 3, "data", "InitialProfiler needs sample"),
    ("data_profiling_pipeline", "AG06", "AG07", 3, 4, "data", "FullProfiler needs column routing"),

    # data_quality_assessment
    ("data_quality_assessment", "AG04", "AG07", 1, 2, "data", "Profiler needs fetched data"),
    ("data_quality_assessment", "AG07", "AG12", 2, 3, "data", "Completeness needs profiling"),
    ("data_quality_assessment", "AG07", "AG13", 2, 3, "data", "Correctness needs profiling"),
    ("data_quality_assessment", "AG07", "AG14", 2, 3, "data", "Anomaly needs profiling context"),
    ("data_quality_assessment", "AG12", "AG17", 3, 4, "data", "Scorer needs completeness"),
    ("data_quality_assessment", "AG13", "AG17", 3, 4, "data", "Scorer needs correctness"),
    ("data_quality_assessment", "AG14", "AG17", 3, 4, "data", "Scorer needs anomaly report"),

    # schema_conformance
    ("schema_conformance", "AG04", "AG07", 1, 2, "data", "Profiler needs data"),
    ("schema_conformance", "AG07", "AG08", 2, 3, "data", "Recommender needs profile"),
    ("schema_conformance", "AG24", "AG08", 2, 3, "data", "Recommender needs core ontology"),
    ("schema_conformance", "AG08", "AG09", 3, 4, "data", "Harmonizer needs gap analysis"),
    ("schema_conformance", "AG09", "AG10", 4, 5, "data", "Enforcer needs harmonized schema"),

    # data_quality_remediation
    ("data_quality_remediation", "AG07", "AG11", 1, 2, "data", "Rules need profiling"),
    ("data_quality_remediation", "AG11", "AG16", 2, 3, "data", "Fixer needs rules"),
    ("data_quality_remediation", "AG16", "AG17", 3, 4, "data", "Re-score after fixes"),

    # mdm_entity_resolution
    ("mdm_entity_resolution", "AG04", "AG24", 1, 2, "data", "Ontology needs data context"),
    ("mdm_entity_resolution", "AG24", "AG23", 2, 3, "data", "Customer ontology needs core"),
    ("mdm_entity_resolution", "AG23", "AG27", 3, 4, "data", "X360 needs customer graph"),
    ("mdm_entity_resolution", "AG27", "AG03", 4, 5, "sequence", "HITL review of matches"),
    ("mdm_entity_resolution", "AG03", "AG28", 5, 6, "hitl_gate", "Golden records after HITL approval"),

    # data_enrichment
    ("data_enrichment", "AG04", "AG07", 1, 2, "data", "Profile before enrichment"),
    ("data_enrichment", "AG07", "AG25", 2, 3, "data", "External enricher needs profile"),
    ("data_enrichment", "AG07", "AG26", 2, 3, "data", "Internal enricher needs profile"),

    # knowledge_graph_setup
    ("knowledge_graph_setup", "AG24", "AG23", 1, 2, "data", "Customer ontology builds on core"),

    # fit_for_use_assessment
    ("fit_for_use_assessment", "AG04", "AG05", 1, 2, "data", "Sample after fetch"),
    ("fit_for_use_assessment", "AG05", "AG07", 2, 3, "data", "Profile after sample"),
    ("fit_for_use_assessment", "AG07", "AG12", 3, 4, "data", "Quality dims need profiling"),
    ("fit_for_use_assessment", "AG07", "AG13", 3, 4, "data", "Quality dims need profiling"),
    ("fit_for_use_assessment", "AG07", "AG14", 3, 4, "data", "Quality dims need profiling"),
    ("fit_for_use_assessment", "AG12", "AG17", 4, 5, "data", "Scorer needs all dims"),
    ("fit_for_use_assessment", "AG13", "AG17", 4, 5, "data", "Scorer needs all dims"),
    ("fit_for_use_assessment", "AG14", "AG17", 4, 5, "data", "Scorer needs all dims"),
    ("fit_for_use_assessment", "AG17", "AG20", 5, 6, "data", "FFU needs quality score"),

    # data_governance
    ("data_governance", "AG07", "AG21", 1, 2, "data", "Lineage needs profile scope"),
    ("data_governance", "AG21", "AG22", 2, 3, "sequence", "Stewardship after lineage"),
    ("data_governance", "AG22", "AG34", 3, 4, "sequence", "Catalog after stewardship"),

    # content_generation
    ("content_generation", "AG04", "AG07", 1, 2, "data", "Profile entity data"),
    ("content_generation", "AG07", "AG30", 2, 3, "data", "Generator needs profile"),
    ("content_generation", "AG30", "AG31", 3, 4, "data", "Optimizer needs generated content"),

    # data_security
    ("data_security", "AG04", "AG32", 1, 2, "data", "Analyzer needs data"),
    ("data_security", "AG32", "AG33", 2, 3, "data", "Protector needs sensitivity report"),

    # data_syndication
    ("data_syndication", "AG28", "AG34", 1, 2, "data", "Catalog golden data"),
    ("data_syndication", "AG34", "AG35", 2, 3, "data", "Syndicate after cataloging"),

    # analytics_insights
    ("analytics_insights", "AG04", "AG05", 1, 2, "data", "Sample after fetch"),
    ("analytics_insights", "AG05", "AG36", 2, 3, "data", "Analyze sample or full data"),
    ("analytics_insights", "AG36", "AG37", 3, 4, "data", "RAG from analysis output"),

    # recommendation_engine
    ("recommendation_engine", "AG04", "AG07", 1, 2, "data", "Profile first"),
    ("recommendation_engine", "AG07", "AG36", 2, 3, "data", "Analysis needs profile"),
    ("recommendation_engine", "AG36", "AG29", 3, 4, "data", "Recommender needs analysis signals"),

    # full_data_quality_lifecycle
    ("full_data_quality_lifecycle", "AG04", "AG05", 1, 2, "data", "Sample after fetch"),
    ("full_data_quality_lifecycle", "AG05", "AG06", 2, 3, "data", "InitialProfiler after sample"),
    ("full_data_quality_lifecycle", "AG06", "AG07", 3, 4, "data", "FullProfiler after initial"),
    ("full_data_quality_lifecycle", "AG07", "AG12", 4, 5, "data", "Quality dims need profile"),
    ("full_data_quality_lifecycle", "AG07", "AG13", 4, 5, "data", "Quality dims need profile"),
    ("full_data_quality_lifecycle", "AG07", "AG14", 4, 5, "data", "Quality dims need profile"),
    ("full_data_quality_lifecycle", "AG12", "AG16", 5, 6, "data", "Fixer needs completeness issues"),
    ("full_data_quality_lifecycle", "AG13", "AG16", 5, 6, "data", "Fixer needs correctness issues"),
    ("full_data_quality_lifecycle", "AG14", "AG16", 5, 6, "data", "Fixer needs anomaly report"),
    ("full_data_quality_lifecycle", "AG16", "AG17", 6, 7, "data", "Re-score after fix"),
    ("full_data_quality_lifecycle", "AG17", "AG20", 7, 8, "data", "FFU needs final score"),

    # semantic_search_setup
    ("semantic_search_setup", "AG04", "AG37", 1, 2, "data", "RAG needs data"),
    ("semantic_search_setup", "AG37", "AG38", 2, 3, "data", "Search needs RAG index"),

    # research_experiment_pipeline
    ("research_experiment_pipeline", "AG04", "AG07", 1, 2, "data", "Profiler needs fetched data"),
    ("research_experiment_pipeline", "AG07", "AG40", 2, 3, "data", "Research orchestrator needs profiling + col_classes"),
]

# ---------------------------------------------------------------------------
# Few-Shot Examples (~20)
#
# These examples teach the LLM reasoning engine in resolve_execution_plan()
# how to select the right plan given an intent + domain context.
#
# Coverage:
#   - Every plan template has at least one example
#   - Domain diversity: Healthcare, Financial, Retail, Industrial, Insurance, MDM
#   - Override scenarios: adding agents, removing optional steps, modifying gates
#   - Ambiguous intents: where two plans are plausible
#   - Compound workflows: where user intent spans multiple plans
#   - Cross-cutting agents: AG01 (Evaluator), AG03 (ESA), AG39 (FinOps)
# ---------------------------------------------------------------------------

EXAMPLES = [
    # ── 1. DATA PROFILING — vendor data ingest ──────────────────────────────
    {
        "example_name": "profile_vendor_dataset",
        "intent_context": {
            "matched_node": {"name": "dataset_profiling", "label": "Dataset Profiling", "level": "specific"},
            "confidence": 0.92,
            "abstract_parents": [{"name": "exploratory_analytics", "label": "Exploratory Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "We received a new vendor file with patient demographics. Profile it and tell us what we're working with.",
        },
        "selected_plan": "data_profiling_pipeline",
        "reasoning": "User wants initial understanding of new data. data_profiling_pipeline: AG04 (fetch) -> AG05 (sample if large) -> AG06 (initial LLM-routed profiling) -> AG07 (full profile). Graph is available but not needed for basic profiling — skip AG24. Direct intent match on 'dataset_profiling'.",
        "plan_overrides": None,
    },
    # ── 2. QUALITY ASSESSMENT — regulatory audit ────────────────────────────
    {
        "example_name": "regulatory_quality_audit",
        "intent_context": {
            "matched_node": {"name": "data_quality_assessment", "label": "Data Quality Assessment", "level": "specific"},
            "confidence": 0.90,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Financial Services / Lending",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Audit the quality of our lending portfolio data before the OCC examination. Need completeness, correctness, and a unified score.",
        },
        "selected_plan": "data_quality_assessment",
        "reasoning": "Regulatory audit requires all quality dimensions. data_quality_assessment: AG04 -> AG07 (profile) -> AG12 (completeness) || AG13 (correctness) || AG14 (anomaly) || AG15 (recency) -> AG17 (score). Add AG15 recency since lending data has time-sensitive regulations. All quality dims run in parallel after profiling.",
        "plan_overrides": {
            "modify_steps": [
                {"agent_id": "AG15", "changes": {"is_optional": False, "condition": None}},
            ],
        },
    },
    # ── 3. SCHEMA CONFORMANCE — FHIR alignment ─────────────────────────────
    {
        "example_name": "fhir_schema_alignment",
        "intent_context": {
            "matched_node": {"name": "schema_conformance", "label": "Schema Conformance", "level": "specific"},
            "confidence": 0.87,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "industry_standard": "FHIR",
            "user_query": "We need to align our EMR data schema to HL7 FHIR for interoperability. Show gaps and recommend changes.",
        },
        "selected_plan": "schema_conformance",
        "reasoning": "FHIR conformance requires schema gap analysis. schema_conformance: AG04 (fetch) -> AG07 (profile) || AG24 (FHIR ontology) in parallel -> AG08 (recommend gaps) -> AG09 (harmonize rules) -> AG10 (enforce DDL). HITL required before AG10 applies DDL. AG24 builds FHIR reference model for AG08 comparison.",
        "plan_overrides": None,
    },
    # ── 4. QUALITY REMEDIATION — fix dirty CRM data ─────────────────────────
    {
        "example_name": "crm_data_cleansing",
        "intent_context": {
            "matched_node": {"name": "data_quality_fix", "label": "Data Quality Fix", "level": "specific"},
            "confidence": 0.89,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Retail & E-commerce",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Our CRM data has lots of missing values, inconsistent phone formats, and duplicate addresses. Fix it.",
        },
        "selected_plan": "data_quality_remediation",
        "reasoning": "User explicitly wants fixes, not just assessment. data_quality_remediation: AG07 (profile to find issues) -> AG11 (create rules for phone format, address normalization) -> AG16 (apply fixes: missing values, dedup, normalization) -> AG17 (re-score). HITL after AG11 to confirm rules, and before AG16 to approve fix strategy. Data modification requires human approval.",
        "plan_overrides": None,
    },
    # ── 5. MDM — multi-hospital patient dedup ───────────────────────────────
    {
        "example_name": "multi_hospital_patient_dedup",
        "intent_context": {
            "matched_node": {"name": "entity_resolution", "label": "Entity Resolution", "level": "specific"},
            "confidence": 0.91,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Deduplicate patient records across 3 hospital systems and create a golden patient master.",
        },
        "selected_plan": "mdm_entity_resolution",
        "reasoning": "Multi-source entity resolution. mdm_entity_resolution: AG04 (fetch from 3 sources) -> AG24 (healthcare ontology for patient entity schema) -> AG23 (customer-specific patient graph) -> AG27 (360 patient view across systems) -> AG03 (HITL: review match clusters, especially edge cases like name variants) -> AG28 (golden patient records). HITL gate is critical here — false matches in patient data are dangerous.",
        "plan_overrides": None,
    },
    # ── 6. ENRICHMENT — address validation via external API ─────────────────
    {
        "example_name": "address_enrichment_external",
        "intent_context": {
            "matched_node": {"name": "data_enrichment", "label": "Data Enrichment", "level": "specific"},
            "confidence": 0.88,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Retail & E-commerce",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Validate and standardize customer addresses using Precisely or Google Maps API. Also geocode for delivery zone mapping.",
        },
        "selected_plan": "data_enrichment",
        "reasoning": "Address validation needs external API. data_enrichment: AG04 (fetch) -> AG07 (profile to see address column quality) -> AG25 (external: Precisely/Google Maps for address standardization + geocoding) -> AG17 (re-score). AG26 internal enricher not needed since user specifically asked for external APIs. Remove AG26 to save cost — track API spend via AG39.",
        "plan_overrides": {
            "remove_agents": ["AG26"],
            "add_steps": [{"agent_id": "AG39", "step_order": 5, "purpose": "Track external API costs from AG25", "is_optional": True}],
        },
    },
    # ── 7. KNOWLEDGE GRAPH — industrial product taxonomy ────────────────────
    {
        "example_name": "industrial_taxonomy_build",
        "intent_context": {
            "matched_node": {"name": "ontology_creation", "label": "Ontology Creation", "level": "specific"},
            "confidence": 0.86,
            "abstract_parents": [{"name": "knowledge_synthesis", "label": "Knowledge Synthesis"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Commercial / Industrial",
            "query_has_dataset": False,
            "graph_available": False,
            "user_query": "Build a knowledge graph for our electrical components product taxonomy so we can standardize part lookups.",
        },
        "selected_plan": "knowledge_graph_setup",
        "reasoning": "No dataset — pure ontology construction. knowledge_graph_setup: AG24 (build core industry ontology for electrical components: entities, relationships, terminology) -> AG23 (customer-specific product taxonomy on top). AG23 is optional but include since user wants customer-specific part lookups. No HITL needed — read-only construction.",
        "plan_overrides": {
            "modify_steps": [{"agent_id": "AG23", "changes": {"is_optional": False}}],
        },
    },
    # ── 8. FIT FOR USE — ML readiness check ─────────────────────────────────
    {
        "example_name": "ml_readiness_check",
        "intent_context": {
            "matched_node": {"name": "fit_for_use", "label": "Fit for Use", "level": "specific"},
            "confidence": 0.88,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Is our patient data good enough to build a readmission prediction model? Check quality, class balance, feature relevancy, and noise.",
            "business_objective": "30-day readmission prediction model",
        },
        "selected_plan": "fit_for_use_assessment",
        "reasoning": "ML fitness assessment. fit_for_use_assessment: AG04 (fetch) -> AG05 (sample) -> AG07 (profile) -> AG12 (completeness) || AG13 (correctness) || AG14 (anomaly) in parallel -> AG17 (unified score) -> AG20 (10-dimension FFU: quality, readiness, feature relevancy, label quality, class balance, bias, noise, sufficiency). AG20 produces actionable remediation plan if data not ready. Business objective (readmission prediction) is critical input to AG20 for fitness evaluation.",
        "plan_overrides": None,
    },
    # ── 9. DATA GOVERNANCE — post-merger integration ────────────────────────
    {
        "example_name": "post_merger_governance",
        "intent_context": {
            "matched_node": {"name": "data_governance", "label": "Data Governance", "level": "specific"},
            "confidence": 0.85,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Financial Services / Lending",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "After our bank merger, we need to establish data governance: lineage from both legacy systems, ownership mapping to new org structure, and a unified catalog.",
        },
        "selected_plan": "data_governance",
        "reasoning": "Post-merger governance setup. data_governance: AG07 (profile to understand data scope from both systems) -> AG21 (lineage: trace from both legacy sources with transformation history) -> AG22 (stewardship: map ownership to new org structure, define maker-checker pairs) -> AG34 (build unified catalog). HITL after AG22 — ownership assignments must be confirmed by management. AG03 (ESA) consolidates the review since both AG21 findings and AG22 assignments need human sign-off together.",
        "plan_overrides": {
            "add_steps": [{"agent_id": "AG03", "step_order": 3, "purpose": "Consolidate HITL: review lineage findings + confirm ownership assignments in single review", "hitl_gate": "before", "hitl_reason": "Merger governance decisions require management approval"}],
        },
    },
    # ── 10. CONTENT GENERATION — product catalog SEO ────────────────────────
    {
        "example_name": "product_catalog_seo",
        "intent_context": {
            "matched_node": {"name": "content_generation", "label": "Content Generation", "level": "specific"},
            "confidence": 0.90,
            "abstract_parents": [],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Retail & E-commerce",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Generate SEO-optimized product descriptions for our 50K item catalog. Need titles, summaries, and detailed descriptions.",
            "content_type": "all_levels",
            "optimization_goal": "seo",
        },
        "selected_plan": "content_generation",
        "reasoning": "Product content at multiple detail levels plus SEO. content_generation: AG04 (fetch product data) -> AG07 (profile to understand product attributes) -> AG30 (generate titles, summaries, detailed descriptions for 50K items) -> AG31 (SEO-optimize all generated content for search ranking). Both AG30 and AG31 are needed. AG39 (FinOps) recommended for 50K items — LLM generation costs could be significant.",
        "plan_overrides": {
            "add_steps": [{"agent_id": "AG39", "step_order": 5, "purpose": "Track content generation costs across 50K items", "is_optional": True}],
        },
    },
    # ── 11. DATA SECURITY — HIPAA compliance scan ───────────────────────────
    {
        "example_name": "hipaa_compliance_scan",
        "intent_context": {
            "matched_node": {"name": "pii_detection", "label": "PII Detection", "level": "specific"},
            "confidence": 0.93,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": False,
            "regulations": ["HIPAA"],
            "user_query": "Scan our clinical trial dataset for PHI/PII before sharing with the research partner. Mask everything that's HIPAA-sensitive.",
        },
        "selected_plan": "data_security",
        "reasoning": "HIPAA-driven PII scan and protection. data_security: AG04 (fetch clinical data) -> AG32 (identify PHI: patient names, MRNs, dates of birth, SSN, addresses, etc. per HIPAA Safe Harbor) -> AG33 (apply masking/de-identification per HIPAA expert determination method). HITL gate before AG33 is critical — wrong masking strategy could either expose PHI or destroy research utility. AG01 (Evaluator) added to validate de-identification completeness post-masking.",
        "plan_overrides": {
            "add_steps": [{"agent_id": "AG01", "step_order": 4, "purpose": "Validate de-identification completeness: verify no residual PHI in masked output"}],
        },
    },
    # ── 12. DATA SYNDICATION — publish golden data ──────────────────────────
    {
        "example_name": "publish_golden_data_downstream",
        "intent_context": {
            "matched_node": {"name": "data_syndication", "label": "Data Syndication", "level": "specific"},
            "confidence": 0.89,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Master Data Management",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Publish our cleaned golden product catalog to the analytics team's data lake and the e-commerce platform's feed.",
        },
        "selected_plan": "data_syndication",
        "reasoning": "Two-target syndication. data_syndication: AG28 (prepare golden data with compliance) -> AG34 (update catalog with published data documentation) -> AG35 (syndicate to analytics data lake + e-commerce feed). HITL before AG35 — external push to two consumers needs approval on entitlements and data scope per consumer. AG03 (ESA) consolidates: one review for both syndication targets.",
        "plan_overrides": None,
    },
    # ── 13. ANALYTICS — hidden pattern discovery ────────────────────────────
    {
        "example_name": "sales_pattern_discovery",
        "intent_context": {
            "matched_node": {"name": "data_analysis", "label": "Data Analysis", "level": "specific"},
            "confidence": 0.87,
            "abstract_parents": [{"name": "exploratory_analytics", "label": "Exploratory Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Retail & E-commerce",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Analyze our 2-year sales data. Find hidden patterns, seasonal trends, product affinities, and anomalous transactions. Need charts and a summary report.",
        },
        "selected_plan": "analytics_insights",
        "reasoning": "Broad exploratory analysis. analytics_insights: AG04 (fetch 2yr sales data) -> AG05 (sample for initial exploration) -> AG36 (full analysis: patterns, trends, correlations, anomalies with multi-modal output) -> AG37 (build RAG for follow-up Q&A on findings). AG37 is valuable here — user will likely want to ask follow-up questions about specific patterns found.",
        "plan_overrides": {
            "modify_steps": [{"agent_id": "AG37", "changes": {"is_optional": False}}],
        },
    },
    # ── 14. RECOMMENDATION — NBA for insurance ──────────────────────────────
    {
        "example_name": "insurance_nba_recommendations",
        "intent_context": {
            "matched_node": {"name": "recommendation", "label": "Recommendation", "level": "specific"},
            "confidence": 0.88,
            "abstract_parents": [{"name": "predictive_analytics", "label": "Predictive Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Insurance",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "For our top 10K policyholders, recommend the next best product (cross-sell/upsell) with reason codes and confidence.",
        },
        "selected_plan": "recommendation_engine",
        "reasoning": "NBA with reason codes. recommendation_engine: AG04 (fetch policyholder + product data) -> AG07 (profile to understand feature space: policy types, claims history, demographics) -> AG36 (analyze: identify cross-sell signals, product affinities, risk segments) -> AG29 (generate recommendations per policyholder with reason codes, confidence scores, using collaborative filtering + rule-based logic). AG29 needs AG36's analysis signals to make informed recommendations.",
        "plan_overrides": None,
    },
    # ── 15. FULL LIFECYCLE — new vendor onboarding ──────────────────────────
    {
        "example_name": "vendor_data_onboarding_e2e",
        "intent_context": {
            "matched_node": {"name": "full_data_quality_lifecycle", "label": "Full Data Quality Lifecycle", "level": "specific"},
            "confidence": 0.86,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Retail & E-commerce",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "New vendor just sent us 500K product records. Need full treatment: ingest, profile, find quality issues, fix them, score, and tell us if it's ready for the catalog.",
        },
        "selected_plan": "full_data_quality_lifecycle",
        "reasoning": "End-to-end lifecycle for new data. full_data_quality_lifecycle: AG04 (ingest 500K records) -> AG05 (sample for initial analysis) -> AG06 (LLM-route columns) -> AG07 (full profile) -> AG12 || AG13 || AG14 in parallel (completeness, correctness, anomaly) -> AG16 (fix: missing values, normalization, dedup) -> AG17 (re-score post-fix) -> AG20 (fit for use: is it catalog-ready?). HITL before AG16 fixes. This is the most comprehensive plan — appropriate for unknown vendor data entering the system for the first time.",
        "plan_overrides": None,
    },
    # ── 16. SEMANTIC SEARCH — knowledge base ────────────────────────────────
    {
        "example_name": "product_knowledge_base",
        "intent_context": {
            "matched_node": {"name": "semantic_search", "label": "Semantic Search", "level": "specific"},
            "confidence": 0.85,
            "abstract_parents": [{"name": "exploratory_analytics", "label": "Exploratory Analytics"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Commercial / Industrial",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Build a searchable knowledge base from our product specs, manuals, and datasheets so sales engineers can find information by natural language.",
        },
        "selected_plan": "semantic_search_setup",
        "reasoning": "RAG + search for product knowledge. semantic_search_setup: AG04 (fetch product docs from files/DB) -> AG37 (build RAG: chunk specs/manuals, embed in vector DB) -> AG38 (enable hybrid search: semantic for natural language + filtered for exact specs). Domain graph is available but not required for search setup — it would help AG37's chunking strategy.",
        "plan_overrides": None,
    },
    # ── 17. AMBIGUOUS — profiling vs quality assessment ─────────────────────
    {
        "example_name": "ambiguous_profile_vs_quality",
        "intent_context": {
            "matched_node": {"name": "dataset_profiling", "label": "Dataset Profiling", "level": "specific"},
            "confidence": 0.74,
            "ambiguity_candidates": [
                {"name": "data_quality_assessment", "label": "Data Quality Assessment", "score": 0.71},
            ],
            "requires_clarification": True,
            "clarification_reason": "Could be basic profiling or full quality audit — user said 'check our data'",
        },
        "domain_context": {
            "domain": "Financial Services / Lending",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Check our data and let us know the state of it.",
        },
        "selected_plan": "data_quality_assessment",
        "reasoning": "Ambiguous between profiling and quality audit. When uncertain, choose the broader plan: data_quality_assessment includes profiling (AG07) AND adds quality dimensions (AG12/AG13/AG14/AG17). User saying 'state of data' implies they want quality verdict, not just column stats. Lower confidence (0.74) acceptable — the broader plan covers both interpretations. If user only wanted basic profiling, the extra quality agents add value without harm.",
        "plan_overrides": None,
    },
    # ── 18. AMBIGUOUS — enrichment vs remediation ───────────────────────────
    {
        "example_name": "ambiguous_enrich_vs_fix",
        "intent_context": {
            "matched_node": {"name": "data_enrichment", "label": "Data Enrichment", "level": "specific"},
            "confidence": 0.76,
            "ambiguity_candidates": [
                {"name": "data_quality_fix", "label": "Data Quality Fix", "score": 0.73},
            ],
            "requires_clarification": True,
            "clarification_reason": "User wants to 'improve' data — could mean enrich from external sources or fix internal quality issues",
        },
        "domain_context": {
            "domain": "Master Data Management",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Improve our customer data — addresses are incomplete and many fields are outdated.",
        },
        "selected_plan": "data_enrichment",
        "reasoning": "Ambiguous between enrichment and remediation, but 'incomplete addresses' and 'outdated' point toward enrichment — the data needs external supplementation, not just internal fixes. data_enrichment: AG04 -> AG07 (profile to see gaps) -> AG25 (external: address completion via API) || AG26 (internal: knowledge graph enrichment for other fields) -> AG17 (re-score). Override: add AG16 (DQ Fixer) after enrichment to also handle any inconsistencies found during profiling, combining both approaches.",
        "plan_overrides": {
            "add_steps": [{"agent_id": "AG16", "step_order": 4, "purpose": "Fix internal inconsistencies discovered during profiling (normalize formats, remove duplicates)", "hitl_gate": "before", "hitl_reason": "Approve fix strategy for internal data modifications"}],
            "modify_steps": [{"agent_id": "AG17", "changes": {"step_order": 5}}],
        },
    },
    # ── 19. COMPOUND — ACORD compliance for insurance ───────────────────────
    {
        "example_name": "insurance_acord_compliance",
        "intent_context": {
            "matched_node": {"name": "schema_conformance", "label": "Schema Conformance", "level": "specific"},
            "confidence": 0.82,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "specific_children": [
                {"name": "data_quality_assessment", "label": "Data Quality Assessment"},
            ],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Insurance",
            "query_has_dataset": True,
            "graph_available": True,
            "industry_standard": "ACORD",
            "user_query": "We need ACORD compliance: assess our schema gaps, harmonize, fix data quality issues, and get a compliance score.",
        },
        "selected_plan": "schema_conformance",
        "reasoning": "Primary intent is schema conformance (ACORD), but user also wants quality fixes and scoring — a compound request. Start with schema_conformance as the base plan: AG04 -> AG07 || AG24 (ACORD ontology) -> AG08 (gap analysis) -> AG09 (harmonize) -> AG10 (enforce). Then add quality agents as post-schema steps. Override: append AG12/AG13 (quality assessment), AG16 (fix), AG17 (score) after schema enforcement. This sequences schema-first, quality-second, which is correct — fix the structure before fixing the data.",
        "plan_overrides": {
            "add_steps": [
                {"agent_id": "AG12", "step_order": 6, "purpose": "Assess completeness after schema harmonization"},
                {"agent_id": "AG13", "step_order": 6, "purpose": "Assess correctness against ACORD rules"},
                {"agent_id": "AG16", "step_order": 7, "purpose": "Fix quality issues found", "hitl_gate": "before", "hitl_reason": "Approve data fixes before applying"},
                {"agent_id": "AG17", "step_order": 8, "purpose": "Final compliance score including schema + quality"},
            ],
        },
    },
    # ── 20. COMPOUND — full MDM + publish golden data ───────────────────────
    {
        "example_name": "mdm_full_to_syndication",
        "intent_context": {
            "matched_node": {"name": "entity_resolution", "label": "Entity Resolution", "level": "specific"},
            "confidence": 0.84,
            "abstract_parents": [{"name": "data_management", "label": "Data Management"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Master Data Management",
            "query_has_dataset": True,
            "graph_available": True,
            "user_query": "Deduplicate our product records from 4 vendor feeds, create golden product master, enrich with missing specs, scan for PII, then publish to the analytics warehouse and the e-commerce platform.",
        },
        "selected_plan": "mdm_entity_resolution",
        "reasoning": "Complex compound: MDM + enrichment + security + syndication. Start with mdm_entity_resolution as base (the core ask is dedup + golden record). Override: after AG28 (golden records), add AG25 (enrich missing specs via external API), AG32 (PII scan before publishing), AG34 (catalog), AG35 (syndicate to 2 targets). This chains 4 plans into one execution DAG. AG03 (ESA) consolidates HITL: one review for match clusters + golden record approval + syndication approval instead of 3 separate interruptions.",
        "plan_overrides": {
            "add_steps": [
                {"agent_id": "AG25", "step_order": 7, "purpose": "Enrich golden records with missing product specs via external API"},
                {"agent_id": "AG32", "step_order": 8, "purpose": "Scan enriched golden records for PII before publishing"},
                {"agent_id": "AG34", "step_order": 9, "purpose": "Update catalog with golden product master documentation"},
                {"agent_id": "AG35", "step_order": 10, "purpose": "Syndicate to analytics warehouse + e-commerce platform", "hitl_gate": "before", "hitl_reason": "Approve syndication targets, entitlements, and data scope per consumer"},
                {"agent_id": "AG39", "step_order": 10, "purpose": "Track total pipeline cost: MDM + enrichment + syndication", "is_optional": True},
            ],
        },
    },
    # ── 21. RESEARCH — optimize churn model across surfaces ───────────────
    {
        "example_name": "optimize_churn_model_research",
        "intent_context": {
            "matched_node": {"name": "ml_research", "label": "ML Research", "level": "specific"},
            "confidence": 0.90,
            "abstract_parents": [{"name": "predictive_modeling", "label": "Predictive Modeling"}],
            "requires_clarification": False,
        },
        "domain_context": {
            "domain": "Healthcare",
            "query_has_dataset": True,
            "graph_available": False,
            "user_query": "Optimize our churn prediction model systematically — try different features, models, ensembles, and hyperparameters. I want a structured research project, not just a quick model run.",
        },
        "selected_plan": "research_experiment_pipeline",
        "reasoning": "User explicitly requests structured research across multiple surfaces (features, models, ensembles, hyperparameters). research_experiment_pipeline: AG04 (fetch churn data) -> AG07 (full profile to understand features and target distribution) -> AG40 (4-phase research: Understand → Design 8-15 experiments across 7 surfaces → Prioritize by impact/cost → Execute with stopping criteria). AG40 is self-stopping and produces research_project.md. No HITL needed — AG40 handles iteration internally.",
        "plan_overrides": None,
    },
]


# ---------------------------------------------------------------------------
# Main seed function
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[seed_execution_graph] {msg}")


def seed() -> None:
    log("Connecting to domain DB...")
    conn = get_execution_conn()

    log("Initializing execution graph schema...")
    init_execution_db(conn)

    # ── Agent cards ────────────────────────────────────────────────────────
    log(f"Upserting {len(AGENT_CARDS)} agent cards...")
    card_ids = {}
    for card in AGENT_CARDS:
        cid = upsert_agent_card(conn, **card)
        card_ids[card["agent_id"]] = cid
        log(f"  {card['agent_id']}: {card['name']} (id={cid})")
    conn.commit()

    # ── Plans ──────────────────────────────────────────────────────────────
    log(f"\nUpserting {len(PLANS)} execution plans...")
    plan_ids = {}
    for plan in PLANS:
        pid = upsert_execution_plan(conn, **plan)
        plan_ids[plan["name"]] = pid
        log(f"  {plan['name']}: {plan['label']} (id={pid})")
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
    log("\nGenerating BGE-M3 embeddings for plans and agent cards...")
    embedded = backfill_execution_embeddings(conn)
    log(f"  {embedded} items embedded.")

    conn.close()

    log("\n" + "=" * 60)
    log("Execution graph seeded successfully.")
    log(f"  Agent cards   : {len(card_ids)}")
    log(f"  Plans         : {len(plan_ids)}")
    log(f"  Steps         : {step_count}")
    log(f"  Dependencies  : {dep_count}")
    log(f"  Examples      : {ex_count}")
    log(f"  Embedded      : {embedded}")
    log("=" * 60)


if __name__ == "__main__":
    seed()
