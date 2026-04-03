"""
seed_mdm_intents.py — Add missing MDM / data-lifecycle intent nodes to the intent graph.

Fills gaps identified by testing Intent Summaries.docx and Expanded_Intents.docx
against the existing graph.

New nodes (12):
  Specific:  data_extraction, data_validation, data_harmonization,
             schema_recommendation, web_scraping, standard_compliance,
             contrastive_learning, embedding_similarity
  Mid:       extract_and_profile, extract_and_validate,
             extract_validate_harmonize, full_data_quality_lifecycle

Run:
    python hooks/seed_mdm_intents.py
"""

from __future__ import annotations

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


def seed(conn) -> None:
    init_intent_db(conn)

    # -----------------------------------------------------------------------
    # 1. New specific nodes
    # -----------------------------------------------------------------------

    upsert_intent_node(
        conn,
        name="data_extraction",
        label="Data Extraction",
        description=(
            "Acquire raw data from specified sources and prepare it as a consumable "
            "data object for subsequent analytical, profiling, or quality workflows. "
            "No transformation or schema inference at this stage."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="ingest",
    )

    upsert_intent_node(
        conn,
        name="data_validation",
        label="Data Validation",
        description=(
            "Assess data correctness, accuracy, and consistency using internal checks "
            "and external industry or domain knowledge. Non-destructive — does not "
            "modify or harmonize data. Produces a validation report."
        ),
        level="mid",
        domain="any",
        lifecycle_stage="assess",
    )

    upsert_intent_node(
        conn,
        name="data_harmonization",
        label="Data Harmonization",
        description=(
            "Transform validated data into standardized formats aligned with industry "
            "and domain standards. Requires validation outputs. Tracks all "
            "transformations for traceability."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="transform",
    )

    upsert_intent_node(
        conn,
        name="schema_recommendation",
        label="Schema Recommendation",
        description=(
            "Recommend an optimal target schema based on data characteristics, "
            "industry standards, and domain ontologies. Decision-support step — "
            "does not enforce schema automatically."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="design",
    )

    upsert_intent_node(
        conn,
        name="web_scraping",
        label="Web Scraping and Data Sourcing",
        description=(
            "Extract structured or semi-structured data from websites, catalogs, "
            "or online sources. Typically precedes enrichment and harmonization."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="ingest",
    )

    upsert_intent_node(
        conn,
        name="standard_compliance",
        label="Industry Standard Compliance",
        description=(
            "Apply or enforce specific industry standards (e.g. NEMA, IEC, ISO, "
            "ANSI, HL7) to data or product attributes. Includes standard selection "
            "and conflict resolution when multiple standards apply."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="transform",
    )

    upsert_intent_node(
        conn,
        name="contrastive_learning",
        label="Contrastive Learning",
        description=(
            "Machine learning technique that learns representations by contrasting "
            "positive pairs against negative pairs. Used for similarity learning, "
            "embedding quality improvement, and few-shot classification."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="model",
    )

    upsert_intent_node(
        conn,
        name="embedding_similarity",
        label="Embedding and Similarity Search",
        description=(
            "Generate vector embeddings for data records and compute similarity "
            "scores. Used for deduplication, record matching, semantic search, "
            "and nearest-neighbour retrieval."
        ),
        level="specific",
        domain="any",
        lifecycle_stage="model",
    )

    conn.commit()
    print("  [+] 8 specific/mid nodes upserted")

    # -----------------------------------------------------------------------
    # 2. Compound / orchestration nodes (mid-level)
    # -----------------------------------------------------------------------

    upsert_intent_node(
        conn,
        name="extract_and_profile",
        label="Extract and Profile Data",
        description=(
            "Compound workflow: first acquire data from sources, then profile it "
            "to understand structure and quality in absence of schema. Extraction "
            "must complete before profiling begins."
        ),
        level="mid",
        domain="any",
        lifecycle_stage="ingest+assess",
    )

    upsert_intent_node(
        conn,
        name="extract_and_validate",
        label="Extract and Validate Data",
        description=(
            "Compound workflow: extract required data, profile it, then validate "
            "accuracy and consistency against industry standards and domain knowledge. "
            "Extraction and profiling precede validation."
        ),
        level="mid",
        domain="any",
        lifecycle_stage="ingest+assess",
    )

    upsert_intent_node(
        conn,
        name="extract_validate_harmonize",
        label="Extract, Validate, and Harmonize Data",
        description=(
            "Compound workflow: extract raw data, validate for accuracy and "
            "consistency, then harmonize to industry-standard formats. "
            "Progressive transformation with full traceability."
        ),
        level="mid",
        domain="any",
        lifecycle_stage="ingest+transform",
    )

    upsert_intent_node(
        conn,
        name="full_data_quality_lifecycle",
        label="End-to-End Data Quality Lifecycle",
        description=(
            "Complete data lifecycle workflow: extract, profile, recommend schema, "
            "validate, improve quality, and harmonize. The most comprehensive "
            "data management orchestration pattern."
        ),
        level="mid",
        domain="any",
        lifecycle_stage="lifecycle",
    )

    conn.commit()
    print("  [+] 4 compound mid-level nodes upserted")

    # -----------------------------------------------------------------------
    # 3. IS_A relationships
    # -----------------------------------------------------------------------

    is_a = [
        # New specifics → existing mid parents
        ("data_extraction",     "IS_A", "data_preparation"),
        ("data_harmonization",  "IS_A", "data_preparation"),
        ("schema_recommendation", "IS_A", "data_preparation"),
        ("web_scraping",        "IS_A", "data_preparation"),
        ("standard_compliance", "IS_A", "data_validation"),
        ("data_validation",     "IS_A", "data_management"),

        # ML technique nodes
        ("contrastive_learning",  "IS_A", "ml_modeling"),
        ("embedding_similarity",  "IS_A", "ml_modeling"),

        # Compound nodes → abstract parent
        ("extract_and_profile",          "IS_A", "data_management"),
        ("extract_and_validate",         "IS_A", "data_management"),
        ("extract_validate_harmonize",   "IS_A", "data_management"),
        ("full_data_quality_lifecycle",  "IS_A", "data_management"),
    ]

    for subj, pred, obj in is_a:
        try:
            add_intent_triplet(conn, subj, pred, obj)
        except Exception as e:
            print(f"  [!] IS_A {subj} → {obj}: {e}")

    conn.commit()
    print("  [+] IS_A triplets added")

    # -----------------------------------------------------------------------
    # 4. REQUIRES / ENABLES / PRECEDES relationships
    # -----------------------------------------------------------------------

    rels = [
        # data_extraction precedes and enables profiling
        ("data_extraction",  "PRECEDES", "data_profiling"),
        ("data_extraction",  "PRECEDES", "dataset_profiling"),
        ("data_extraction",  "ENABLES",  "data_profiling"),

        # validation requires profiling
        ("data_validation",  "REQUIRES", "data_profiling"),
        ("data_validation",  "PRECEDES", "data_harmonization"),
        ("data_validation",  "ENABLES",  "data_harmonization"),

        # harmonization requires validation
        ("data_harmonization", "REQUIRES", "data_validation"),

        # standard compliance requires validation
        ("standard_compliance", "REQUIRES", "data_validation"),

        # schema recommendation requires profiling
        ("schema_recommendation", "REQUIRES", "data_profiling"),
        ("schema_recommendation", "PRECEDES", "data_harmonization"),

        # web scraping enables extraction
        ("web_scraping",   "ENABLES", "data_extraction"),
        ("web_scraping",   "PRECEDES", "data_extraction"),

        # compound workflow internal requirements
        ("extract_and_profile",        "REQUIRES", "data_extraction"),
        ("extract_and_validate",       "REQUIRES", "data_extraction"),
        ("extract_and_validate",       "REQUIRES", "data_validation"),
        ("extract_validate_harmonize", "REQUIRES", "data_extraction"),
        ("extract_validate_harmonize", "REQUIRES", "data_validation"),
        ("extract_validate_harmonize", "REQUIRES", "data_harmonization"),
        ("full_data_quality_lifecycle", "REQUIRES", "data_extraction"),
        ("full_data_quality_lifecycle", "REQUIRES", "data_validation"),
        ("full_data_quality_lifecycle", "REQUIRES", "data_harmonization"),
        ("full_data_quality_lifecycle", "REQUIRES", "schema_recommendation"),

        # embedding similarity relates to entity matching
        ("embedding_similarity", "ENABLES", "entity_matching"),
        ("embedding_similarity", "ENABLES", "entity_deduplication"),

        # contrastive learning enables embedding quality
        ("contrastive_learning", "ENABLES", "embedding_similarity"),

        # schema_recommendation vs schema_mapping
        ("schema_recommendation", "COMPETES_WITH", "schema_mapping"),
    ]

    for subj, pred, obj in rels:
        try:
            add_intent_triplet(conn, subj, pred, obj)
        except Exception as e:
            print(f"  [!] {pred} {subj} → {obj}: {e}")

    conn.commit()
    print("  [+] REQUIRES/ENABLES/PRECEDES/COMPETES_WITH triplets added")

    # -----------------------------------------------------------------------
    # 5. Aliases
    # -----------------------------------------------------------------------

    aliases = {
        "data_extraction": [
            ("extract data", "phrase"),
            ("data ingestion", "synonym"),
            ("data acquisition", "synonym"),
            ("pull data", "phrase"),
            ("load data", "phrase"),
            ("get data", "phrase"),
            ("downstream analysis", "keyword"),
            ("ingest", "keyword"),
        ],
        "data_validation": [
            ("validate data", "phrase"),
            ("data accuracy", "phrase"),
            ("data consistency", "phrase"),
            ("validation report", "phrase"),
            ("check data quality", "phrase"),
            ("accuracy check", "phrase"),
            ("consistency check", "phrase"),
            ("verify data", "phrase"),
        ],
        "data_harmonization": [
            ("harmonize data", "phrase"),
            ("harmonization", "keyword"),
            ("standardize data", "phrase"),
            ("data standardization", "synonym"),
            ("align to standard", "phrase"),
            ("industry standard format", "phrase"),
            ("cross-system interoperability", "phrase"),
        ],
        "schema_recommendation": [
            ("recommend schema", "phrase"),
            ("schema suggestion", "synonym"),
            ("schema design", "phrase"),
            ("target schema", "phrase"),
            ("optimal schema", "phrase"),
            ("schema alignment", "phrase"),
            ("field mapping confidence", "phrase"),
        ],
        "web_scraping": [
            ("scrape website", "phrase"),
            ("web extraction", "synonym"),
            ("create data from websites", "phrase"),
            ("website data", "phrase"),
            ("online catalog", "keyword"),
            ("crawl", "keyword"),
            ("scraping", "keyword"),
        ],
        "standard_compliance": [
            ("nema standard", "keyword"),
            ("iec standard", "keyword"),
            ("iso standard", "keyword"),
            ("ansi standard", "keyword"),
            ("industry standard", "phrase"),
            ("use nema", "phrase"),
            ("use iec", "phrase"),
            ("standard compliance", "phrase"),
            ("apply standard", "phrase"),
        ],
        "contrastive_learning": [
            ("contrast learning", "phrase"),
            ("contrastive loss", "keyword"),
            ("positive negative pairs", "phrase"),
            ("siamese network", "synonym"),
            ("triplet loss", "keyword"),
            ("few-shot learning", "synonym"),
        ],
        "embedding_similarity": [
            ("embeddings and similarity", "phrase"),
            ("vector similarity", "phrase"),
            ("semantic similarity", "synonym"),
            ("cosine similarity", "keyword"),
            ("nearest neighbor", "phrase"),
            ("embedding search", "phrase"),
            ("vector search", "phrase"),
            ("similarity matching", "phrase"),
        ],
        "extract_and_profile": [
            ("extract and profile", "phrase"),
            ("extract then profile", "phrase"),
            ("profile data without schema", "phrase"),
            ("data not available profile", "phrase"),
        ],
        "extract_and_validate": [
            ("extract and validate", "phrase"),
            ("extract then validate", "phrase"),
            ("validate data not available", "phrase"),
            ("data not available validate", "phrase"),
        ],
        "extract_validate_harmonize": [
            ("extract validate harmonize", "phrase"),
            ("extract and harmonize", "phrase"),
            ("data not available harmonize", "phrase"),
            ("end to end harmonization", "phrase"),
        ],
        "full_data_quality_lifecycle": [
            ("end to end data quality", "phrase"),
            ("full data lifecycle", "phrase"),
            ("extraction to harmonization", "phrase"),
            ("data quality lifecycle", "phrase"),
            ("clean usable data", "phrase"),
        ],
    }

    count = 0
    for intent_name, alias_list in aliases.items():
        for alias_text, alias_type in alias_list:
            try:
                add_intent_alias(conn, intent_name, alias_text, alias_type)
                count += 1
            except Exception as e:
                print(f"  [!] alias {intent_name}/{alias_text}: {e}")

    conn.commit()
    print(f"  [+] {count} aliases added")

    # -----------------------------------------------------------------------
    # 6. Embed all new nodes
    # -----------------------------------------------------------------------

    n = embed_intent_nodes(conn)
    conn.commit()
    print(f"  [+] {n} nodes embedded")


if __name__ == "__main__":
    print("Seeding MDM / data-lifecycle intent nodes...")
    conn = get_intent_conn()
    try:
        seed(conn)
        print("Done.")
    finally:
        conn.close()
