"""
seed_intents.py — Seed DataAXL-specific intent nodes.

Product data prep, product syndication, and customer data prep intents.
"""

from __future__ import annotations

import os
import sys

_HOOKS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HOOKS_DIR not in sys.path:
    sys.path.insert(0, _HOOKS_DIR)


# DataAXL intent definitions
INTENTS = [
    # Abstract level
    {
        "name": "product_data_management",
        "label": "Product Data Management",
        "description": "Manage product master data lifecycle — ingest, harmonize, enrich, syndicate",
        "level": "abstract",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "customer_data_management",
        "label": "Customer Data Management",
        "description": "Manage customer master data lifecycle — ingest, match, merge, golden record",
        "level": "abstract",
        "domain": "Master Data Management",
        "lifecycle_stage": "preparation",
    },
    # Mid level — Product
    {
        "name": "product_data_prep",
        "label": "Product Data Prep",
        "description": "Ingest product data from Syndigo/Stibo, harmonize schemas, enrich attributes",
        "level": "mid",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "product_data_syndication",
        "label": "Product Data Syndication",
        "description": "Distribute product data to marketplace and retail endpoints",
        "level": "mid",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "publishing",
    },
    # Mid level — Customer
    {
        "name": "customer_data_prep",
        "label": "Customer Data Prep",
        "description": "Ingest customer data from SAP/JDE, search and match, build golden records",
        "level": "mid",
        "domain": "Master Data Management",
        "lifecycle_stage": "preparation",
    },
    # Specific level — Product
    {
        "name": "syndigo_ingest",
        "label": "Syndigo Ingest",
        "description": "Ingest product attributes and digital assets from Syndigo PIM",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "stibo_ingest",
        "label": "Stibo Ingest",
        "description": "Ingest product data from Stibo STEP MDM platform",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "product_harmonize",
        "label": "Product Schema Harmonize",
        "description": "Harmonize product schemas across multiple source systems into unified structure",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "product_enrich",
        "label": "Product Attribute Enrichment",
        "description": "Enrich product records with external data, imagery, taxonomy codes",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "enrichment",
    },
    {
        "name": "amazon_syndication",
        "label": "Amazon Product Feed",
        "description": "Format and push product data to Amazon Seller Central / Vendor Central",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "publishing",
    },
    {
        "name": "gdsn_syndication",
        "label": "GDSN Product Syndication",
        "description": "Syndicate product data via GDSN (Global Data Synchronisation Network)",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "publishing",
    },
    {
        "name": "walmart_syndication",
        "label": "Walmart Product Feed",
        "description": "Format and push product data to Walmart marketplace",
        "level": "specific",
        "domain": "Commercial / Industrial",
        "lifecycle_stage": "publishing",
    },
    # Specific level — Customer
    {
        "name": "sap_customer_ingest",
        "label": "SAP Customer Ingest",
        "description": "Ingest customer master data from SAP ERP",
        "level": "specific",
        "domain": "Master Data Management",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "jde_customer_ingest",
        "label": "JDE Customer Ingest",
        "description": "Ingest customer data from JD Edwards ERP",
        "level": "specific",
        "domain": "Master Data Management",
        "lifecycle_stage": "preparation",
    },
    {
        "name": "customer_search_match",
        "label": "Customer Search & Match",
        "description": "Fuzzy match and deduplicate customer records across systems",
        "level": "specific",
        "domain": "Master Data Management",
        "lifecycle_stage": "enrichment",
    },
    {
        "name": "customer_golden_record",
        "label": "Customer Golden Record",
        "description": "Build and manage the golden customer record with survivorship rules",
        "level": "specific",
        "domain": "Master Data Management",
        "lifecycle_stage": "enrichment",
    },
    {
        "name": "address_enrichment",
        "label": "Address Enrichment",
        "description": "Validate and enrich customer addresses using external address databases",
        "level": "specific",
        "domain": "Master Data Management",
        "lifecycle_stage": "enrichment",
    },
]

TRIPLETS = [
    # IS_A hierarchy
    ("product_data_prep", "IS_A", "product_data_management"),
    ("product_data_syndication", "IS_A", "product_data_management"),
    ("customer_data_prep", "IS_A", "customer_data_management"),
    ("syndigo_ingest", "IS_A", "product_data_prep"),
    ("stibo_ingest", "IS_A", "product_data_prep"),
    ("product_harmonize", "IS_A", "product_data_prep"),
    ("product_enrich", "IS_A", "product_data_prep"),
    ("amazon_syndication", "IS_A", "product_data_syndication"),
    ("gdsn_syndication", "IS_A", "product_data_syndication"),
    ("walmart_syndication", "IS_A", "product_data_syndication"),
    ("sap_customer_ingest", "IS_A", "customer_data_prep"),
    ("jde_customer_ingest", "IS_A", "customer_data_prep"),
    ("customer_search_match", "IS_A", "customer_data_prep"),
    ("customer_golden_record", "IS_A", "customer_data_prep"),
    ("address_enrichment", "IS_A", "customer_data_prep"),
    # Dependencies
    ("product_harmonize", "REQUIRES", "syndigo_ingest"),
    ("product_harmonize", "REQUIRES", "stibo_ingest"),
    ("product_enrich", "REQUIRES", "product_harmonize"),
    ("amazon_syndication", "REQUIRES", "product_enrich"),
    ("gdsn_syndication", "REQUIRES", "product_enrich"),
    ("walmart_syndication", "REQUIRES", "product_enrich"),
    ("customer_search_match", "REQUIRES", "sap_customer_ingest"),
    ("customer_golden_record", "REQUIRES", "customer_search_match"),
    ("address_enrichment", "REQUIRES", "customer_golden_record"),
    # Enables
    ("product_enrich", "ENABLES", "product_data_syndication"),
    ("customer_golden_record", "ENABLES", "address_enrichment"),
    # Precedes
    ("syndigo_ingest", "PRECEDES", "product_harmonize"),
    ("sap_customer_ingest", "PRECEDES", "customer_search_match"),
]

ALIASES = [
    ("syndigo_ingest", "syndigo"),
    ("syndigo_ingest", "pim ingest"),
    ("stibo_ingest", "stibo"),
    ("stibo_ingest", "step mdm"),
    ("product_harmonize", "harmonize products"),
    ("product_harmonize", "schema harmonization"),
    ("product_enrich", "enrich products"),
    ("product_enrich", "product enrichment"),
    ("amazon_syndication", "amazon feed"),
    ("amazon_syndication", "seller central"),
    ("gdsn_syndication", "gdsn"),
    ("gdsn_syndication", "global data sync"),
    ("walmart_syndication", "walmart feed"),
    ("walmart_syndication", "walmart marketplace"),
    ("sap_customer_ingest", "sap customer"),
    ("jde_customer_ingest", "jde customer"),
    ("jde_customer_ingest", "jd edwards"),
    ("customer_search_match", "customer dedup"),
    ("customer_search_match", "fuzzy match"),
    ("customer_golden_record", "golden record"),
    ("customer_golden_record", "customer 360"),
    ("address_enrichment", "address validation"),
    ("address_enrichment", "address cleansing"),
    ("product_data_prep", "product prep"),
    ("product_data_syndication", "product syndication"),
    ("customer_data_prep", "customer prep"),
]


def seed_intents(conn, app_id: str = "dataaxl") -> int:
    """Seed DataAXL intent nodes, triplets, and aliases. Returns node count."""
    from intent_graph import (
        init_intent_db, upsert_intent_node, add_intent_triplet, add_intent_alias,
    )

    init_intent_db(conn)

    count = 0
    for intent in INTENTS:
        upsert_intent_node(
            conn,
            name=intent["name"],
            label=intent["label"],
            description=intent["description"],
            level=intent["level"],
            domain=intent["domain"],
            lifecycle_stage=intent.get("lifecycle_stage"),
            app_id=app_id,
        )
        count += 1
    conn.commit()

    for subj, pred, obj in TRIPLETS:
        try:
            add_intent_triplet(conn, subj, pred, obj)
        except Exception:
            pass
    conn.commit()

    for intent_name, alias in ALIASES:
        try:
            add_intent_alias(conn, intent_name, alias)
        except Exception:
            pass
    conn.commit()

    return count


if __name__ == "__main__":
    from intent_graph import get_intent_conn
    conn = get_intent_conn()
    n = seed_intents(conn)
    conn.close()
    print(f"Seeded {n} DataAXL intent nodes.")
