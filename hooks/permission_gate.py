"""
permission_gate.py — Pre-execution permission validation for DESO agents.

Checks the capability matrix to determine whether an agent requires HITL
approval, handles PII-flagged datasets, and validates credential availability
for external API agents.

Public API
----------
validate_pre_execution(agent_id, dcp, user_context) -> dict
"""

from __future__ import annotations

from typing import Optional

# HITL-required agents (from capability_matrix.md)
_HITL_AGENTS: set[str] = {
    "AG03", "AG10", "AG11", "AG16", "AG22", "AG28", "AG33", "AG35",
}

# Agents with irreversible side effects
_IRREVERSIBLE_AGENTS: set[str] = {
    "AG10", "AG16", "AG28", "AG33", "AG35",
}

# Agents that access external APIs
_EXTERNAL_API_AGENTS: set[str] = {
    "AG25",  # DataEnricher.External
}

# Common PII column patterns
_PII_PATTERNS: list[str] = [
    "ssn", "social_security", "patient_id", "mrn", "medical_record",
    "date_of_birth", "dob", "email", "phone", "address", "zip_code",
    "credit_card", "bank_account", "passport", "driver_license",
    "national_id", "tax_id", "npi",
]


def validate_pre_execution(
    agent_id: str,
    dcp: dict,
    user_context: Optional[dict] = None,
) -> dict:
    """Validate whether an agent can proceed without additional approval.

    Parameters
    ----------
    agent_id : str
        Agent identifier (e.g. 'AG16').
    dcp : dict
        Domain Context Pack from SA1.
    user_context : dict | None
        Optional context with keys: 'credentials' (dict of available API keys),
        'hitl_pre_approved' (list of pre-approved agent IDs).

    Returns
    -------
    dict with keys:
        approved       : bool
        reason         : str
        requires_hitl  : bool
        pii_detected   : bool
        warnings       : list[str]
    """
    user_context = user_context or {}
    pre_approved = set(user_context.get("hitl_pre_approved", []))
    credentials = user_context.get("credentials", {})

    warnings: list[str] = []
    requires_hitl = False
    pii_detected = False

    # Check: agent requires HITL?
    if agent_id in _HITL_AGENTS and agent_id not in pre_approved:
        requires_hitl = True

    # Check: agent has irreversible side effects?
    if agent_id in _IRREVERSIBLE_AGENTS:
        warnings.append(
            f"{agent_id} has irreversible side effects — changes cannot be undone."
        )

    # Check: dataset has PII columns?
    columns = dcp.get("columns", [])
    if isinstance(columns, list):
        col_lower = [str(c).lower() for c in columns]
        for pattern in _PII_PATTERNS:
            matching = [c for c in col_lower if pattern in c]
            if matching:
                pii_detected = True
                warnings.append(
                    f"PII indicator '{pattern}' found in columns: {matching}"
                )

    # Check: external API agent needs credentials?
    if agent_id in _EXTERNAL_API_AGENTS:
        if not credentials.get("external_api_key"):
            return {
                "approved": False,
                "reason": f"{agent_id} requires external API credentials not found in context.",
                "requires_hitl": requires_hitl,
                "pii_detected": pii_detected,
                "warnings": warnings,
            }

    # If HITL required and not pre-approved, block
    if requires_hitl:
        return {
            "approved": False,
            "reason": f"{agent_id} requires HITL approval before execution.",
            "requires_hitl": True,
            "pii_detected": pii_detected,
            "warnings": warnings,
        }

    return {
        "approved": True,
        "reason": "All checks passed.",
        "requires_hitl": False,
        "pii_detected": pii_detected,
        "warnings": warnings,
    }
