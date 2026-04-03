"""
self_improver.py -- Post-run self-improvement hook for the data_analysis skill.

After every analysis run, call run_self_improvement() to:
  1. Append run metadata to run_history.jsonl
  2. Ask Claude (or use keyword heuristics) to suggest improvements
  3. Auto-apply safe improvements:
       - New/extra keywords -> domain_overrides.json  (extra_keywords)
       - New domain profiles -> domain_overrides.json  (extra_profiles)
       - Lessons learned    -> SKILL.md  (## Learned Patterns section)
  4. Queue complex improvements to pending_improvements.md for human review

Public API
----------
log_run(run_meta)                                            -> str (log path)
review_run(run_meta, anthropic_api_key)                      -> dict (suggestions)
apply_improvements(suggestions)                              -> dict (applied stats)
run_self_improvement(run_meta, anthropic_api_key, auto_apply)-> dict (full report)
improvement_report(result)                                   -> str (plain-English)

run_meta expected keys
----------------------
dataset_path      : str
query             : str
domain_detected   : str
domain_method     : str   ('claude' | 'heuristic')
domain_confidence : str   ('high' | 'medium' | 'low')
domain_correct    : bool | None   (None = unknown)
domain_actual     : str | None    (supply when domain_correct is False)
analysis_type     : str   ('profile'|'anomaly'|'comparison'|'aggregation'|'sampling')
workflows_run     : list[str]     e.g. ['A'] or ['A','C']
row_count         : int
col_count         : int
column_names      : list[str]
issues            : list[str]     free-text issue strings from the run
findings_summary  : str
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────

_HOOKS_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(_HOOKS_DIR)                         # research/data-analysis/
OVERRIDES_PATH    = os.path.join(BASE_DIR, "domain_overrides.json")
RUN_HISTORY_PATH  = os.path.join(BASE_DIR, "run_history.jsonl")
PENDING_PATH      = os.path.join(BASE_DIR, "pending_improvements.md")
SKILL_MD_PATH     = r"C:\Users\Ravi\.claude\skills\data_analysis\SKILL.md"

CLAUDE_MODEL = "claude-sonnet-4-6"


# ── 1. Run logging ─────────────────────────────────────────────────────────────

def log_run(run_meta: dict) -> str:
    """Append run metadata to run_history.jsonl. Returns the log file path."""
    entry = {
        **run_meta,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(RUN_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return RUN_HISTORY_PATH


# ── 2. Review ─────────────────────────────────────────────────────────────────

_REVIEW_PROMPT = """\
You are an AI skill improvement engine reviewing a completed data analysis run.

Run metadata:
{meta_json}

Based on the metadata, produce improvement suggestions as a JSON object with exactly
these keys (no extra keys, no commentary outside the JSON):

{{
  "domain_assessment": {{
    "was_correct": true | false | null,
    "suggested_domain": "<domain name, or null>",
    "new_keywords": ["keyword1", "keyword2"],
    "new_profile": {{
      "role": "<analyst role description>",
      "analysis_lens": "<what to focus on>",
      "anomaly_guidance": "<domain-specific anomaly patterns>",
      "key_metrics": ["metric1", "metric2"]
    }} or null
  }},
  "skill_notes": ["short actionable lesson 1", "short actionable lesson 2"],
  "pending_improvements": ["complex change description 1"]
}}

Rules:
- new_keywords: lowercase, relevant to the suggested_domain; include column-name signals seen in this run.
- new_profile: only set if suggested_domain is a genuinely new domain not already in:
    Healthcare, Pharmaceutical, Industrial Equipment, Electrical Components,
    Financial Services / Lending, Insurance, Travel & Hospitality,
    Retail & E-commerce, HR & People Analytics, Education, General
- skill_notes: 1-2 short bullet points a future analyst should remember about this dataset type.
- pending_improvements: things that require changing Python logic or multi-step workflows.
Return only valid JSON.
"""


def review_run(
    run_meta: dict,
    anthropic_api_key: Optional[str] = None,
) -> dict:
    """Ask Claude (or heuristics) to analyse the run and suggest improvements."""
    from hooks.llm_client import call_llm
    text = call_llm(
        _REVIEW_PROMPT.format(meta_json=json.dumps(run_meta, indent=2, default=str)),
        max_tokens=900,
        expect_json=True,
    )
    if text:
        try:
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            suggestions = json.loads(text)
            suggestions["_review_method"] = "llm"
            return suggestions
        except Exception as exc:
            suggestions = _heuristic_review(run_meta)
            suggestions["_review_error"] = str(exc)
            return suggestions

    return _heuristic_review(run_meta)


def _heuristic_review(run_meta: dict) -> dict:
    """Keyword-based review fallback when Claude API is unavailable."""
    suggestions: dict = {
        "domain_assessment": {
            "was_correct": run_meta.get("domain_correct"),
            "suggested_domain": run_meta.get("domain_actual"),
            "new_keywords": [],
            "new_profile": None,
        },
        "skill_notes": [],
        "pending_improvements": [],
        "_review_method": "heuristic",
    }

    col_text = " ".join(run_meta.get("column_names", [])).lower()
    issues   = run_meta.get("issues", [])

    # -- MDM signal detection --
    mdm_signals = [
        "cleansed", "mailing name", "golden record", "mdm id", "grm",
        "survivorship", "dedup", "master data", "baxter mdm",
        "intentional duplicate", "cleansed city", "cleansed province",
        "cleansed postal", "grm process state", "grm state",
    ]
    found_mdm = [s for s in mdm_signals if s in col_text]
    if found_mdm:
        da = suggestions["domain_assessment"]
        if not da["suggested_domain"]:
            da["suggested_domain"] = "Master Data Management"
        da["new_keywords"] = list(set(da.get("new_keywords", []) + found_mdm))
        if "Master Data Management" not in (
            "Healthcare Pharmaceutical Industrial Equipment Electrical Components "
            "Financial Services / Lending Insurance Travel & Hospitality "
            "Retail & E-commerce HR & People Analytics Education General"
        ):
            da["new_profile"] = {
                "role": (
                    "master data management analyst with expertise in entity resolution, "
                    "address standardisation, and golden record creation"
                ),
                "analysis_lens": (
                    "Focus on source vs cleansed field divergence, survivorship rule outcomes, "
                    "duplicate candidate pairs, and MDM pipeline completeness."
                ),
                "anomaly_guidance": (
                    "Flag null cleansed fields (pipeline not yet run), name mismatches between "
                    "source and mailing name, inconsistent GRM state transitions, asymmetric "
                    "block sizes in source-vs-cleansed comparisons, and duplicate records that "
                    "were not merged."
                ),
                "key_metrics": [
                    "golden record completeness %",
                    "duplicate match rate",
                    "cleansing success rate",
                    "null cleansed field %",
                    "survivorship rule coverage",
                ],
            }

    # -- Issue-specific notes --
    for issue in issues:
        il = issue.lower()
        if "asymmetric" in il or "asymmetric_block" in il:
            suggestions["skill_notes"].append(
                "Asymmetric block comparison: when the two ranges have different column counts, "
                "compare semantic field pairs (e.g. Legal Name vs Mailing Name) rather than "
                "whole-block identity."
            )
        if "null_cleansed" in il or "cleansed" in il and "null" in il:
            suggestions["skill_notes"].append(
                "Null cleansed fields (Cleansed City, Cleansed Province, etc.) indicate the "
                "address standardisation pipeline has not yet written back results -- treat as "
                "pipeline-incomplete, not a data quality error in the source."
            )
        if "domain_mismatch" in il:
            suggestions["pending_improvements"].append(
                f"Domain mismatch on '{run_meta.get('dataset_path','?')}': {issue}"
            )

    return suggestions


# ── 3. Apply improvements ─────────────────────────────────────────────────────

def _load_overrides() -> dict:
    if os.path.exists(OVERRIDES_PATH):
        try:
            with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"extra_keywords": {}, "extra_profiles": {}}


def _save_overrides(overrides: dict) -> None:
    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)


def apply_improvements(suggestions: dict) -> dict:
    """
    Auto-apply safe improvements; queue complex ones.

    Returns dict with:
      applied_keywords  : {domain: [new_kws]}
      applied_profiles  : [domain_names]
      skill_notes_added : int
      queued            : int
    """
    result = {
        "applied_keywords":  {},
        "applied_profiles":  [],
        "skill_notes_added": 0,
        "queued":            0,
    }

    overrides = _load_overrides()
    da        = suggestions.get("domain_assessment", {})
    domain    = da.get("suggested_domain")
    new_kws   = da.get("new_keywords") or []
    new_prof  = da.get("new_profile")

    # 1. Merge extra keywords into overrides
    if domain and new_kws:
        bucket   = overrides.setdefault("extra_keywords", {})
        existing = set(bucket.get(domain, []))
        added    = [kw for kw in new_kws if kw not in existing]
        if added:
            bucket.setdefault(domain, []).extend(added)
            result["applied_keywords"][domain] = added

    # 2. Add new domain profile (only if domain not already known)
    if domain and new_prof:
        ep = overrides.setdefault("extra_profiles", {})
        if domain not in ep:
            ep[domain] = new_prof
            result["applied_profiles"].append(domain)

    if result["applied_keywords"] or result["applied_profiles"]:
        _save_overrides(overrides)

    # 3. Append skill notes to SKILL.md (## Learned Patterns section)
    notes = suggestions.get("skill_notes", [])
    if notes and os.path.exists(SKILL_MD_PATH):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(SKILL_MD_PATH, "r", encoding="utf-8") as fr:
            existing_md = fr.read()

        with open(SKILL_MD_PATH, "a", encoding="utf-8") as fa:
            if "## Learned Patterns" not in existing_md:
                fa.write("\n\n---\n\n## Learned Patterns\n\n")
                fa.write("Auto-updated by `hooks/self_improver.py` after each run.\n")
            fa.write(f"\n### {today}\n")
            for note in notes:
                fa.write(f"- {note}\n")

        result["skill_notes_added"] = len(notes)

    # 4. Queue complex improvements
    pending = suggestions.get("pending_improvements", [])
    if pending:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with open(PENDING_PATH, "a", encoding="utf-8") as fp:
            fp.write(f"\n## {ts}\n\n")
            for item in pending:
                fp.write(f"- [ ] {item}\n")
        result["queued"] = len(pending)

    return result


# ── 4. Main entry point ───────────────────────────────────────────────────────

def run_self_improvement(
    run_meta: dict,
    anthropic_api_key: Optional[str] = None,
    auto_apply: bool = True,
) -> dict:
    """
    Full self-improvement cycle:
      log -> review -> apply -> return report dict.

    Pass the result to improvement_report() for a printable summary.
    """
    log_run(run_meta)
    suggestions = review_run(run_meta, anthropic_api_key=anthropic_api_key)
    applied     = apply_improvements(suggestions) if auto_apply else {}
    return {"suggestions": suggestions, "applied": applied}


def improvement_report(result: dict) -> str:
    """Return a plain-English summary of what was improved this run."""
    s = result.get("suggestions", {})
    a = result.get("applied", {})
    da = s.get("domain_assessment", {})

    lines = ["=== SELF-IMPROVEMENT REPORT ==="]

    # Domain review
    if da.get("suggested_domain"):
        correct = da.get("was_correct")
        status  = "correct" if correct is True else ("incorrect" if correct is False else "unknown")
        lines.append(f"Domain detection was {status}. Suggested domain: {da['suggested_domain']}")
        method = s.get("_review_method", "?")
        lines.append(f"Review method: {method}")

    # Applied keyword additions
    for domain, kws in a.get("applied_keywords", {}).items():
        preview = ", ".join(kws[:6]) + ("..." if len(kws) > 6 else "")
        lines.append(f"  + Added {len(kws)} keyword(s) for '{domain}': {preview}")

    # Applied new profiles
    for domain in a.get("applied_profiles", []):
        lines.append(f"  + Added new domain profile: '{domain}'")

    if not a.get("applied_keywords") and not a.get("applied_profiles"):
        lines.append("  No domain overrides needed.")

    # Skill notes
    n = a.get("skill_notes_added", 0)
    if n:
        lines.append(f"  + {n} lesson(s) appended to SKILL.md -> Learned Patterns")

    # Pending queue
    q = a.get("queued", 0)
    if q:
        lines.append(f"  ~ {q} complex improvement(s) queued in pending_improvements.md")

    # Print the notes themselves
    notes = s.get("skill_notes", [])
    if notes:
        lines.append("\nLessons learned this run:")
        for note in notes:
            lines.append(f"  - {note}")

    err = s.get("_review_error")
    if err:
        lines.append(f"\n[Warning] Claude review failed, used heuristic fallback: {err}")

    lines.append("================================")
    return "\n".join(lines)
