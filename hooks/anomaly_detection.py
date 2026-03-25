"""
anomaly_detection.py — Combined numeric + categorical/text anomaly detection hook.

Numeric:     z-score | modified z-score (robust, default) | IQR
             Handles pipe-delimited values (e.g. "250||600") automatically.

Categorical: exact-match → fuzzy → Claude semantic judgment (optional).
             Requires ANTHROPIC_API_KEY and pip install anthropic.
             Falls back to "out-of-vocabulary" flag when Claude is unavailable.

Public API
----------
detect_anomalies(df, ...)    -> (annotated_df, summary_df)
anomaly_summary(summary_df)  -> str   plain-English narrative
"""

from __future__ import annotations

import re
import json
import difflib
import unicodedata
from typing import Optional

import pandas as pd
import numpy as np

# scipy — optional (needed for mod_z only)
try:
    from scipy.stats import median_abs_deviation as _scipy_mad
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── text normalisation helpers ───────────────────────────────────────────────

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}

def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[\-_/\\()\[\]{}:,]+", " ", s)
    for word, digit in _NUMBER_WORDS.items():
        s = re.sub(rf"\b{re.escape(word)}\b", digit, s)
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fuzzy_best(query: str, candidates: list[str]) -> tuple[str | None, float]:
    best, best_ratio = None, 0.0
    for c in candidates:
        r = difflib.SequenceMatcher(None, query, c).ratio()
        if r > best_ratio:
            best_ratio, best = r, c
    return best, best_ratio


# ── numeric parsing & scoring ────────────────────────────────────────────────

def _parse_pipe_numeric(val, pipe_agg: str = "mean") -> float:
    """Parse a value that may be pipe-delimited (e.g. '250||600') into a float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val)
    if "||" in s:
        nums = []
        for p in s.split("||"):
            p = p.strip()
            if p:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
        if not nums:
            return np.nan
        agg_fn = {"mean": np.mean, "min": min, "max": max, "first": lambda x: x[0]}
        return float(agg_fn.get(pipe_agg, np.mean)(nums))
    try:
        return float(s.strip())
    except ValueError:
        return np.nan


def _numeric_scores(
    series: pd.Series,
    method: str,
    z_threshold: float,
    iqr_factor: float,
) -> tuple[pd.Series, pd.Series]:
    """Return (score_series, flag_series) for a pre-parsed numeric Series."""
    score = pd.Series(np.nan, index=series.index, dtype=float)
    flag = pd.Series(False, index=series.index)
    clean = series.dropna()

    if clean.empty or len(clean) < 2:
        return score, flag

    if method == "z":
        mu = clean.mean()
        sigma = clean.std(ddof=0)
        if sigma > 0:
            score = (series - mu).abs() / sigma
            flag = score > z_threshold

    elif method == "mod_z":
        if not _SCIPY:
            raise ImportError(
                "scipy is required for mod_z. Install with: pip install scipy"
            )
        med = clean.median()
        mad = _scipy_mad(clean, scale="normal")
        if mad > 0:
            score = 0.6745 * (series - med).abs() / mad
            flag = score > z_threshold
        else:
            # degenerate: all values identical — fall back to z-score
            mu, sigma = clean.mean(), clean.std(ddof=0)
            if sigma > 0:
                score = (series - mu).abs() / sigma
                flag = score > z_threshold

    elif method == "iqr":
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        def _iqr_score(x: float) -> float:
            if pd.isna(x):
                return np.nan
            if x < lower:
                return (lower - x) / (iqr if iqr > 0 else 1.0)
            if x > upper:
                return (x - upper) / (iqr if iqr > 0 else 1.0)
            return 0.0

        score = series.apply(_iqr_score)
        flag = score > 0

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'z', 'mod_z', or 'iqr'.")

    return score, flag


# ── Claude judgment ──────────────────────────────────────────────────────────

_judgment_cache: dict[str, dict] = {}


def _judge_claude(
    col_name: str,
    domain_prompt: str,
    expected_values: list[str],
    raw_value: str,
    client,
    model: str,
) -> tuple[str, float, str]:
    """Call Claude to judge whether raw_value is normal or anomalous.

    Returns (verdict, score, reason).
    verdict : 'NORMAL' | 'ANOMALY' | 'UNKNOWN'
    score   : float 0-1 (probability the value is valid)
    reason  : short explanation string
    """
    cache_key = f"{col_name}|||{raw_value}"
    if cache_key in _judgment_cache:
        c = _judgment_cache[cache_key]
        return c["verdict"], float(c["score"]), c["reason"]

    prompt = (
        f"You are a data quality expert reviewing a tabular dataset.\n\n"
        f"Column: {col_name}\n"
        f"Column description: {domain_prompt}\n"
        f"Known valid values (sample): {json.dumps(expected_values[:50])}\n\n"
        f'Observed value: "{raw_value}"\n\n'
        f"Decide whether this value is a normal valid entry or an anomaly "
        f"(data entry error, wrong unit, multi-value packing, range expression, "
        f"unsupported format, or genuinely out-of-vocabulary).\n\n"
        f"Reply with exactly three lines and nothing else:\n"
        f"VERDICT: NORMAL or ANOMALY\n"
        f"SCORE: 0.00-1.00 (probability the value is valid/normal)\n"
        f"REASON: one short phrase (max 10 words)"
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        verdict, score, reason = "ANOMALY", 0.5, ""
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().upper()
                verdict = "NORMAL" if "NORMAL" in v else "ANOMALY"
            elif line.upper().startswith("SCORE:"):
                try:
                    score = float(re.search(r"[\d.]+", line.split(":", 1)[1])[0])
                    score = max(0.0, min(1.0, score))
                except (TypeError, ValueError):
                    pass
            elif line.upper().startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
        result = {"verdict": verdict, "score": score, "reason": reason}
        _judgment_cache[cache_key] = result
        return verdict, score, reason
    except Exception as e:
        return "UNKNOWN", 0.0, f"API error: {e}"


# ── column type auto-detection ───────────────────────────────────────────────

def _is_numeric_col(
    series: pd.Series,
    detection_ratio: float = 0.8,
    pipe_agg: str = "mean",
) -> bool:
    """Return True if >= detection_ratio of non-null values parse as numbers."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = non_null.apply(lambda x: _parse_pipe_numeric(x, pipe_agg))
    return (parsed.notna().sum() / len(non_null)) >= detection_ratio


# ── column insertion helper ───────────────────────────────────────────────────

def _insert_adjacent(df: pd.DataFrame, base_col: str, new_cols: dict) -> pd.DataFrame:
    """Insert new_cols immediately after base_col in column order."""
    cols = list(df.columns)
    if base_col not in cols:
        for k, v in new_cols.items():
            df[k] = v
        return df
    idx = cols.index(base_col)
    left = cols[: idx + 1]
    right = cols[idx + 1:]
    out = df[left].copy()
    for k, v in new_cols.items():
        out[k] = v
    for c in right:
        out[c] = df[c].values
    return out


# ── public API ───────────────────────────────────────────────────────────────

def detect_anomalies(
    df: pd.DataFrame,
    columns_config: Optional[dict] = None,
    numeric_cols: Optional[list[str]] = None,
    cat_cols: Optional[list[str]] = None,
    method: str = "mod_z",
    z_threshold: float = 3.0,
    iqr_factor: float = 1.5,
    pipe_agg: str = "mean",
    fuzzy_threshold: float = 0.88,
    anthropic_api_key: Optional[str] = None,
    claude_model: str = "claude-sonnet-4-6",
    numeric_detection_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect anomalies in a DataFrame — numeric and categorical columns.

    Parameters
    ----------
    df                      : input DataFrame (string dtype is fine — auto-parsed)
    columns_config          : dict  col -> {"expected_values": [...], "domain_prompt": "...",
                              "type": "numeric"|"categorical" (optional override)}
    numeric_cols            : explicit numeric columns (overrides auto-detect)
    cat_cols                : explicit categorical columns (must pair with columns_config for
                              guided detection; otherwise flagged as out-of-vocabulary only)
    method                  : numeric scoring  'z' | 'mod_z' (default) | 'iqr'
    z_threshold             : flag threshold for z / mod_z  (default 3.0)
    iqr_factor              : IQR fence multiplier  (default 1.5)
    pipe_agg                : how to resolve pipe-delimited numerics  'mean'|'min'|'max'|'first'
    fuzzy_threshold         : minimum fuzzy-match ratio to accept as Normal  (default 0.88)
    anthropic_api_key       : Anthropic key — if None, Claude step is skipped
    claude_model            : Claude model for semantic judgment
    numeric_detection_ratio : fraction parseable-as-number to auto-classify as numeric

    Returns
    -------
    annotated_df : original DataFrame + per-column triples:
                     {col}_score, {col}_flag, {col}_detected_anomaly
    summary_df   : one row per analysed column with aggregate statistics
    """
    columns_config = columns_config or {}
    annotated = df.copy()
    summary_rows: list[dict] = []

    # ── Build sets of columns to process ────────────────────────────────────
    guided_num = {
        col for col, meta in columns_config.items()
        if meta.get("type") == "numeric"
    }
    guided_cat = {
        col for col, meta in columns_config.items()
        if meta.get("type") != "numeric"
    }

    if numeric_cols is not None:
        explicit_num = set(numeric_cols)
    else:
        explicit_num = set()

    if cat_cols is not None:
        explicit_cat = set(cat_cols)
    else:
        explicit_cat = set()

    # Auto-detect unassigned columns
    all_assigned = guided_num | guided_cat | explicit_num | explicit_cat
    auto_num: set[str] = set()
    for col in df.columns:
        if col not in all_assigned and _is_numeric_col(df[col], numeric_detection_ratio, pipe_agg):
            auto_num.add(col)

    process_num = (guided_num | explicit_num | auto_num) & set(df.columns)
    process_cat = (guided_cat | explicit_cat) & set(df.columns)
    # If a column appears in both, numeric wins
    process_cat -= process_num

    # ── Set up Claude client ─────────────────────────────────────────────────
    claude_client = None
    if anthropic_api_key:
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
        except ImportError:
            print(
                "Warning: 'anthropic' not installed — Claude judgment skipped. "
                "Run: pip install anthropic"
            )

    # ── Numeric processing ───────────────────────────────────────────────────
    for col in sorted(process_num):
        parsed = df[col].apply(lambda x: _parse_pipe_numeric(x, pipe_agg))
        parsed_num = pd.to_numeric(parsed, errors="coerce")

        score, flag = _numeric_scores(parsed_num, method, z_threshold, iqr_factor)
        detected = df[col].where(flag.fillna(False), "")

        annotated = _insert_adjacent(annotated, col, {
            f"{col}_score":             score,
            f"{col}_flag":              flag,
            f"{col}_detected_anomaly":  detected,
        })

        n_non_null = int(parsed_num.notna().sum())
        n_anomaly = int(flag.sum())
        anom_vals = parsed_num[flag.fillna(False)].dropna()

        summary_rows.append({
            "column":         col,
            "type":           "numeric",
            "method":         method,
            "non_null":       n_non_null,
            "anomaly_count":  n_anomaly,
            "anomaly_pct":    round(n_anomaly / n_non_null * 100, 2) if n_non_null else 0.0,
            "col_mean":       round(float(parsed_num.mean()), 4) if n_non_null else None,
            "col_std":        round(float(parsed_num.std()), 4) if n_non_null > 1 else None,
            "anomaly_min":    round(float(anom_vals.min()), 4) if not anom_vals.empty else None,
            "anomaly_max":    round(float(anom_vals.max()), 4) if not anom_vals.empty else None,
            "claude_calls":   0,
        })

    # ── Categorical processing ────────────────────────────────────────────────
    for col in sorted(process_cat):
        meta = columns_config.get(col, {})
        expected_values: list[str] = meta.get("expected_values", [])
        domain_prompt: str = meta.get(
            "domain_prompt", f"Valid entries for column '{col}'."
        )

        norm_map = {_normalize(v): v for v in expected_values}
        norm_candidates = list(norm_map.keys())

        detected_list: list[str] = []
        score_list: list[float] = []
        flag_list: list[str] = []
        claude_calls = 0

        for raw in df[col].astype(str).tolist():
            # Null / missing
            if raw.strip().lower() in ("", "nan", "none", "n/a", "na", "null"):
                detected_list.append("")
                score_list.append(1.0)
                flag_list.append("Normal (null/missing)")
                continue

            norm = _normalize(raw)

            # Exact normalised match
            if norm in norm_map:
                detected_list.append("")
                score_list.append(1.0)
                flag_list.append("Normal (exact)")
                continue

            # Fuzzy match
            if norm_candidates:
                best, ratio = _fuzzy_best(norm, norm_candidates)
                if best is not None and ratio >= fuzzy_threshold:
                    detected_list.append("")
                    score_list.append(round(ratio, 4))
                    flag_list.append(f"Normal (fuzzy {ratio:.2f})")
                    continue

            # Claude semantic judgment
            if claude_client and expected_values:
                verdict, score, reason = _judge_claude(
                    col, domain_prompt, expected_values, raw, claude_client, claude_model
                )
                claude_calls += 1
                if verdict == "NORMAL":
                    detected_list.append("")
                    score_list.append(score)
                    flag_list.append(f"Normal (claude: {reason})")
                else:
                    detected_list.append(raw)
                    score_list.append(score)
                    flag_list.append(f"Anomaly (claude: {reason})")
            else:
                # No Claude — flag as out-of-vocabulary
                detected_list.append(raw)
                score_list.append(0.0)
                flag_list.append("Anomaly (out-of-vocabulary)")

        annotated = _insert_adjacent(annotated, col, {
            f"{col}_score":             score_list,
            f"{col}_flag":              flag_list,
            f"{col}_detected_anomaly":  detected_list,
        })

        n_anomaly = sum(1 for f in flag_list if f.startswith("Anomaly"))
        n_non_null = sum(1 for f in flag_list if "null/missing" not in f)

        summary_rows.append({
            "column":         col,
            "type":           "categorical",
            "method":         "exact+fuzzy+claude" if claude_client else "exact+fuzzy",
            "non_null":       n_non_null,
            "anomaly_count":  n_anomaly,
            "anomaly_pct":    round(n_anomaly / n_non_null * 100, 2) if n_non_null else 0.0,
            "col_mean":       None,
            "col_std":        None,
            "anomaly_min":    None,
            "anomaly_max":    None,
            "claude_calls":   claude_calls,
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("anomaly_count", ascending=False)
        .reset_index(drop=True)
    )
    return annotated, summary_df


def anomaly_summary(summary_df: pd.DataFrame) -> str:
    """Return a plain-English summary of detect_anomalies() results."""
    n_cols = len(summary_df)
    total_anomalies = int(summary_df["anomaly_count"].sum())
    total_claude = int(summary_df.get("claude_calls", pd.Series([0])).sum())

    lines = [
        "ANOMALY DETECTION SUMMARY",
        "=" * 60,
        f"Columns analysed : {n_cols}",
        f"Total anomalies  : {total_anomalies:,}",
    ]
    if total_claude > 0:
        lines.append(f"Claude API calls : {total_claude} (unique values judged semantically)")
    lines.append("")

    flagged = summary_df[summary_df["anomaly_count"] > 0]
    clean = summary_df[summary_df["anomaly_count"] == 0]

    if flagged.empty:
        lines.append("[OK] No anomalies detected across all analysed columns.")
    else:
        lines.append("COLUMNS WITH ANOMALIES:")
        for _, row in flagged.iterrows():
            bar = "#" * min(30, max(1, int(row["anomaly_pct"] / 2)))
            lines.append(
                f"  {row['column']:<36s} {row['anomaly_count']:6,} anomalies "
                f"({row['anomaly_pct']:5.1f}%)  [{row['type'][:11]}]  {bar}"
            )
            if row["type"] == "numeric" and row["anomaly_min"] is not None:
                lines.append(
                    f"    -> outlier range: [{row['anomaly_min']} - {row['anomaly_max']}]"
                    f"  mean={row['col_mean']}  std={row['col_std']}"
                )

    if not clean.empty:
        lines.append(f"\nColumns with no anomalies ({len(clean)}):")
        for _, row in clean.iterrows():
            lines.append(f"  {row['column']:<36s} [OK]  [{row['type'][:11]}]")

    lines.append("\nREMEDIATION GUIDE:")
    for _, row in flagged.iterrows():
        if row["type"] == "numeric":
            lines.append(
                f"  {row['column']}: check outlier range [{row['anomaly_min']} - {row['anomaly_max']}] "
                f"against source data; consider winsorising or capping."
            )
        else:
            lines.append(
                f"  {row['column']}: review out-of-vocabulary / anomalous values; "
                f"standardise encoding or expand expected-values list in columns_config."
            )

    result = "\n".join(lines)
    return result.encode("ascii", errors="replace").decode("ascii")
