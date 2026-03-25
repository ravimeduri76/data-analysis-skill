"""Sensitive column sanitizer — redacts PII columns before any output."""

import re
import pandas as pd

_SENSITIVE_PATTERNS = re.compile(
    r"\b(name|email|phone|mobile|ssn|nric|dob|birth|password|passwd|"
    r"credit.?card|card.?number|cvv|pan|iban|address|zip|postal)\b",
    re.IGNORECASE,
)


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with sensitive columns replaced by '<REDACTED>'."""
    df = df.copy()
    for col in df.columns:
        if _SENSITIVE_PATTERNS.search(col):
            df[col] = "<REDACTED>"
    return df


def sensitive_columns(df: pd.DataFrame) -> list[str]:
    """Return list of column names that match the sensitive pattern."""
    return [c for c in df.columns if _SENSITIVE_PATTERNS.search(c)]
