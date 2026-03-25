"""Column classifier — labels each column as Identifier, Fact, Dimension, or Context.

Rules (pure logic, no LLM):
  Identifier  — unique values == row count  (likely a key / ID column)
  Fact        — numeric dtype              (measurable quantity)
  Dimension   — non-numeric, low cardinality (≤ 20 % unique ratio, max 50 distinct)
  Context     — everything else            (free text, high-cardinality strings)
"""

import re
import pandas as pd

_DIM_MAX_UNIQUE = 50
_DIM_MAX_RATIO = 0.20    # unique / total rows


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "identifiers": [],
        "facts":       [],
        "dimensions":  [],
        "context":     [],
    }

    n = len(df)
    for col in df.columns:
        s = df[col]
        nuniq = s.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(s)

        # Identifier: all-unique AND either non-numeric, or column name looks like a key
        _id_name = re.search(r'\b(id|key|uuid|code|ref|no|num|number|sku)\b', col, re.IGNORECASE)
        if nuniq == n and n > 1 and (not is_numeric or bool(_id_name)):
            result["identifiers"].append(col)
        elif is_numeric:
            result["facts"].append(col)
        elif nuniq <= _DIM_MAX_UNIQUE and (n == 0 or nuniq / n <= _DIM_MAX_RATIO):
            result["dimensions"].append(col)
        else:
            result["context"].append(col)

    return result
