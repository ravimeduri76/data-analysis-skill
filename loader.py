"""Dataset loader — detects format from extension and returns a pandas DataFrame."""

from pathlib import Path
import pandas as pd


def _check_dep(package: str, install_name: str) -> None:
    """Raise ImportError with a clear install hint if a package is missing."""
    import importlib
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"Missing optional dependency '{package}'. "
            f"Install it with:  pip install {install_name}"
        )


def _load_excel(path: Path, engine: str | None = None) -> pd.DataFrame:
    """Load an Excel file, auto-detecting blank leading rows used as padding.

    Strategy: read with no header (header=None) so every row is data, then
    find the first row that looks like column headers (mostly strings, no
    Unnamed placeholders). Use that row index as the header row.
    """
    _check_dep("openpyxl", "openpyxl")

    # Read raw with no header so all rows are visible as data
    raw = pd.read_excel(path, header=None, engine=engine, nrows=20)

    header_row = 0  # default: first row is header
    for i, row in raw.iterrows():
        non_null = row.dropna()
        if len(non_null) == 0:
            continue  # blank row — keep scanning
        # A real header row has mostly string values
        str_count = sum(isinstance(v, str) for v in non_null)
        if str_count >= max(1, len(non_null) // 2):
            header_row = i
            break

    df = pd.read_excel(path, header=header_row, engine=engine)

    # Drop columns that are entirely unnamed AND entirely null (padding cols)
    unnamed_null = [
        c for c in df.columns
        if str(c).startswith("Unnamed:") and df[c].isna().all()
    ]
    if unnamed_null:
        df = df.drop(columns=unnamed_null)

    if header_row > 0:
        print(f"INFO: Skipped {header_row} leading blank row(s) in Excel file.")

    return df


def _load_ods(path: Path) -> pd.DataFrame:
    _check_dep("odf", "odfpy")
    return _load_excel(path, engine="odf")


def _read_csv_robust(path, sep=","):
    """Try multiple encodings so non-UTF-8 CSVs load without error."""
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Could not read CSV with any of the attempted encodings: {path}")


_LOADERS = {
    ".csv":     lambda p: _read_csv_robust(p),
    ".tsv":     lambda p: _read_csv_robust(p, sep="\t"),
    ".json":    lambda p: _load_json(p),
    ".parquet": lambda p: pd.read_parquet(p),
    ".xlsx":    lambda p: _load_excel(p),
    ".xls":     lambda p: _load_excel(p),
    ".ods":     lambda p: _load_ods(p),
}

_MAX_ROWS = 50_000


def load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataset file into a DataFrame.

    Raises FileNotFoundError if the path does not exist.
    Raises ValueError if the extension is unsupported.
    Warns and samples if the dataset exceeds 50,000 rows.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = p.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(_LOADERS)}"
        )

    df = loader(p)

    if len(df) > _MAX_ROWS:
        print(
            f"WARNING: Dataset has {len(df):,} rows (limit {_MAX_ROWS:,}). "
            f"Sampling {_MAX_ROWS:,} rows for analysis."
        )
        df = df.sample(_MAX_ROWS, random_state=42).reset_index(drop=True)

    return df


def save_csv(df: pd.DataFrame, path: str) -> str:
    """Save a DataFrame to CSV with UTF-8 BOM encoding (safe for Excel on Windows).

    Returns the absolute path of the saved file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return str(out.resolve())


def _load_json(path: Path) -> pd.DataFrame:
    """Load JSON — handles both records-list and nested objects via json_normalize."""
    import json
    from pandas import json_normalize

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return json_normalize(raw)
    if isinstance(raw, dict):
        # Try common envelope keys, then normalise the whole dict
        for key in ("data", "records", "items", "results"):
            if key in raw and isinstance(raw[key], list):
                return json_normalize(raw[key])
        return json_normalize([raw])

    raise ValueError("JSON root must be a list or dict.")
