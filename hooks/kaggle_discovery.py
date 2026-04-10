"""Kaggle dataset discovery and download hook.

Discovers trending Kaggle datasets, filters by size and format,
applies domain diversity heuristics, and downloads the best candidate.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag-to-domain mapping (Kaggle tags → internal domain names)
# ---------------------------------------------------------------------------
TAG_DOMAIN_MAP: dict[str, str] = {
    "healthcare": "Healthcare",
    "health": "Healthcare",
    "hospitals": "Healthcare",
    "medicine": "Healthcare",
    "finance": "Financial Services",
    "banking": "Financial Services",
    "lending": "Financial Services / Lending",
    "insurance": "Insurance",
    "retail": "Retail & E-commerce",
    "ecommerce": "Retail & E-commerce",
    "e-commerce": "Retail & E-commerce",
    "travel": "Travel & Hospitality",
    "education": "Education",
    "human resources": "HR & People Analytics",
    "employment": "HR & People Analytics",
    "pharmaceutical": "Pharmaceutical",
    "energy": "Industrial Equipment",
    "manufacturing": "Mechanical Components",
    "supply chain": "Healthcare Supply Chain / GPO",
    "data science": "General",
}

# Tabular file extensions we accept
_TABULAR_EXTS = {".csv", ".json", ".parquet", ".xlsx", ".xls", ".tsv"}

# Max dataset size in bytes (100 MB)
_MAX_BYTES = 100 * 1024 * 1024

# Sort strategies rotated by day-of-month
_SORT_MODES = ["hottest", "votes", "updated"]


def _sort_mode_for_today() -> str:
    """Rotate sort mode by day-of-month (mod 3)."""
    return _SORT_MODES[datetime.now().day % len(_SORT_MODES)]


def _kaggle_list(sort_by: str, page: int) -> list[dict]:
    """Call `kaggle datasets list` and return parsed JSON."""
    cmd = [
        "kaggle", "datasets", "list",
        "--sort-by", sort_by,
        "--max-size", str(_MAX_BYTES),
        "--file-type", "csv",
        "--page", str(page),
        "--csv",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        log.error("kaggle CLI not found — run: pip install kaggle")
        return []
    except subprocess.TimeoutExpired:
        log.warning("kaggle CLI timed out (page %d)", page)
        return []

    if result.returncode != 0:
        # Kaggle CLI on Windows may return non-zero due to encoding issues
        # but still produce valid CSV on stdout — try to use it
        if not result.stdout.strip():
            err = result.stderr.strip()
            log.warning("kaggle CLI error: %s", err)
            return []
        log.debug("kaggle CLI returned %d but has stdout — proceeding", result.returncode)

    # Strip warning/info lines that kaggle CLI prints to stdout
    raw_lines = result.stdout.strip().splitlines()
    csv_lines = [ln for ln in raw_lines if not ln.startswith(("Warning:", "Info:", "Error:"))]
    if len(csv_lines) < 2:
        return []

    import csv as _csv
    import io
    reader = _csv.DictReader(io.StringIO("\n".join(csv_lines)))
    return list(reader)


def _dataset_domain(tags_str: str) -> Optional[str]:
    """Map Kaggle tag string to an internal domain name."""
    if not tags_str:
        return None
    for tag in tags_str.lower().replace(",", " ").split():
        tag = tag.strip()
        if tag in TAG_DOMAIN_MAP:
            return TAG_DOMAIN_MAP[tag]
    return None


def discover_dataset(
    analyzed_slugs: set[str],
    strategy: str = "auto",
    domain_history: list[str] | None = None,
    max_pages: int = 15,
) -> dict | None:
    """Find one new Kaggle dataset to analyze.

    Strategy:
    1. Try primary sort mode with domain diversity filter (up to max_pages).
    2. If nothing found, rotate through ALL sort modes with diversity.
    3. If still nothing, relax domain diversity and try all modes again.

    Returns a dict with keys: slug, title, size, url, tags, domain_hint
    or None if nothing suitable is found.
    """
    primary_sort = _sort_mode_for_today() if strategy == "auto" else strategy
    recent_domains = set(domain_history[-3:]) if domain_history else set()

    log.info("Discovering dataset (primary_sort=%s, skip=%d slugs, recent_domains=%s)",
             primary_sort, len(analyzed_slugs), recent_domains)

    # --- Pass 1: primary sort mode, with domain diversity ---
    result = _scan_pages(primary_sort, max_pages, analyzed_slugs, recent_domains)
    if result:
        return result

    # --- Pass 2: try other sort modes, with domain diversity ---
    for mode in _SORT_MODES:
        if mode == primary_sort:
            continue
        log.info("Pass 2 — trying sort mode: %s", mode)
        result = _scan_pages(mode, max_pages, analyzed_slugs, recent_domains)
        if result:
            return result

    # --- Pass 3: relax domain diversity, try all sort modes ---
    log.info("Pass 3 — relaxing domain diversity filter")
    for mode in _SORT_MODES:
        result = _scan_pages(mode, max_pages, analyzed_slugs, skip_domains=set())
        if result:
            return result

    log.warning("No suitable dataset found after exhaustive search (%d pages x %d modes)",
                max_pages, len(_SORT_MODES))
    return None


def _scan_pages(
    sort_by: str,
    max_pages: int,
    analyzed_slugs: set[str],
    skip_domains: set[str],
) -> dict | None:
    """Scan up to max_pages of Kaggle listings, skipping analyzed slugs and
    optionally skipping datasets whose domain is in skip_domains."""
    for page in range(1, max_pages + 1):
        rows = _kaggle_list(sort_by, page)
        if not rows:
            break

        for row in rows:
            slug = row.get("ref", "").strip()
            if not slug or slug in analyzed_slugs:
                continue

            domain_hint = _dataset_domain(row.get("tags", ""))
            if skip_domains and domain_hint and domain_hint in skip_domains:
                log.debug("Skipping %s — domain %s seen recently", slug, domain_hint)
                continue

            title = row.get("title", slug.split("/")[-1])
            size = int(row.get("totalBytes", 0) or 0)
            url = f"https://www.kaggle.com/datasets/{slug}"

            log.info("Selected: %s (%s, %d bytes, sort=%s, page=%d)",
                     slug, title, size, sort_by, page)
            return {
                "slug": slug,
                "title": title,
                "size": size,
                "url": url,
                "tags": row.get("tags", ""),
                "domain_hint": domain_hint,
            }

    return None


def download_dataset(slug: str, dest_dir: str) -> list[str]:
    """Download and unzip a Kaggle dataset. Returns list of extracted file paths."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        slug,
        "-p", str(dest),
        "--unzip",
    ]
    log.info("Downloading %s -> %s", slug, dest)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        log.error("Download timed out for %s", slug)
        return []

    if result.returncode != 0:
        log.error("Download failed: %s", result.stderr.strip())
        return []

    # Handle case where --unzip didn't work (some datasets come as .zip)
    for zf in dest.glob("*.zip"):
        try:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(dest)
            zf.unlink()
        except Exception as exc:
            log.warning("Failed to unzip %s: %s", zf, exc)

    files = [str(f) for f in dest.rglob("*") if f.is_file()]
    log.info("Downloaded %d file(s)", len(files))
    return files


def pick_best_file(directory: str) -> str | None:
    """Pick the largest tabular file in a directory."""
    best: tuple[int, str] | None = None
    for f in Path(directory).rglob("*"):
        if f.suffix.lower() in _TABULAR_EXTS and f.is_file():
            sz = f.stat().st_size
            if best is None or sz > best[0]:
                best = (sz, str(f))
    if best:
        log.info("Best file: %s (%d bytes)", best[1], best[0])
    return best[1] if best else None
