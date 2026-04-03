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

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return []

    import csv as _csv
    import io
    reader = _csv.DictReader(io.StringIO(result.stdout))
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
) -> dict | None:
    """Find one new Kaggle dataset to analyze.

    Returns a dict with keys: slug, title, size, url, tags, domain_hint
    or None if nothing suitable is found.
    """
    sort_by = _sort_mode_for_today() if strategy == "auto" else strategy
    recent_domains = set(domain_history[-3:]) if domain_history else set()

    log.info("Discovering dataset (sort=%s, skip=%d slugs, recent_domains=%s)",
             sort_by, len(analyzed_slugs), recent_domains)

    for page in range(1, 6):  # up to 5 pages
        rows = _kaggle_list(sort_by, page)
        if not rows:
            break

        for row in rows:
            slug = row.get("ref", "").strip()
            if not slug or slug in analyzed_slugs:
                continue

            # Prefer domain diversity
            domain_hint = _dataset_domain(row.get("tags", ""))
            if domain_hint and domain_hint in recent_domains:
                log.debug("Skipping %s — domain %s seen recently", slug, domain_hint)
                continue

            title = row.get("title", slug.split("/")[-1])
            size = int(row.get("totalBytes", 0) or 0)
            url = f"https://www.kaggle.com/datasets/{slug}"

            log.info("Selected: %s (%s, %d bytes)", slug, title, size)
            return {
                "slug": slug,
                "title": title,
                "size": size,
                "url": url,
                "tags": row.get("tags", ""),
                "domain_hint": domain_hint,
            }

    log.warning("No suitable dataset found after scanning 5 pages")
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
