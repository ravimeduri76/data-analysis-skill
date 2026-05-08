"""
embedding_registry.py - Shared BGE-M3 embedding singleton via gaxl-embedding-sdk.

Delegates model loading and inference to SentenceTransformersProvider from
gaxl-embedding-sdk, while preserving the legacy public API so existing graph
hooks (domain_graph, intent_graph, execution_graph, semantic_store) need no
changes.

Public API
----------
get_embed_model()  -> SentenceTransformer | None   # cached singleton
embed_texts(texts) -> np.ndarray                   # (N, 1024) L2-normalised
cosine_rank(q_vec, candidates) -> list             # sorted (index, score) pairs

Notes
-----
- _MODEL_CACHE keyed by "model" stores the SDK provider (not a raw
  SentenceTransformer). Tests that set _MODEL_CACHE["model"] = None to
  simulate unavailability still work because embed_texts raises when the
  cached entry is None.
- normalize_embeddings=True is passed to the SDK provider so vectors come
  back L2-normalised; cosine similarity == dot product (the harness contract
  relied on by run_embeddings, intent_nodes, domain_node_embeddings, etc.).
- GAXL_EMBED_DEVICE env var controls device placement (default: cpu).
  Passed to SentenceTransformersProvider's device kwarg — required because
  sentence-transformers' autoselect picks Intel XPU on this box, which
  fails at inference time.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from gaxl_embedding_sdk.dto import EmbeddingRequest

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-m3"
_DEVICE = os.environ.get("GAXL_EMBED_DEVICE", "cpu")
_MODEL_CACHE: dict = {}
_LAST_ERROR: Optional[str] = None


def _load_hf_token() -> None:
    """Set HF_TOKEN in environment from .env if not already present."""
    if os.environ.get("HF_TOKEN"):
        return
    env_path = os.path.join(os.path.expanduser("~"), "Data_Analysis", ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN=") and len(line) > 9:
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                    return
    except OSError:
        pass


def _get_provider():
    """Lazy-load the SDK provider; cache singleton; pin model to configured device.

    Only successful providers are cached. Failures are logged at WARNING and
    re-raised; callers (embed_texts) surface them with the original cause so
    transient errors (memory pressure, transient HF unreachable) can recover
    on the next call instead of poisoning the process.
    """
    global _LAST_ERROR
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]
    try:
        import warnings
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        _load_hf_token()
        from gaxl_embedding_sdk.providers.sentence_transformers import (
            SentenceTransformersProvider,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            provider = SentenceTransformersProvider(
                model_name=_MODEL_NAME,
                device=_DEVICE,
                normalize_embeddings=True,
            )
        _MODEL_CACHE["model"] = provider
        _LAST_ERROR = None
        return provider
    except Exception as exc:
        _LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logger.warning("BGE-M3 provider construction failed: %s", _LAST_ERROR,
                       exc_info=True)
        return None


def get_embed_model():
    """Return the underlying SentenceTransformer for callers that .encode() directly.

    Returns None if the SDK / sentence-transformers stack is unavailable.
    Kept for backward compatibility with domain_graph, intent_graph,
    execution_graph; new code should prefer embed_texts().
    """
    provider = _get_provider()
    if provider is None:
        return None
    return getattr(provider, "_model", None)


def reset_cache() -> None:
    """Clear the cached provider so the next call retries construction.

    Useful in long-running processes when a transient failure (e.g. OOM at
    cold start) left the cache poisoned in older versions; harmless after
    the no-cache-on-failure fix but kept for tests and operator recovery.
    """
    global _LAST_ERROR
    _MODEL_CACHE.clear()
    _LAST_ERROR = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, 1024).

    Goes through the SDK provider, which is configured with
    normalize_embeddings=True at construction time.
    """
    provider = _get_provider()
    if provider is None:
        cause = _LAST_ERROR or "unknown (no exception captured)"
        raise RuntimeError(f"BGE-M3 provider unavailable: {cause}")
    response = provider.embed(EmbeddingRequest(texts=list(texts)))
    return np.array(response.vectors, dtype=np.float32)


def cosine_rank(
    q_vec: np.ndarray,
    candidates: list[tuple],
) -> list[tuple[int, float]]:
    """Rank candidates by cosine similarity to query vector.

    Parameters
    ----------
    q_vec       : (1024,) L2-normalised query embedding
    candidates  : list of (index_or_id, embedding_list_or_array) tuples

    Returns
    -------
    list of (index_or_id, score) tuples sorted descending by score
    """
    results = []
    for idx, emb in candidates:
        if not emb:
            continue
        vec = np.array(emb, dtype=np.float32)
        score = float(np.dot(q_vec, vec))
        results.append((idx, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results
