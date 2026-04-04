"""
embedding_registry.py — Shared BGE-M3 embedding singleton and utilities.

Consolidates the duplicate _get_embed_model() implementations across
domain_graph.py, intent_graph.py, execution_graph.py into a single
cached module. All callers should delegate to this module.

Public API
----------
get_embed_model()  -> SentenceTransformer | None   # cached singleton
embed_texts(texts) -> np.ndarray                   # (N, 1024) L2-normalised
cosine_rank(q_vec, candidates) -> list              # sorted (index, score) pairs
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

_MODEL_NAME = "BAAI/bge-m3"
_MODEL_CACHE: dict = {}


def get_embed_model():
    """Load BGE-M3 model (cached singleton). Returns None if unavailable."""
    if "model" not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            import warnings
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _MODEL_CACHE["model"] = SentenceTransformer(_MODEL_NAME)
        except Exception:
            _MODEL_CACHE["model"] = None
    return _MODEL_CACHE["model"]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (N, 1024). Raises if model unavailable."""
    model = get_embed_model()
    if model is None:
        raise RuntimeError("BGE-M3 model unavailable (sentence-transformers not installed)")
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


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
