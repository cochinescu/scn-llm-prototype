"""In-memory toy retrieval used only to time the phase-modulation path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalParameters:
    temporal_window_hours: float
    semantic_weight: float
    recency_weight: float


def retrieval_parameters(state: np.ndarray) -> RetrievalParameters:
    """Turn the first-cut state vector into interpretable retrieval weights."""
    focus, energy = float(state[5]), float(state[6])
    return RetrievalParameters(
        temporal_window_hours=6.0 + 18.0 * focus,
        semantic_weight=0.55 + 0.35 * focus,
        recency_weight=0.10 + 0.35 * energy,
    )


def make_toy_corpus(size: int = 512, dimensions: int = 24, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(size, dimensions))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    timestamps = np.linspace(0, 24 * 30, num=size)
    return embeddings, timestamps


def retrieve(
    query: np.ndarray,
    embeddings: np.ndarray,
    timestamps: np.ndarray,
    now_hours: float,
    params: RetrievalParameters | None = None,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return top documents under baseline or phase-modulated toy retrieval."""
    query = query / np.linalg.norm(query)
    semantic = embeddings @ query
    if params is None:
        scores = semantic
    else:
        age = np.maximum(now_hours - timestamps, 0.0)
        recency = np.exp(-age / params.temporal_window_hours)
        scores = params.semantic_weight * semantic + params.recency_weight * recency
    indices = np.argsort(scores)[-top_k:][::-1]
    return indices, scores[indices]
