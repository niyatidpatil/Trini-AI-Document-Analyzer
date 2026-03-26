# core/reranker.py
from __future__ import annotations
from typing import List, Optional
import numpy as np

from sentence_transformers import CrossEncoder

# Cache the model so it doesn't reload over and over
_CE: Optional[CrossEncoder] = None

def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Cross-encoder that scores (query, passage) pairs.
    This one is small(ish), fast, and good quality.
    """
    global _CE
    if _CE is None:
        _CE = CrossEncoder(model_name)
    return _CE

def rerank_scores(query: str, passages: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Returns a numpy array of scores (higher = better) for each passage.
    """
    if not passages:
        return np.zeros((0,), dtype="float32")
    ce = load_cross_encoder()
    pairs = [(query, p) for p in passages]
    scores = ce.predict(pairs, batch_size=batch_size)
    # Convert to numpy for easy sorting
    return np.array(scores, dtype="float32")

def rerank_topk(query: str, passages: List[str], k: int) -> List[int]:
    """
    Return indices of the top-k passages after reranking.
    """
    scores = rerank_scores(query, passages)
    if scores.size == 0:
        return []
    order = np.argsort(-scores)  # descending
    return order[:k].tolist()
