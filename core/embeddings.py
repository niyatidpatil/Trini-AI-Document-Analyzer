# core/embeddings.py
# ──────────────────────────────────────────────────────────────────────────────
# Vector store backed by Pinecone (cloud) + SentenceTransformer for embeddings.
# Embedding model : all-MiniLM-L6-v2  →  384-dimensional vectors
# Pinecone index  : trini-ai  |  us-east-1  |  384 dims  |  cosine metric
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "trini-ai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384          # must match Pinecone index dimension

# ── Cached singletons (avoids reloading model on every call) ──────────────────
_ENCODER: Optional[SentenceTransformer] = None
_PC_INDEX = None                  # Pinecone Index object

# ── Local docstore (full text backup — Pinecone metadata has a 40 KB limit) ───
STORE_DIR = Path(".store")
STORE_DIR.mkdir(exist_ok=True)
DOCSTORE_PATH = STORE_DIR / "docstore.json"
CFG_PATH = STORE_DIR / "config.json"


# ── Encoder ───────────────────────────────────────────────────────────────────
def load_encoder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the SentenceTransformer once and keep it in memory."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer(model_name)
    return _ENCODER


# ── Pinecone connection ───────────────────────────────────────────────────────
def get_pinecone_index():
    """Connect to Pinecone and return the index object (cached after first call)."""
    global _PC_INDEX
    if _PC_INDEX is None:
        if not PINECONE_API_KEY:
            raise EnvironmentError(
                "PINECONE_API_KEY is not set. "
                "Add it to your .env file or Cloud Run environment variables."
            )
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _PC_INDEX = pc.Index(PINECONE_INDEX)
    return _PC_INDEX


# ── Math helpers ──────────────────────────────────────────────────────────────
def l2_normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalise each row to unit length so that inner-product == cosine similarity.
    Small epsilon prevents division by zero.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


# ── Embedding ─────────────────────────────────────────────────────────────────
def embed_texts(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """
    Convert a list of strings → float32 numpy array of shape (N, 384).
    progress_cb(done, total) is called after every batch if provided.
    """
    enc = load_encoder(model_name)
    embs: List[np.ndarray] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i: i + batch_size]
        batch_embs = enc.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embs.append(batch_embs)
        if progress_cb:
            progress_cb(min(i + len(batch), total), total)

    X = np.vstack(embs) if embs else np.zeros(
        (0, EMBEDDING_DIM), dtype="float32")
    return l2_normalize(X.astype("float32"))


def embed_query(query: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a single query string → 1-D float32 array of shape (384,)."""
    enc = load_encoder(model_name)
    q = enc.encode([query], convert_to_numpy=True, show_progress_bar=False)
    return l2_normalize(q.astype("float32"))[0]


# ── Vector ID ─────────────────────────────────────────────────────────────────
def make_vector_id(source: str, page: Any, chunk: Any) -> str:
    """
    Create a stable, unique ID for each chunk so Pinecone can deduplicate
    on re-upload (upsert replaces existing vectors with the same ID).
    """
    raw = f"{source}|{page}|{chunk}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ── Upsert into Pinecone ──────────────────────────────────────────────────────
def upsert_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 100,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Embed `chunks` and upsert them into Pinecone.

    Each chunk dict must look like:
        {"content": "...", "metadata": {"source": "...", "page": N, "chunk": N, ...}}

    Returns the number of vectors successfully upserted.
    """
    if not chunks:
        return 0

    index = get_pinecone_index()
    texts = [c.get("content", "") for c in chunks]
    embs = embed_texts(texts, model_name=model_name, progress_cb=progress_cb)

    vectors = []
    for chunk, emb in zip(chunks, embs):
        meta = chunk.get("metadata", {})
        source = str(meta.get("source",    "unknown"))
        page = meta.get("page",          0)
        chunk_i = meta.get("chunk",         0)
        content = chunk.get("content",      "")

        vectors.append({
            "id":     make_vector_id(source, page, chunk_i),
            "values": emb.tolist(),
            "metadata": {
                # Store the text in metadata so we can retrieve it without a
                # separate lookup. Pinecone allows up to 40 KB per vector.
                "text":       content[:3000],       # ~750 tokens ≈ 3 000 chars
                "source":     source,
                "page":       page,
                "chunk":      chunk_i,
                "num_chunks": meta.get("num_chunks", 1),
            },
        })

    # Pinecone recommends batches ≤ 100 vectors
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    return total_upserted


# ── Query Pinecone ────────────────────────────────────────────────────────────
def search(
    query_emb: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone with a pre-computed embedding.
    Returns a list of hit dicts: {"text", "source", "page", "chunk", "score"}.
    """
    index = get_pinecone_index()

    if query_emb.ndim > 1:
        query_emb = query_emb.flatten()

    results = index.query(
        vector=query_emb.tolist(),
        top_k=top_k,
        include_metadata=True,
    )

    hits: List[Dict[str, Any]] = []
    for match in results.matches:
        m = match.metadata or {}
        hits.append({
            "text":   m.get("text",   ""),
            "source": m.get("source", "Unknown"),
            "page":   m.get("page",   "-"),
            "chunk":  m.get("chunk",  0),
            "score":  match.score,
        })
    return hits


# ── Fingerprinting (cache invalidation) ───────────────────────────────────────
def fingerprint_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    MD5 fingerprint of the chunk set.
    If this changes between runs, the index needs to be rebuilt.
    """
    m = hashlib.md5()
    for ch in chunks:
        meta = ch.get("metadata", {})
        m.update(
            ("|".join([
                str(meta.get("source", "")),
                str(meta.get("page",   "")),
                str(meta.get("chunk",  "")),
                str(len(ch.get("content", ""))),
            ]) + "\n").encode("utf-8")
        )
    return m.hexdigest()


# ── Local docstore helpers ────────────────────────────────────────────────────
def save_docstore(rows: List[Dict[str, Any]], path: Path = DOCSTORE_PATH) -> None:
    """Persist full-text docstore locally as a JSON backup."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def load_docstore(path: Path = DOCSTORE_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict[str, Any], path: Path = CFG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_config(path: Path = CFG_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────────────
def mmr_select(
    query_vec: np.ndarray,
    cand_texts: List[str],
    embed_fn: Callable[[List[str]], np.ndarray],
    top_k: int,
    lambda_relevance: float = 0.7,
) -> List[int]:
    """
    Select top_k diverse + relevant candidates using MMR.

    MMR score = λ · relevance(query, cᵢ)  −  (1−λ) · max_j∈S similarity(cᵢ, cⱼ)

    lambda_relevance = 1.0  →  pure relevance (no diversity)
    lambda_relevance = 0.0  →  pure diversity (no relevance)
    Default 0.7 balances both.

    Returns list of indices into `cand_texts` in selection order.
    """
    if not cand_texts:
        return []

    X = embed_fn(cand_texts)                   # (N, d) — already L2-normalised
    q = query_vec.reshape(1, -1)               # (1, d)
    rel = (q @ X.T).flatten()                    # cosine relevance scores (N,)

    selected:   List[int] = []
    candidates: set = set(range(len(cand_texts)))

    while len(selected) < min(top_k, len(cand_texts)):
        if not selected:
            # First pick: highest relevance
            i = int(np.argmax(rel))
        else:
            X_sel = X[selected]              # (k, d)
            sim_selected = X @ X_sel.T              # (N, k)
            max_sim = sim_selected.max(axis=1)  # (N,)
            mmr_scores = lambda_relevance * rel - \
                (1 - lambda_relevance) * max_sim
            masked = np.full_like(mmr_scores, -1e9)
            for j in candidates:
                masked[j] = mmr_scores[j]
            i = int(np.argmax(masked))

        selected.append(i)
        candidates.remove(i)

    return selected
