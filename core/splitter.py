from __future__ import annotations
from typing import List, Dict, Any
import tiktoken

# token-aware chunker using OpenAI cl100k_base tokenizer

def chunk_text(
        text: str,
        chunk_size: int = 750,
        chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Returns a list of pieces with exact token spans:
    [{"text": str, "start_token": int, "end_token": int}, ...]
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text or "")
    pieces: List[Dict[str, Any]] = []

    if not tokens:
        return pieces

    # Guard against pathological params
    if chunk_overlap >= chunk_size and chunk_size > 0:
        # keep a small overlap instead of exploding redundancy
        chunk_overlap = max(0, chunk_size // 5)

    start = 0
    n = len(tokens)

    while start < n:
        end = min(n, start + chunk_size)
        chunk_tokens = tokens[start:end]
        piece_text = enc.decode(chunk_tokens)

        pieces.append({
            "text": piece_text,
            "start_token": start,
            "end_token": end,  # exclusive
        })

        if end >= n:
            break

        # slide window with overlap
        start = max(0, end - chunk_overlap)

    return pieces


def chunk_docs(
        docs: List,  # accepts Doc dataclass or dict-like objects
        chunk_size: int = 750,
        chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Split each document (Doc or dict) into overlapping chunks.
    Adds 'chunk', 'num_chunks', and exact token spans for traceability.
    """
    out: List[Dict[str, Any]] = []

    for d in docs:
        # Support Doc dataclass and dict-like inputs
        if hasattr(d, "content") and hasattr(d, "metadata"):
            content = getattr(d, "content", "") or ""
            metadata = getattr(d, "metadata", {}) or {}
        elif isinstance(d, dict):
            content = d.get("content", "") or ""
            metadata = d.get("metadata", {}) or {}
        else:
            content, metadata = "", {}

        pieces = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for i, piece in enumerate(pieces):
            meta = {
                **metadata,
                "chunk": i,
                "num_chunks": len(pieces),
                "start_token": piece["start_token"],
                "end_token": piece["end_token"],  # exclusive
            }

            out.append({
                "content": piece["text"],
                "metadata": meta,
            })

    return out
