# core/rag.py
# ──────────────────────────────────────────────────────────────────────────────
# Answer generation using Google Gemini 2.0 Flash via Vertex AI.
# Project : genai-chatsearch
# Region  : us-east4
# Model   : gemini-2.0-flash
#
# Authentication: on Google Cloud Run, Application Default Credentials (ADC)
# are set automatically — no key file needed.
# For local development, run:  gcloud auth application-default login
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Vertex AI Config ──────────────────────────────────────────────────────────
GCP_PROJECT = os.getenv("GCP_PROJECT",  "genai-chatsearch")
GCP_REGION = os.getenv("GCP_REGION",   "us-east4")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


# ── Vertex AI initialisation (cached) ────────────────────────────────────────
_VERTEX_INIT = False


def _init_vertex() -> None:
    """
    Initialise Vertex AI once per process.
    On Cloud Run, ADC picks up credentials from the attached service account
    automatically — nothing extra needed.
    Locally: run `gcloud auth application-default login` first.
    """
    global _VERTEX_INIT
    if not _VERTEX_INIT:
        import vertexai
        vertexai.init(project=GCP_PROJECT, location=GCP_REGION)
        _VERTEX_INIT = True


# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = """\
You are Trini, a friendly and patient document assistant.
Your job is to explain complex document content clearly and simply — \
as if you are explaining it to a curious 15-year-old.
Read the context carefully and answer the user's question.
Always cite your sources using bracketed numbers like [1], [2] \
that refer to the numbered sources shown in the context.
Do not invent or guess facts that are not in the context.
If the question cannot be answered from the context, say so clearly and kindly.\
"""

USER_TEMPLATE = """\
Question:
{question}

Context (with numbered sources):
{context}

Instructions:
- Write a clear, friendly answer in 3–6 sentences.
- Use simple language, short sentences, and avoid jargon.
- Use bracketed citations [1], [2], … whenever you state a specific fact.
- If multiple sources support the same point, cite them all (e.g., [1][3]).
- Do NOT include raw file paths — just the bracketed numbers.
- Answer directly without any preamble or introduction.\
"""


# ── Public API ────────────────────────────────────────────────────────────────
def generate_answer(
    query: str,
    hits: List[Dict[str, Any]],
    max_chars_context: int = 12_000,
) -> str:
    """
    Generate a cited, plain-English answer from retrieved chunks.

    Parameters
    ----------
    query             : the user's question
    hits              : ranked list of chunk dicts  {"text", "source", "page", …}
    max_chars_context : safety cap on total context length sent to the model

    Returns
    -------
    str — Gemini's answer with inline citations, or a readable fallback.
    """
    context, _ = _build_context(hits, max_chars=max_chars_context)

    if not context.strip():
        return (
            "I couldn't find any relevant text in the document to answer "
            "your question. Try rephrasing or uploading a different document."
        )

    try:
        return _generate_with_gemini(query, context)
    except Exception as exc:
        # Surface the error in dev; fall back to raw-excerpt display
        print(f"[rag.py] Gemini call failed: {exc}")
        return _fallback_extract(query, hits)


# ── Context builder ───────────────────────────────────────────────────────────
def _build_context(
    hits: List[Dict[str, Any]],
    max_chars: int = 12_000,
) -> tuple[str, List[str]]:
    """
    Format retrieved chunks as a numbered source list:

        [1] SOURCE=report.pdf  PAGE=3
            <chunk text>

        [2] SOURCE=report.pdf  PAGE=7
            <chunk text>
        …

    Returns (context_string, citation_labels).
    """
    parts: List[str] = []
    citation_map: List[str] = []
    running = 0

    for i, h in enumerate(hits, start=1):
        src = h.get("source", "Unknown source")
        page = h.get("page",   "-")
        txt = (h.get("text") or "").strip()
        label = f"[{i}] SOURCE={src}  PAGE={page}"
        block = f"{label}\n{txt}\n"

        if running + len(block) > max_chars:
            remaining = max_chars - running
            if remaining <= 0:
                break
            block = block[:remaining]

        parts.append(block)
        citation_map.append(f"{Path(str(src)).name} (page {page})")
        running += len(block)

        if running >= max_chars:
            break

    return "\n\n".join(parts), citation_map


# ── Gemini backend ────────────────────────────────────────────────────────────
def _generate_with_gemini(query: str, context: str) -> str:
    """
    Call Gemini 2.0 Flash via Vertex AI and return the answer text.
    """
    _init_vertex()

    from vertexai.generative_models import GenerativeModel, GenerationConfig

    model = GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_INSTRUCTIONS,
    )

    prompt = USER_TEMPLATE.format(question=query, context=context)

    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.2,       # low temperature → factual, consistent answers
            max_output_tokens=800,  # enough for a thorough but concise answer
        ),
    )

    return response.text.strip()


# ── No-LLM fallback ───────────────────────────────────────────────────────────
def _fallback_extract(query: str, hits: List[Dict[str, Any]]) -> str:
    """
    Shown when Gemini is unavailable (e.g. local dev without ADC set up).
    Displays the most relevant raw excerpts in a readable format.
    """
    if not hits:
        return "No retrieved evidence to answer the question."

    prioritised = _prioritise_conclusionish(hits)

    lines = ["**Here's what I found in your document:**", ""]
    for i, h in enumerate(prioritised[:4], start=1):
        src = h.get("source", "Unknown")
        page = h.get("page",   "-")
        text = (h.get("text") or "").strip().replace("\n", " ")
        snippet = textwrap.shorten(text, width=400, placeholder="…")
        lines.append(f"**[{i}] {Path(str(src)).name} — page {page}**")
        lines.append(snippet)
        lines.append("")

    lines.append(
        "_Note: AI explanation is unavailable right now. "
        "Showing raw excerpts from your document._"
    )
    return "\n".join(lines)


def _prioritise_conclusionish(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heuristic: surface chunks that contain conclusion/results language first."""
    KEYWORDS = (
        "conclusion", "results", "findings", "outcome",
        "discussion", "summary", "key", "important", "main",
    )

    def score(h: Dict[str, Any]) -> int:
        t = (h.get("text") or "").lower()
        return sum(1 for k in KEYWORDS if k in t)

    return sorted(hits, key=score, reverse=True)
