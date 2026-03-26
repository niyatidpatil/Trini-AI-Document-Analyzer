# app.py
# ──────────────────────────────────────────────────────────────────────────────
# Trini AI — Intelligent Document Analyzer
# Frontend  : Streamlit
# Vector DB : Pinecone  (cloud, persistent)
# LLM       : Gemini 2.0 Flash via Vertex AI
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np

# ── Our modules ───────────────────────────────────────────────────────────────
from core.loader import (
    load_pdfs, load_txts, load_urls,
    load_docx, load_pptx, load_xlsx,
    get_last_extraction_report, Doc,
)
from core.splitter import chunk_docs
from core.embeddings import (
    embed_texts,
    embed_query,
    upsert_chunks,       # ← sends vectors to Pinecone
    search,              # ← queries Pinecone
    fingerprint_chunks,
    save_docstore, load_docstore, save_config, load_config,
    DOCSTORE_PATH,
    mmr_select,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Trini AI", page_icon="🤖", layout="wide")
st.title("🤖 Trini AI — Intelligent Document Analyzer")
st.caption("Upload your documents and ask questions. Trini will explain them simply, with cited sources.")

# ── Session state initialisation ──────────────────────────────────────────────
defaults = {
    "docs":              [],
    "chunks":            [],
    "index_ready":       False,
    "fingerprint":       None,
    "search_results":    [],
    "last_query":        "",
    "raw_preview_chars": 1200,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "PDF, DOCX, TXT, PPTX, XLSX",
    type=["pdf", "txt", "docx", "pptx", "xlsx"],
    accept_multiple_files=True,
)

url_block = st.sidebar.text_area("Or paste URLs (one per line)")

st.sidebar.divider()
st.sidebar.header("⚙️ Chunking Settings")
chunk_size = st.sidebar.number_input(
    "Chunk size (tokens)",
    min_value=200, max_value=2000, value=750, step=50,
    help="Number of tokens per chunk. 750 is optimal for most documents.",
)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap (tokens)",
    min_value=0, max_value=500, value=150, step=10,
    help="Overlap between chunks so context is not lost at boundaries.",
)

st.sidebar.divider()
st.sidebar.header("🛠️ Debug")
show_raw_text = st.sidebar.checkbox("Show raw extracted text", value=False)
st.session_state.raw_preview_chars = st.sidebar.slider(
    "Preview length (chars)", min_value=200, max_value=4000, value=1200, step=100
)

col_btn1, col_btn2 = st.sidebar.columns(2)
build_btn = col_btn1.button("📥 Load Docs", use_container_width=True)
clear_btn = col_btn2.button("🗑️ Clear", use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
TMP_DIR = Path(".tmp_uploads")
TMP_DIR.mkdir(exist_ok=True)


def _save_uploads(files) -> Dict[str, List[str]]:
    out = {"pdf": [], "txt": [], "docx": [], "pptx": [], "xlsx": []}
    if not files:
        return out
    for uf in files:
        p = TMP_DIR / uf.name
        p.write_bytes(uf.getbuffer())
        key = p.suffix.lower().lstrip(".")
        if key in out:
            out[key].append(str(p))
    return out


def _ingest(upload_map: Dict[str, List[str]], urls: List[str]) -> List[Doc]:
    docs: List[Doc] = []
    if upload_map.get("pdf"):
        docs.extend(load_pdfs(upload_map["pdf"]))
    if upload_map.get("txt"):
        docs.extend(load_txts(upload_map["txt"]))
    if upload_map.get("docx"):
        docs.extend(load_docx(upload_map["docx"]))
    if upload_map.get("pptx"):
        docs.extend(load_pptx(upload_map["pptx"]))
    if upload_map.get("xlsx"):
        docs.extend(load_xlsx(upload_map["xlsx"]))
    if urls:
        docs.extend(load_urls(urls))
    return docs


# ── Actions ───────────────────────────────────────────────────────────────────
if clear_btn:
    for k in ["docs", "chunks", "search_results"]:
        st.session_state[k] = []
    st.session_state.last_query = ""
    st.session_state.index_ready = False
    st.session_state.fingerprint = None
    st.success("Cleared all documents and search results.")

if build_btn:
    saved = _save_uploads(uploaded_files)
    urls = [u.strip() for u in (url_block or "").splitlines() if u.strip()]

    with st.spinner("Reading and parsing documents..."):
        docs = _ingest(saved, urls)

    st.session_state.docs = docs
    st.session_state.chunks = []
    st.session_state.index_ready = False

    if docs:
        st.success(f"Loaded **{len(docs)}** document page(s).")
    else:
        st.warning("No documents loaded. Please upload at least one file.")

    if saved.get("pdf"):
        with st.expander("📊 PDF Extraction Report"):
            st.json(get_last_extraction_report())


# ── Section 1: Loaded Documents ───────────────────────────────────────────────
st.divider()
st.subheader("📄 Loaded Documents")
docs = st.session_state.docs

if not docs:
    st.info("Upload files in the sidebar and click **📥 Load Docs** to get started.")
else:
    st.write(f"**{len(docs)}** page(s) loaded.")

    with st.expander("Preview first document"):
        st.json({
            "metadata": docs[0].metadata,
            "content_preview": (
                docs[0].content[:st.session_state.raw_preview_chars]
                + ("..." if len(docs[0].content) >
                   st.session_state.raw_preview_chars else "")
            ),
        })

    if show_raw_text:
        st.markdown("##### Raw extracted text")
        for i, d in enumerate(docs):
            st.markdown(
                f"**Doc {i+1}** — `{d.metadata.get('source')}` | `{d.metadata.get('type')}`")
            st.code(
                d.content[:st.session_state.raw_preview_chars]
                + ("\n..." if len(d.content) >
                   st.session_state.raw_preview_chars else "")
            )


# ── Section 2: Chunking ───────────────────────────────────────────────────────
st.divider()
st.subheader("✂️ Token-Aware Chunking")

if not docs:
    st.info("No documents loaded yet.")
else:
    chunks = chunk_docs(docs, chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap))
    st.session_state.chunks = chunks

    st.write(
        f"Split into **{len(chunks)}** chunks ({chunk_size}-token, {chunk_overlap}-token overlap).")

    if chunks:
        with st.expander("Preview first 2 chunks"):
            for i, ch in enumerate(chunks[:2]):
                src = ch["metadata"].get("source")
                num = ch["metadata"].get("num_chunks", "?")
                st.markdown(f"**Chunk {i+1}/{num}** — `{src}`")
                st.code(ch["content"][:800] +
                        ("\n..." if len(ch["content"]) > 800 else ""))

        with st.expander("🔍 Chunk continuity audit"):
            if len(chunks) >= 2:
                total = len(chunks)
                idx = st.number_input(
                    "Boundary between chunk i and i+1",
                    min_value=0, max_value=total - 2, value=0, step=1,
                )
                st.caption("Last 200 chars of chunk i")
                st.code(chunks[idx]["content"][-200:])
                st.caption("First 200 chars of chunk i+1")
                st.code(chunks[idx + 1]["content"][:200])
            else:
                st.info("Need at least 2 chunks to audit a boundary.")


# ── Section 3: Embed & Push to Pinecone ──────────────────────────────────────
st.divider()
st.subheader("🧠 Embed & Store in Pinecone")

col_e1, col_e2 = st.columns(2)

with col_e1:
    embed_btn = st.button(
        "🚀 Embed & Upload to Pinecone",
        use_container_width=True,
        disabled=not bool(st.session_state.chunks),
    )

with col_e2:
    verify_btn = st.button(
        "✅ Verify Pinecone Connection",
        use_container_width=True,
    )

if embed_btn:
    chunks = st.session_state.chunks
    if not chunks:
        st.warning("No chunks to embed. Load and chunk documents first.")
    else:
        prog = st.progress(0, text="Embedding chunks...")

        def _cb(done, total):
            prog.progress(done / max(1, total),
                          text=f"Embedding... {done}/{total}")

        try:
            with st.spinner("Uploading vectors to Pinecone..."):
                total_upserted = upsert_chunks(chunks, progress_cb=_cb)

            # Save fingerprint + local docstore backup
            fp = fingerprint_chunks(chunks)
            cfg = {"fingerprint": fp, "model": "all-MiniLM-L6-v2", "dim": 384}
            save_config(cfg)

            metas = []
            for c in chunks:
                md = c["metadata"].copy()
                metas.append({
                    "source":     md.get("source"),
                    "file_name":  md.get("file_name"),
                    "page":       md.get("page"),
                    "chunk":      md.get("chunk"),
                    "num_chunks": md.get("num_chunks"),
                    "text":       c.get("content", ""),
                })
            save_docstore(metas)

            st.session_state.fingerprint = fp
            st.session_state.index_ready = True

            prog.empty()
            st.success(
                f"✅ **{total_upserted}** vectors upserted into Pinecone index `trini-ai`. "
                f"Your document is ready to query!"
            )
        except Exception as e:
            st.error(f"Pinecone upload failed: {e}")
            st.info(
                "Check that PINECONE_API_KEY is set in your environment variables.")

if verify_btn:
    try:
        from core.embeddings import get_pinecone_index
        pc_index = get_pinecone_index()
        stats = pc_index.describe_index_stats()
        st.success(
            f"✅ Connected to Pinecone `trini-ai`  |  "
            f"**{stats.total_vector_count}** vectors stored  |  "
            f"Dimension: **{stats.dimension}**"
        )
    except Exception as e:
        st.error(f"Could not connect to Pinecone: {e}")
        st.info(
            "Ensure PINECONE_API_KEY is set and the `trini-ai` index exists in us-east-1.")

if st.session_state.index_ready:
    st.success("🟢 Pinecone index ready — scroll down to ask questions.")


# ── Section 4: Search & Ask ───────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Ask Trini a Question")

if not st.session_state.index_ready:
    st.info("Complete the steps above (Load → Chunk → Embed) before asking questions.")
else:
    qcol1, qcol2 = st.columns([3, 1])
    with qcol1:
        user_q = st.text_input(
            "Your question:",
            placeholder="e.g., What are the main findings of this paper?",
            key="q_input",
        )
    with qcol2:
        top_k = st.number_input(
            "Top-K results", min_value=1, max_value=20, value=5, step=1)

    use_reranker = st.checkbox("🔁 Use reranker (cross-encoder)", value=True)
    candidates_n = st.number_input(
        "Candidates for reranker/MMR",
        min_value=top_k, max_value=100, value=max(30, top_k), step=1,
    )
    use_mmr = st.checkbox("🌐 Use MMR diversity filter", value=True)
    lambda_rel = st.slider("MMR relevance weight λ", 0.1, 0.9, 0.7, 0.05)

    if st.button("🔍 Search", use_container_width=True, disabled=not bool(user_q.strip())):
        try:
            with st.spinner("Searching Pinecone..."):
                qvec = embed_query(user_q)
                hits = search(qvec, top_k=int(candidates_n))

            if not hits:
                st.warning("No results found. Try rephrasing your question.")
            else:
                cand_texts = [h["text"] for h in hits]

                # Optional cross-encoder reranking
                if use_reranker and cand_texts:
                    from core.reranker import rerank_scores
                    ce_scores = rerank_scores(user_q, cand_texts)
                    order = list(np.argsort(-ce_scores))
                    hits = [hits[i] for i in order]
                    cand_texts = [cand_texts[i] for i in order]
                    st.caption(
                        f"Reranked {len(hits)} candidates with cross-encoder.")

                # Optional MMR diversity
                if use_mmr and cand_texts:
                    sel_idx = mmr_select(
                        qvec, cand_texts, embed_texts,
                        top_k=int(top_k), lambda_relevance=float(lambda_rel),
                    )
                    hits = [hits[i] for i in sel_idx]
                    st.caption(
                        f"MMR selected {len(hits)} diverse results (λ={lambda_rel:.2f}).")
                else:
                    hits = hits[:int(top_k)]

                st.session_state.search_results = hits
                st.session_state.last_query = user_q
                st.success(
                    f"Found {len(hits)} results. Scroll down for Trini's answer.")

        except Exception as e:
            st.error(f"Search failed: {e}")


# ── Section 5: Results & RAG Answer ──────────────────────────────────────────
hits = st.session_state.search_results

if hits:
    st.divider()
    st.subheader("📋 Retrieved Sources")
    st.caption(f"Results for: *\"{st.session_state.last_query}\"*")

    for rank, h in enumerate(hits, start=1):
        src = h.get("source", "Unknown")
        page = h.get("page")
        full_text = h.get("text", "")
        preview = full_text[:280] + ("..." if len(full_text) > 280 else "")

        st.markdown(
            f"**[{rank}]** `{Path(str(src)).name}` — Page `{page if page is not None else '-'}`"
        )
        st.code(preview)

        with st.expander("View full chunk", expanded=False):
            st.text_area("Full text", full_text,
                         height=220, key=f"chunk_{rank}")
            st.download_button(
                label="⬇️ Download chunk",
                data=full_text,
                file_name=f"chunk_{rank}_page_{page}.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"dl_{rank}",
            )

    # RAG Answer
    st.divider()
    st.subheader("💬 Trini's Answer")
    st.caption(
        "Trini explains the answer simply — as if to a 15-year-old — "
        "with citations [1], [2] pointing to the sources above."
    )

    max_chunks = st.number_input(
        "Max source chunks to include",
        min_value=1, max_value=len(hits), value=min(4, len(hits)),
        key="rag_max_ctx",
    )

    if st.button("🧩 Ask Trini", use_container_width=True, key="rag_btn"):
        from core.rag import generate_answer

        selected = hits[:int(max_chunks)]
        for h in selected:
            if not h.get("text"):
                h["text"] = h.get("preview", "")

        with st.spinner("Trini is thinking..."):
            try:
                answer = generate_answer(
                    query=st.session_state.last_query,
                    hits=selected,
                )
                st.markdown("### 🤖 Answer")
                st.write(answer)
                st.caption(
                    "Citations [1], [2], … refer to the numbered sources listed above.")
            except Exception as e:
                st.error(f"Could not generate answer: {e}")
                st.info(
                    "Ensure Vertex AI is enabled on project `genai-chatsearch` "
                    "and the Cloud Run service account has the **Vertex AI User** role."
                )

elif st.session_state.index_ready:
    st.info("Ask a question above to see results here.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🤖 Trini AI  •  PyMuPDF Block Parser  •  "
    "Token-Aware Chunking  •  Pinecone Vector DB  •  "
    "Cross-Encoder Reranker  •  MMR Diversity  •  Gemini 2.0 Flash"
)
