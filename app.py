# app.py — Trini AI  |  Production Chat Interface
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import streamlit as st
import numpy as np

from core.loader import load_pdfs, load_docx, load_txts
from core.splitter import chunk_docs
from core.embeddings import (
    embed_texts, embed_query, upsert_chunks,
    search, fingerprint_chunks, save_config, mmr_select,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trini AI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #09090f !important;
    color: #e2e2ee !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
}
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container {
    padding: 2rem 1.5rem 6rem !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0e0e1c !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2.5rem 1.75rem 2rem !important;
    max-width: 100% !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
    gap: 12px !important;
}

/* Avatar sizing */
[data-testid="stChatMessage"] img,
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    font-size: 18px !important;
    border-radius: 50% !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #14141f;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    border-top-left-radius: 6px;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    line-height: 1.75;
    color: #d8d8e8;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: linear-gradient(135deg, #6f5fff, #9b55ff);
    border-radius: 20px;
    border-top-right-radius: 6px;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    line-height: 1.75;
    color: #fff;
    box-shadow: 0 6px 24px rgba(111,95,255,0.25);
}

/* ── Chat input area ── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(760px, 100vw) !important;
    padding: 1rem 1.5rem 1.5rem !important;
    background: linear-gradient(to top, #09090f 70%, transparent) !important;
    z-index: 999 !important;
}
[data-testid="stChatInput"] > div {
    background: #14141f !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 18px !important;
    padding: 4px 8px !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: rgba(111,95,255,0.6) !important;
    box-shadow: 0 4px 30px rgba(111,95,255,0.15), 0 0 0 3px rgba(111,95,255,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    color: #e2e2ee !important;
    line-height: 1.5 !important;
    padding: 0.5rem 0.25rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #444 !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #6f5fff, #9b55ff) !important;
    border-radius: 10px !important;
    width: 34px !important; height: 34px !important;
    border: none !important;
    transition: opacity 0.15s !important;
}
[data-testid="stChatInput"] button:hover { opacity: 0.85 !important; }
[data-testid="stChatInput"] button:disabled { opacity: 0.3 !important; }

/* ── Upload strip (above chat input) ── */
.upload-strip {
    position: fixed;
    bottom: 90px;
    left: 50%;
    transform: translateX(-50%);
    width: min(760px, 100vw);
    padding: 0 1.5rem;
    z-index: 998;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, #6f5fff, #9b55ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.25rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
.stButton button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton button:disabled { opacity: 0.3 !important; transform: none !important; }
.stButton button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #888 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(111,95,255,0.05) !important;
    border: 1.5px dashed rgba(111,95,255,0.35) !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] > div { padding: 1rem !important; }

/* ── Expander (sources) ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    margin-top: -0.25rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #555 !important;
    padding: 0.5rem 0.75rem !important;
}
[data-testid="stExpander"] summary:hover { color: #888 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
}

/* ── Progress ── */
[data-testid="stSpinner"] > div { border-top-color: #6f5fff !important; }

/* ── Sidebar text ── */
[data-testid="stSidebar"] p { font-size: 0.9rem !important; color: #777 !important; }
[data-testid="stSidebar"] .stButton button { font-size: 0.9rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 5px; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages":    [],
    "index_ready": False,
    "doc_name":    None,
    "fingerprint": None,
    "show_upload": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

TMP_DIR = Path(".tmp_uploads")
TMP_DIR.mkdir(exist_ok=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
       color:#fff;letter-spacing:-0.02em;margin:0 0 4px'>
       🤖 Trini AI
    </p>
    <p style='font-size:0.82rem;color:#555;margin:0 0 1.5rem'>
       Intelligent Document Analyzer
    </p>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Pages**")
    st.page_link("app.py",        label="💬 Chat",            icon=None)
    st.page_link("pages/debug.py", label="🛠️ Pipeline Console", icon=None)

    st.divider()

    if st.session_state.index_ready:
        st.markdown(f"""
        <div style='padding:12px 14px;background:rgba(52,211,153,0.08);
             border:1px solid rgba(52,211,153,0.18);border-radius:12px;
             margin-bottom:1rem'>
            <p style='font-size:0.72rem;color:#34d399;font-weight:600;
               text-transform:uppercase;letter-spacing:.05em;margin:0'>
               🟢 Active document
            </p>
            <p style='font-size:0.85rem;color:#ccc;margin:4px 0 0;
               overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>
               {st.session_state.doc_name}
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            st.session_state.update({
                "messages": [], "index_ready": False,
                "doc_name": None, "show_upload": False,
            })
            st.rerun()

    st.divider()
    st.markdown("""
    <p style='font-size:0.75rem;color:#333;margin:0'>
        Built with PyMuPDF · Pinecone · Gemini 2.0 Flash
    </p>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:14px;margin-bottom:1.5rem'>
    <div style='width:42px;height:42px;flex-shrink:0;
         background:linear-gradient(135deg,#6f5fff,#ff5fa0);
         border-radius:13px;display:flex;align-items:center;
         justify-content:center;font-size:20px;
         box-shadow:0 4px 16px rgba(111,95,255,0.3)'>
        🤖
    </div>
    <div>
        <p style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;
           color:#fff;letter-spacing:-0.03em;margin:0'>Trini AI</p>
        <p style='font-size:0.8rem;color:#555;margin:0'>
           Intelligent Document Analyzer</p>
    </div>
    <div style='margin-left:auto;font-size:0.7rem;padding:4px 12px;
         border-radius:20px;background:rgba(111,95,255,0.12);
         color:#9d8fff;border:1px solid rgba(111,95,255,0.22);
         font-weight:500'>
        Gemini 2.0 Flash
    </div>
</div>
<div style='height:1px;background:rgba(255,255,255,0.07);margin-bottom:1.5rem'></div>
""", unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages and not st.session_state.index_ready:
    st.markdown("""
    <div style='text-align:center;padding:4rem 1rem 2rem'>
        <div style='font-size:3.5rem;margin-bottom:1rem'>📄</div>
        <p style='font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;
           color:#fff;letter-spacing:-0.04em;margin:0 0 0.75rem'>
           Drop a doc, ask anything.
        </p>
        <p style='font-size:0.95rem;color:#444;line-height:1.7;
           max-width:380px;margin:0 auto 1.5rem'>
           Use the <strong style='color:#666'>+</strong> button next to the
           chat box to upload a PDF or DOCX.<br>
           Every answer comes with page citations.
        </p>
        <div style='display:flex;flex-wrap:wrap;gap:8px;justify-content:center'>
            <span style='background:rgba(255,255,255,0.05);
                  border:1px solid rgba(255,255,255,0.09);border-radius:20px;
                  padding:8px 16px;font-size:0.85rem;color:#555'>
                Summarize this paper
            </span>
            <span style='background:rgba(255,255,255,0.05);
                  border:1px solid rgba(255,255,255,0.09);border-radius:20px;
                  padding:8px 16px;font-size:0.85rem;color:#555'>
                What are the key findings?
            </span>
            <span style='background:rgba(255,255,255,0.05);
                  border:1px solid rgba(255,255,255,0.09);border-radius:20px;
                  padding:8px 16px;font-size:0.85rem;color:#555'>
                Explain like I'm 15
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.messages and st.session_state.index_ready:
    st.markdown(f"""
    <div style='text-align:center;padding:3rem 1rem 2rem'>
        <div style='font-size:2.5rem;margin-bottom:0.75rem'>✨</div>
        <p style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;
           color:#fff;letter-spacing:-0.02em;margin:0 0 0.4rem'>
           Ready. Ask anything.
        </p>
        <p style='font-size:0.85rem;color:#444;margin:0'>{st.session_state.doc_name}</p>
    </div>
    """, unsafe_allow_html=True)


# ── Conversation history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"],
                         avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} cited source(s)"):
                for i, s in enumerate(msg["sources"], 1):
                    fname = Path(str(s.get("source", ""))).name or "Document"
                    page = s.get("page")
                    page_str = f"page {int(page) + 1}" if page is not None else "—"
                    text = str(s.get("text", ""))[:300]
                    st.markdown(
                        f"**[{i}]** `{fname}` — {page_str}\n\n"
                        f"<span style='font-size:0.82rem;color:#555;line-height:1.6'>"
                        f"{text}…</span>",
                        unsafe_allow_html=True,
                    )


# ── Upload panel (shown when + is clicked) ────────────────────────────────────
if st.session_state.show_upload:
    with st.container():
        st.markdown("""
        <div style='background:#14141f;border:1px solid rgba(111,95,255,0.3);
             border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:1rem'>
            <p style='font-size:0.9rem;font-weight:500;color:#aaa;margin:0 0 0.75rem'>
                📎 Upload a document
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload document",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key="chat_uploader",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            if uploaded_file:
                analyze = st.button("⚡ Analyze & Start Chat",
                                    use_container_width=True)
                if analyze:
                    p = TMP_DIR / uploaded_file.name
                    p.write_bytes(uploaded_file.getbuffer())

                    with st.spinner("Reading document..."):
                        ext = p.suffix.lower()
                        docs = (load_pdfs([p]) if ext == ".pdf"
                                else load_docx([p]) if ext == ".docx"
                                else load_txts([p]))

                    with st.spinner(f"Chunking {len(docs)} page(s)..."):
                        chunks = chunk_docs(
                            docs, chunk_size=750, chunk_overlap=150)

                    with st.spinner(f"Uploading {len(chunks)} vectors to Pinecone..."):
                        try:
                            upsert_chunks(chunks)
                            fp = fingerprint_chunks(chunks)
                            save_config({"fingerprint": fp,
                                         "model": "all-MiniLM-L6-v2", "dim": 384})
                            st.session_state.update({
                                "index_ready": True,
                                "doc_name":    uploaded_file.name,
                                "fingerprint": fp,
                                "messages":    [],
                                "show_upload": False,
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
        with col2:
            if st.button("✕ Cancel", use_container_width=True):
                st.session_state.show_upload = False
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ── Chat input row — + button + text input ────────────────────────────────────
input_col, btn_col = st.columns([1, 12])

with input_col:
    plus_label = "✕" if st.session_state.show_upload else "＋"
    if st.button(plus_label, key="plus_btn",
                 help="Upload a document" if not st.session_state.show_upload
                 else "Close upload panel"):
        st.session_state.show_upload = not st.session_state.show_upload
        st.rerun()

with btn_col:
    user_input = st.chat_input(
        "Ask anything about your document…"
        if st.session_state.index_ready
        else "Upload a document first (use the + button) →",
        disabled=not st.session_state.index_ready,
    )


# ── Handle response ───────────────────────────────────────────────────────────
if user_input and st.session_state.index_ready:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Trini is thinking…"):
            try:
                qvec = embed_query(user_input)
                hits = search(qvec, top_k=30)

                if not hits:
                    answer = ("I couldn't find relevant content for that question. "
                              "Try rephrasing your question.")
                    sources = []
                else:
                    texts = [h["text"] for h in hits]

                    # Rerank with cross-encoder
                    from core.reranker import rerank_scores
                    scores = rerank_scores(user_input, texts)
                    hits = [hits[i] for i in np.argsort(-scores)]
                    texts = [hits[i]["text"] for i in range(len(hits))]

                    # MMR diversity filter
                    sel = mmr_select(qvec, texts, embed_texts,
                                     top_k=5, lambda_relevance=0.7)
                    final = [hits[i] for i in sel]

                    # Gemini answer
                    from core.rag import generate_answer
                    answer = generate_answer(user_input, final)
                    sources = final

            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander(f"📎 {len(sources)} cited source(s)"):
                for i, s in enumerate(sources, 1):
                    fname = Path(str(s.get("source", ""))).name or "Document"
                    page = s.get("page")
                    page_str = f"page {int(page) + 1}" if page is not None else "—"
                    text = str(s.get("text", ""))[:300]
                    st.markdown(
                        f"**[{i}]** `{fname}` — {page_str}\n\n"
                        f"<span style='font-size:0.82rem;color:#555;line-height:1.6'>"
                        f"{text}…</span>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
    })
