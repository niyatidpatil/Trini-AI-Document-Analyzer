import streamlit as st
import os
import tempfile
import logging
import sys
import fitz  # PyMuPDF
import json
from dotenv import load_dotenv
import google.auth 
from google.auth.transport.requests import Request
from fpdf import FPDF

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage, MessageRole
from pinecone import Pinecone

# --- 0. CONFIGURATION ---
st.set_page_config(page_title="Trini AI", page_icon="💠", layout="wide")
load_dotenv()

# --- CSS: SCROLLABLE CHAT ---
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 18px !important; }
    .stApp { background-color: #0a0a0a; color: #e5e5e5; }
    
    /* Improved Chat Bubble */
    .stChatMessage { 
        background-color: #171717; 
        border: 1px solid #262626; 
        border-radius: 12px; 
        padding: 20px; 
        margin-bottom: 10px;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #262626; }
    
    /* File List Badge */
    .file-badge {
        background-color: #0f766e;
        color: #ccfbf1;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-bottom: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. AUTHENTICATION ---
credentials = None
project_id = None
try:
    credentials, project_id = google.auth.default()
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
except Exception:
    pass

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or project_id
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "trini-ai" 

if not PINECONE_API_KEY:
    st.error("CRITICAL: PINECONE_API_KEY not found.")
    st.stop()

# --- 2. STATE MANAGEMENT ---
if "saved_notes" not in st.session_state: st.session_state.saved_notes = []
if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Hello, I'm Trini. Create a Workspace ID in the sidebar to start fresh."}]
if "current_sources" not in st.session_state: st.session_state.current_sources = [] 
if "index" not in st.session_state: st.session_state.index = None
if "user_id" not in st.session_state: st.session_state.user_id = "guest_session"
if "file_list" not in st.session_state: st.session_state.file_list = [] # Track uploaded files

# --- 3. SETUP MODELS ---
SYSTEM_PROMPT = """
You are Trini, a precise research assistant.
1. Answer using ONLY the provided Context from the current Workspace.
2. If the user asks about a document not in the context, say "That document is not in this Workspace."
3. Maintain conversation history.
"""

def load_models():
    llm = Vertex(
        model="gemini-2.0-flash-001", 
        temperature=0.1, # Low temp for precision
        max_tokens=8192,
        context_window=1000000,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials, 
        system_prompt=SYSTEM_PROMPT
    )
    # Local Embeddings (Stable, Free)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

def get_vector_store(namespace: str):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)

def wipe_memory():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        # Delete only current namespace
        index.delete(delete_all=True, namespace=st.session_state.user_id)
        
        # Reset State
        st.session_state.index = None
        st.session_state.file_list = []
        st.session_state.messages = [{"role": "assistant", "content": f"Workspace '{st.session_state.user_id}' erased. Clean slate."}]
        st.toast("Workspace Wiped!", icon="🧹")
        st.rerun()
    except Exception as e:
        st.error(f"Wipe Failed: {e}")

def ingest_new_files(uploaded_files):
    all_documents = []
    new_filenames = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # 32MB Check (Cloud Run Limit)
            if uploaded_file.size > 32 * 1024 * 1024:
                raise MemoryError(f"{uploaded_file.name} is > 32MB. Please compress it.")
            
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            new_filenames.append(uploaded_file.name)
            
            if uploaded_file.name.lower().endswith(".pdf"):
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    text = page.get_text("blocks")
                    page_text = "\n".join([b[4] for b in text])
                    if len(page_text) > 10:
                        all_documents.append(Document(text=page_text, metadata={"filename": uploaded_file.name, "page_number": page_num + 1}))
            else:
                from llama_index.readers.file import DocxReader
                if uploaded_file.name.lower().endswith(".docx"):
                    loader = DocxReader()
                    docs = loader.load_data(file=file_path)
                    all_documents.extend(docs)

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    vector_store = get_vector_store(namespace=st.session_state.user_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(all_documents, storage_context=storage_context, transformations=[splitter], show_progress=True)
    
    # Update File List
    st.session_state.file_list.extend(new_filenames)
    return index

def load_existing_index():
    try:
        return VectorStoreIndex.from_vector_store(vector_store=get_vector_store(namespace=st.session_state.user_id))
    except: return None

def get_chat_history():
    return [ChatMessage(role=(MessageRole.USER if m['role']=='user' else MessageRole.ASSISTANT), content=m['content']) for m in st.session_state.messages]

# Init Models
load_models() 
if st.session_state.index is None: st.session_state.index = load_existing_index()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("💠 Trini Menu")
    
    # WORKSPACE MANAGER
    st.markdown("### 📂 Workspace")
    workspace_id = st.text_input("Workspace ID:", value=st.session_state.user_id, help="Change this name to switch 'Brains'.")
    
    if workspace_id != st.session_state.user_id:
        st.session_state.user_id = workspace_id
        st.session_state.index = None # Force reload
        st.session_state.file_list = [] # Clear file list for new space
        st.session_state.messages = [{"role": "assistant", "content": f"Switched to workspace: **{workspace_id}**"}]
        st.rerun()
        
    # DATA CONTROLS
    st.markdown("---")
    st.markdown("### 🧠 Brain Contents")
    if st.session_state.file_list:
        for f in st.session_state.file_list:
            st.markdown(f"<div class='file-badge'>📄 {f}</div>", unsafe_allow_html=True)
    else:
        st.caption("No files tracked in this session.")

    uploaded_files = st.file_uploader("Add Files:", type=["pdf", "docx"], accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Ingest"):
            if uploaded_files:
                with st.spinner("Learning..."):
                    try:
                        st.session_state.index = ingest_new_files(uploaded_files)
                        st.success("Done!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
    with col2:
        if st.button("🧹 Wipe Brain", help="Delete ALL data in this Workspace"):
            wipe_memory()
            
    st.markdown("---")
    if st.button("🗑️ Clear Chat Only", help="Keep data, clear history"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat cleared. Memory retained."}]
        st.rerun()

# --- 5. CHAT AREA ---
st.title("💠 Trini AI")

# Show Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Trini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.index is None:
            st.warning(f"Workspace '{st.session_state.user_id}' is empty. Upload a file.")
        else:
            with st.spinner("Thinking..."):
                try:
                    chat_engine = st.session_state.index.as_chat_engine(
                        chat_mode="context", 
                        llm=Settings.llm, 
                        similarity_top_k=30,
                        system_prompt=SYSTEM_PROMPT
                    )
                    response = chat_engine.chat(prompt, chat_history=get_chat_history())
                    st.markdown(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                    
                    # Sources
                    with st.expander("📚 Sources"):
                        for node in response.source_nodes[:5]:
                            fname = node.node.metadata.get("filename", "Doc")
                            page = node.node.metadata.get("page_number", "?")
                            st.caption(f"📄 {fname} (Page {page})")
                            st.info(node.node.get_content()[:200] + "...")
                except Exception as e:
                    st.error(f"Error: {str(e)}")