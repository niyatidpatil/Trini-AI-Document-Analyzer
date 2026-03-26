# 🤖 Trini AI — Intelligent Document Analyzer

> An enterprise-grade Retrieval-Augmented Generation (RAG) engine that extracts, summarizes, and retrieves high-relevance insights with citations from complex documents — explained simply, as if you're 15.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io)
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-green)](https://pinecone.io)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-orange)](https://cloud.google.com/vertex-ai)
[![Cloud Run](https://img.shields.io/badge/Deployed-Google%20Cloud%20Run-blue)](https://cloud.google.com/run)

---

## 📌 About This Project

Trini AI solves the **"Black Box" problem** in AI document analysis. Most LLM-powered tools hallucinate facts or fail to cite their sources — making them unreliable for research and decision-making.

Trini AI is built around a **High-Precision RAG pipeline** that retrieves only verified, cited information directly from your uploaded documents. Every answer is grounded in your source material, with bracketed citations pointing back to the exact page it came from.

---

## ✨ Features

- **Block-Level PDF Parsing** — Custom PyMuPDF parser that preserves document structure (tables, columns, headers) that breaks standard readers
- **Token-Aware Chunking** — Uses `tiktoken` to split documents at exact token boundaries with configurable overlap, preserving context across chunk edges
- **Cloud-Native Vector Storage** — Pinecone vector database for persistent, scalable semantic memory
- **Two-Stage Retrieval** — Pinecone semantic search followed by cross-encoder reranking for precision
- **MMR Diversity Filter** — Maximal Marginal Relevance ensures retrieved results are both relevant and diverse
- **Gemini 2.0 Flash** — Fast, grounded answer generation via Google Vertex AI
- **Citation Engine** — Every answer includes bracketed citations [1], [2] mapping to source pages — no hallucinations
- **ELI15 Explanations** — Answers are written simply, as if explaining to a 15-year-old
- **OCR Fallback** — Tesseract OCR handles scanned or image-based PDFs
- **Multi-Format Support** — PDF, DOCX, PPTX, XLSX, TXT

---

## 🏗️ Architecture

```
Document Upload
      ↓
PyMuPDF Block-Level Parser  (+ OCR fallback for scanned PDFs)
      ↓
Token-Aware Chunker  (tiktoken cl100k_base, 750 tokens, 150 overlap)
      ↓
SentenceTransformer Embeddings  (all-MiniLM-L6-v2, 384 dims)
      ↓
Pinecone Vector DB  (cloud-native, persistent)
      ↓
Semantic Search  →  Cross-Encoder Reranker  →  MMR Diversity Filter
      ↓
Gemini 2.0 Flash (Vertex AI)
      ↓
Cited Answer  (ELI15 style with [1][2] citations)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Frontend | Streamlit |
| LLM Engine | Google Vertex AI — Gemini 2.0 Flash |
| Vector Database | Pinecone (cloud-native, persistent, 384-dim cosine) |
| Embeddings | SentenceTransformers — all-MiniLM-L6-v2 |
| Reranker | Cross-Encoder — ms-marco-MiniLM-L-6-v2 |
| PDF Parser | PyMuPDF (block-level) + Tesseract OCR |
| Chunking | tiktoken (token-aware, cl100k_base) |
| Deployment | Google Cloud Run (serverless, auto-scaling) |
| Containerisation | Docker |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Google Cloud Platform account with Vertex AI enabled
- Pinecone account and API key
- Docker (for local container testing)
- `gcloud` CLI installed

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/niyatidpatil/Trini-AI-Document-Analyzer.git
   cd Trini-AI-Document-Analyzer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Open .env and fill in your actual values
   ```

5. **Authenticate with Google Cloud (local dev)**
   ```bash
   gcloud auth application-default login
   ```

6. **Run locally**
   ```bash
   streamlit run app.py
   ```

### Environment Variables

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | Your Pinecone index name (`trini-ai`) |
| `GCP_PROJECT` | Your GCP project ID |
| `GCP_REGION` | GCP region (`us-east4`) |
| `GEMINI_MODEL` | Gemini model name (`gemini-2.0-flash`) |

---

## ☁️ Deployment (Google Cloud Run)

```bash
gcloud run deploy trini-ai \
  --source . \
  --region us-east4 \
  --project genai-chatsearch \
  --allow-unauthenticated
```

---

## 📁 Project Structure

```
Trini-AI-Document-Analyzer/
├── core/
│   ├── __init__.py
│   ├── embeddings.py     # Pinecone upsert, MMR (NumPy), SentenceTransformer
│   ├── loader.py         # PyMuPDF block parser, OCR, multi-format ingestion
│   ├── rag.py            # Gemini 2.0 Flash via Vertex AI, citation prompting
│   ├── reranker.py       # Cross-encoder reranking
│   └── splitter.py       # Token-aware chunking (tiktoken)
├── .streamlit/
├── .env.example
├── .gitignore
├── app.py                # Streamlit UI
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 🔒 Security

- API keys and secrets are stored as environment variables only
- `.env` is excluded from version control via `.gitignore`
- Production secrets are managed via Google Cloud Run environment variables

---

## 🌐 Live Demo

[https://trini-ai-330544738472.us-east4.run.app](https://trini-ai-330544738472.us-east4.run.app)