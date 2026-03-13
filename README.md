# Trini AI: Intelligent Document Analyzer

## About This Project
Trini AI is a robust Retrieval-Augmented Generation (RAG) engine designed to transform how users interact with closed-knowledge bases. It allows users to upload complex documents and instantly extract precise, context-aware answers. By focusing on highly relevant and diverse information retrieval, it significantly reduces the time spent searching through dense reports and documentation, streamlining research and decision-making workflows.

## Features
* **Intelligent Document Processing:** Seamless ingestion and parsing of complex document formats (PDF, DOCX).
* **Context-Aware Retrieval:** Implements Maximal Marginal Relevance (MMR) to ensure search results are both highly accurate and diverse, preventing repetitive answers.
* **Advanced Reranking:** Dynamically re-evaluates and scores retrieved chunks to surface the most critical insights first.
* **Interactive Workspace:** Features a clean, tabbed interface allowing users to chat with their data, save vital notes, and access a comprehensive user guide.

## Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **LLM Engine:** Google Vertex AI (Gemini 2.0 Flash)
* **Vector Infrastructure:** Pinecone, FAISS
* **Optimization:** Token-aware chunking, Reranker

## Getting Started

### Prerequisites
* Python 3.9+
* Google Cloud Platform (GCP) Account with Vertex AI enabled
* Pinecone Account and API Key

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/niyatidpatil/Trini-AI-Document-Analyzer.git
   cd Trini-AI-Document-Analyzer
