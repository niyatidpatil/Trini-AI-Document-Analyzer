# ── Trini AI — Dockerfile ─────────────────────────────────────────────────────
# Base image: Python 3.11 slim (small footprint, Cloud Run optimised)
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tesseract-ocr  : OCR fallback for scanned PDFs
# libgl1         : required by PyMuPDF (fitz) for PDF rendering
# libglib2.0-0   : required by PyMuPDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# If requirements.txt doesn't change, this layer is reused on every build
# (much faster rebuilds).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py .
COPY core/ ./core/

# Copy Streamlit config if it exists
COPY .streamlit/ ./.streamlit/

# ── Runtime directories ───────────────────────────────────────────────────────
# Create the folders the app writes to at runtime.
# On Cloud Run these are ephemeral (reset on each instance restart) — that is
# fine because Pinecone holds the persistent vectors.
RUN mkdir -p .store .tmp_uploads

# ── Port ──────────────────────────────────────────────────────────────────────
# Cloud Run injects $PORT at runtime (default 8080). Streamlit reads it below.
EXPOSE 8080

# ── Start command ─────────────────────────────────────────────────────────────
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]