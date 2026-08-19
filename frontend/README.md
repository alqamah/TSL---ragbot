# Industrial SOP RAG Assistant — Streamlit Frontend

A modern, responsive web user interface for the Industrial SOP RAG Assistant, completely decoupled from the backend RAG pipeline dependencies.

---

## Architecture & Features

- **Decoupled Architecture**: Isolated dependencies via `frontend/requirements.txt` (`streamlit`, `requests`).
- **Live Health & Telemetry**: Dynamic heartbeat check against the FastAPI backend, Qdrant vector database point counts, and active model status.
- **Interactive Document Ingestion**: Direct drag-and-drop file upload (`.xlsx`, `.docx`, `.doc`, `.txt`, `.csv`) and server directory ingestion.
- **Multilingual Support**: English, Hindi, and Hinglish semantic search and grounded synthesis.
- **Rich Source Citations**: Collapsible citation cards with similarity scores, sheet/section metadata, and text previews.
- **Latency Breakdown Chips**: Real-time visualization of query embedding, Qdrant retrieval, and Gemini synthesis timings.

---

## Quick Start

### 1. Ensure Backend is Running
In your root terminal:
```powershell
.venv\Scripts\uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 2. Launch the Frontend

#### Option A: Using the PowerShell Launcher (Automated)
```powershell
powershell -ExecutionPolicy Bypass -File frontend/run.ps1
```

#### Option B: Manual Setup
```powershell
# Create separate virtual environment
python -m venv frontend/.venv_frontend

# Activate environment
frontend\.venv_frontend\Scripts\activate

# Install dependencies
pip install -r frontend/requirements.txt

# Run Streamlit app
streamlit run frontend/app.py --server.port 8501
```

Access the UI at: **`http://localhost:8501`**

---

## Deploying Frontend on Render.com

1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** $\rightarrow$ **Web Service**.
2. Connect your GitHub repository: `https://github.com/alqamah/TSL---ragbot`.
3. Configure the following settings:
   - **Name**: `ragbot-frontend`
   - **Language**: `Python 3`
   - **Branch**: `main`
   - **Region**: Same region as your backend (e.g. Frankfurt / Oregon)
   - **Build Command**: `pip install --upgrade pip && pip install -r frontend/requirements.txt`
   - **Start Command**: `streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.fileWatcherType none --browser.gatherUsageStats false`
   - **Instance Type**: `Free`
4. Add the **Environment Variables**:
   - `PYTHON_VERSION` = `3.11.9`
   - `RAG_BACKEND_URL` = `https://ragbot-backend-q5wj.onrender.com`
5. Click **Create Web Service**.

