### 💻 CLI Reference Commands

Here are the commands to run each component from the project root (`d:\TSL\ragbot`):

#### 1. Backend in **Cloud Mode** (Remote Qdrant)

**PowerShell:**
```powershell
$env:QDRANT_MODE="cloud"
.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
*Or using the entry script with flags:*
```powershell
.venv\Scripts\python.exe src/api/app.py --cloud --port 8000
```

---

#### 2. Backend in **Local Mode** (Embedded Disk Qdrant `./qdrant_db`)

**PowerShell:**
```powershell
$env:QDRANT_MODE="local"
.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
*Or using the entry script with flags:*
```powershell
.venv\Scripts\python.exe src/api/app.py --local --port 8000
```

---

#### 3. Running the **Frontend** (Streamlit UI)

**Option A: Using the Automated PowerShell Script:**
```powershell
powershell -ExecutionPolicy Bypass -File frontend/run.ps1
```

**Option B: Direct Execution via Python Environment:**
```powershell
$env:PYTHONPATH="."
frontend\.venv_frontend\Scripts\streamlit.exe run frontend/app.py --server.port 8501
```

---

#### 4. Running the **Interactive CLI REPL** (Without Web UI)

- **Cloud Mode:**
  ```powershell
  .venv\Scripts\python.exe src/main.py --cloud
  ```
- **Local Mode:**
  ```powershell
  .venv\Scripts\python.exe src/main.py --local
  ```
***************************************************

  To run the backend in **Local DB mode** (embedded Qdrant storage at `./qdrant_db`), you can use any of the following methods:

---

### **Option 1: Using Uvicorn (Recommended for Development)**

Since `QDRANT_MODE=local` is already configured in your [`.env`](file:///d:/TSL/ragbot/.env) file:

```powershell
# From the project root (PowerShell)
.venv\Scripts\uvicorn.exe src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

### **Option 2: Using the Python Application Launcher with `--local` Flag**

You can use the built-in CLI flags inside [`src/api/app.py`](file:///d:/TSL/ragbot/src/api/app.py):

```powershell
# Explicitly forces local embedded mode
.venv\Scripts\python.exe src/api/app.py --local --port 8000
```
*(or `.venv\Scripts\python.exe src/api/app.py --db-mode local`)*

---

### **Option 3: Inline Environment Variable (Explicit Override)**

If you want to explicitly enforce both local mode and UTF-8 encoding in PowerShell:

```powershell
$env:QDRANT_MODE = "local"
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\uvicorn.exe src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

### **How to Verify it's in Local Mode**

Once the server starts, check the console output or the `/api/v1/status` endpoint:

* **Console log output**:
  ```text
  [VectorStore] Initializing local Qdrant storage at './qdrant_db'...
  [API] RAGPipeline initialized (Storage mode: local_path (./qdrant_db)).
  ```
* **Status endpoint**: `http://127.0.0.1:8000/api/v1/status`
  ```json
  {
    "storage_mode": "local_path (./qdrant_db)",
    "status": "online"
  }
  ```