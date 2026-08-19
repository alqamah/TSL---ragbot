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