import requests
from typing import Any, Dict, Optional, Tuple


class RAGAPIClient:
    """HTTP Client for interacting with the Industrial SOP RAG Assistant FastAPI backend."""

    def __init__(self, base_url: str = "https://ragbot-backend-q5wj.onrender.com", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if the backend API service is reachable and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return True, resp.json()
            return False, {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def get_status(self) -> Tuple[bool, Dict[str, Any]]:
        """Fetch system status, active models, and indexed points telemetry."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/status", timeout=8)
            if resp.status_code == 200:
                return True, resp.json()
            return False, {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def query(
        self, query_text: str, top_k: int = 3, show_sources: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """Execute a semantic search and RAG synthesis query."""
        payload = {
            "query": query_text,
            "top_k": top_k,
            "show_sources": show_sources,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/query",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            return False, {"error": f"Error ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Connection failed: {str(e)}"}

    def upload_file(self, file_bytes: bytes, filename: str) -> Tuple[bool, Dict[str, Any]]:
        """Upload a document file to the backend for indexing."""
        try:
            files = {"file": (filename, file_bytes)}
            resp = requests.post(
                f"{self.base_url}/api/v1/ingest/upload",
                files=files,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Upload failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Upload error: {str(e)}"}

    def ingest_by_path(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Ingest a single document specified by local path on the server."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/ingest/file",
                json={"file_path": file_path},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Ingestion failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Request error: {str(e)}"}

    def ingest_directory(self, directory_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Batch ingest an entire directory of documents."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/ingest/directory",
                json={"directory_path": directory_path},
                timeout=self.timeout * 2,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Directory ingestion failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Request error: {str(e)}"}
