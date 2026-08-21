import os
import requests
from typing import Any, Dict, Optional, Tuple


class RAGAPIClient:
    """HTTP Client for interacting with the Industrial SOP RAG Assistant FastAPI backend."""

    DEFAULT_BASE_URL = "http://127.0.0.1:8000"


    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        url = base_url or os.getenv("RAG_BACKEND_URL") or self.DEFAULT_BASE_URL
        self.base_url = str(url).rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if the backend API service is reachable and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            return False, {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def get_status(self) -> Tuple[bool, Dict[str, Any]]:
        """Fetch system status, active models, and indexed points telemetry."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/status", timeout=12)
            if resp.status_code == 200:
                return True, resp.json()
            return False, {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def get_logs(self, since: int = 0) -> Tuple[bool, Dict[str, Any]]:
        """Stream backend terminal output newer than the given cursor."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/logs",
                params={"since": since},
                timeout=10,
            )
            if resp.status_code == 200:
                return True, resp.json()
            return False, {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def list_indexed_files(self) -> Tuple[bool, Dict[str, Any]]:
        """Fetch the distinct source files stored in the vector database."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/files", timeout=12)
            if resp.status_code == 200:
                return True, resp.json()
            return False, {"error": f"HTTP {resp.status_code}: {resp.text}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": str(e)}

    def reset_database(self) -> Tuple[bool, Dict[str, Any]]:
        """Reset and wipe the vector database collection."""
        try:
            resp = requests.post(f"{self.base_url}/api/v1/reset", timeout=self.timeout)
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Reset failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Reset request failed: {str(e)}"}

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
        except requests.exceptions.Timeout:
            return False, {
                "error": "The backend request timed out (>120s). Render is likely still building and deploying your latest push or waking up from sleep. Please wait a minute and retry."
            }
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Connection failed: {str(e)}"}

    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_table_rows: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Upload a document file to the backend for indexing with custom chunking parameters."""
        try:
            files = {"file": (filename, file_bytes)}
            data = {}
            if chunk_size is not None:
                data["chunk_size"] = str(chunk_size)
            if chunk_overlap is not None:
                data["chunk_overlap"] = str(chunk_overlap)
            if max_table_rows is not None:
                data["max_table_rows"] = str(max_table_rows)

            resp = requests.post(
                f"{self.base_url}/api/v1/ingest/upload",
                files=files,
                data=data if data else None,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Upload failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Upload error: {str(e)}"}

    def ingest_by_path(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_table_rows: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Ingest a single document specified by local path on the server."""
        try:
            payload: Dict[str, Any] = {"file_path": file_path}
            if chunk_size is not None:
                payload["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                payload["chunk_overlap"] = chunk_overlap
            if max_table_rows is not None:
                payload["max_table_rows"] = max_table_rows

            resp = requests.post(
                f"{self.base_url}/api/v1/ingest/file",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Ingestion failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Request error: {str(e)}"}

    def ingest_directory(
        self,
        directory_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_table_rows: Optional[int] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Batch ingest an entire directory of documents."""
        try:
            payload: Dict[str, Any] = {"directory_path": directory_path}
            if chunk_size is not None:
                payload["chunk_size"] = chunk_size
            if chunk_overlap is not None:
                payload["chunk_overlap"] = chunk_overlap
            if max_table_rows is not None:
                payload["max_table_rows"] = max_table_rows

            resp = requests.post(
                f"{self.base_url}/api/v1/ingest/directory",
                json=payload,
                timeout=self.timeout * 2,
            )
            if resp.status_code == 200:
                return True, resp.json()
            detail = resp.json().get("detail", resp.text) if "json" in resp.headers.get("content-type", "") else resp.text
            return False, {"error": f"Directory ingestion failed ({resp.status_code}): {detail}"}
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Request error: {str(e)}"}
