import os
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.api.app import app


def test_api_endpoints():
    print("=" * 60)
    print("TESTING FASTAPI ENDPOINTS WITH TESTCLIENT")
    print("=" * 60)

    with TestClient(app) as client:
        # 1. Test Root
        response = client.get("/")
        print(f"GET / -> Status {response.status_code}: {response.json()}")
        assert response.status_code == 200
        assert response.json()["service"] == "Industrial SOP RAG Assistant API"

        # 2. Test Health
        response = client.get("/health")
        print(f"GET /health -> Status {response.status_code}: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # 3. Test Status
        response = client.get("/api/v1/status")
        print(f"GET /api/v1/status -> Status {response.status_code}: {response.json()}")
        assert response.status_code == 200
        assert response.json()["collection_name"] == "industrial_sops_gemini"

        # 4. Test Ingest File
        sample_file = os.path.join(PROJECT_ROOT, "io", "SOP", "01_test.XLSX")
        if os.path.exists(sample_file):
            response = client.post("/api/v1/ingest/file", json={"file_path": sample_file})
            print(f"POST /api/v1/ingest/file -> Status {response.status_code}: {response.json()}")
            assert response.status_code == 200
            assert response.json()["status"] == "success"

        # 5. Test Query
        query_payload = {
            "query": "कंप्रेसर को स्टार्ट कैसे करें?",
            "top_k": 2,
            "show_sources": True,
        }
        response = client.post("/api/v1/query", json=query_payload)
        print(f"POST /api/v1/query -> Status {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        print(f"Answer snippet: {data['answer'][:120]}...")
        print(f"Total latency: {data['metrics']['total_e2e_latency_ms']} ms")
        assert "answer" in data
        assert len(data["sources"]) > 0

    print("\nALL API TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    test_api_endpoints()
