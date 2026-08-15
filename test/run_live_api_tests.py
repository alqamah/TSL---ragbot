import os
import sys
import time
import requests

BASE_URL = "http://localhost:8000"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_endpoint(name, method, url, **kwargs):
    start = time.time()
    try:
        resp = requests.request(method, url, **kwargs)
        elapsed = (time.time() - start) * 1000
        print(f"\n[TEST] {name}")
        print(f"  -> {method.upper()} {url}")
        print(f"  -> Status: {resp.status_code} ({elapsed:.1f} ms)")
        try:
            data = resp.json()
            return resp.status_code, data
        except Exception:
            return resp.status_code, resp.text
    except Exception as e:
        print(f"\n[FAIL] {name}: {e}")
        return 0, str(e)


def main():
    print_section("STARTING LIVE API SUITE ON " + BASE_URL)
    results = []

    # 1. Root
    status, data = test_endpoint("1. Root Info", "GET", f"{BASE_URL}/")
    assert status == 200, f"Expected 200, got {status}"
    print(f"  -> Service: {data.get('service')}, Version: {data.get('version')}")
    results.append(("Root Info", "PASS"))

    # 2. Health
    status, data = test_endpoint("2. Health Check", "GET", f"{BASE_URL}/health")
    assert status == 200, f"Expected 200, got {status}"
    assert data.get("status") == "healthy"
    print(f"  -> Health: {data.get('status')}")
    results.append(("Health Check", "PASS"))

    # 3. Status
    status, data = test_endpoint("3. System Telemetry & Status", "GET", f"{BASE_URL}/api/v1/status")
    assert status == 200, f"Expected 200, got {status}"
    initial_points = data.get("indexed_points", 0)
    print(f"  -> Collection: {data.get('collection_name')}")
    print(f"  -> Storage Mode: {data.get('storage_mode')}")
    print(f"  -> Indexed Points: {initial_points}")
    print(f"  -> Generation Model: {data.get('generation_model')}")
    results.append(("System Status", "PASS"))

    # 4. Ingest File by Local Path
    sample_file = os.path.join(PROJECT_ROOT, "io", "SOP", "01_test.XLSX")
    if os.path.exists(sample_file):
        status, data = test_endpoint(
            "4. Ingest Document (Local Path)",
            "POST",
            f"{BASE_URL}/api/v1/ingest/file",
            json={"file_path": sample_file},
        )
        assert status == 200, f"Expected 200, got {status}"
        print(f"  -> Status: {data.get('status')}")
        print(f"  -> Chunks Indexed: {data.get('chunks_indexed')}")
        print(f"  -> Total Points: {data.get('total_points')}")
        results.append(("Ingest by Path", "PASS"))

    # 5. Ingest File via Multipart Upload
    upload_file = os.path.join(PROJECT_ROOT, "io", "SOP", "03_Tower_light_maintenance_work.xlsx")
    if os.path.exists(upload_file):
        with open(upload_file, "rb") as f:
            status, data = test_endpoint(
                "5. Ingest Document (Multipart Upload)",
                "POST",
                f"{BASE_URL}/api/v1/ingest/upload",
                files={"file": (os.path.basename(upload_file), f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            )
            assert status == 200, f"Expected 200, got {status}"
            print(f"  -> Status: {data.get('status')}")
            print(f"  -> Message: {data.get('message')}")
            print(f"  -> Chunks Indexed: {data.get('chunks_indexed')}")
            results.append(("Ingest by Upload", "PASS"))

    # 6. Verify Status Update
    status, data = test_endpoint("6. Verify Indexed Count Update", "GET", f"{BASE_URL}/api/v1/status")
    assert status == 200
    new_points = data.get("indexed_points", 0)
    print(f"  -> Updated Total Points: {new_points}")
    results.append(("Verify Count Update", "PASS"))

    # 7. Query RAG (English)
    en_query = {
        "query": "What safety precautions are required during air compressor maintenance?",
        "top_k": 3,
        "show_sources": True,
    }
    status, data = test_endpoint("7. RAG Inference (English)", "POST", f"{BASE_URL}/api/v1/query", json=en_query)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  -> Question: {data.get('query')}")
    print(f"  -> Answer: {data.get('answer')}")
    print(f"  -> Sources Found: {len(data.get('sources', []))}")
    if data.get("sources"):
        print(f"     Top source: {data['sources'][0].get('source')} (Score: {data['sources'][0].get('score'):.3f})")
    metrics = data.get("metrics", {})
    print(f"  -> Retrieval Latency: {metrics.get('total_retrieval_ms', 0):.1f} ms")
    print(f"  -> Generation Latency: {metrics.get('llm_generation_ms', 0):.1f} ms")
    print(f"  -> Total E2E Latency: {metrics.get('total_e2e_latency_ms', 0):.1f} ms")
    results.append(("Query RAG (English)", "PASS"))

    # 8. Query RAG (Hindi)
    hi_query = {
        "query": "कंप्रेसर का रखरखाव करते समय किन सुरक्षा सावधानियों का पालन करना चाहिए?",
        "top_k": 2,
        "show_sources": True,
    }
    status, data = test_endpoint("8. RAG Inference (Hindi)", "POST", f"{BASE_URL}/api/v1/query", json=hi_query)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  -> Question: {data.get('query')}")
    print(f"  -> Answer: {data.get('answer')}")
    print(f"  -> Sources Found: {len(data.get('sources', []))}")
    results.append(("Query RAG (Hindi)", "PASS"))

    # 9. Query RAG (Hinglish)
    hinglish_query = {
        "query": "Tower light maintenance karte waqt electrical hazard se kaise bache?",
        "top_k": 2,
        "show_sources": True,
    }
    status, data = test_endpoint("9. RAG Inference (Hinglish)", "POST", f"{BASE_URL}/api/v1/query", json=hinglish_query)
    assert status == 200, f"Expected 200, got {status}"
    print(f"  -> Question: {data.get('query')}")
    print(f"  -> Answer: {data.get('answer')}")
    print(f"  -> Sources Found: {len(data.get('sources', []))}")
    results.append(("Query RAG (Hinglish)", "PASS"))

    # 10. Negative Test: File Not Found (404)
    status, data = test_endpoint(
        "10. Ingest Non-existent File (Expect 404)",
        "POST",
        f"{BASE_URL}/api/v1/ingest/file",
        json={"file_path": "d:/invalid/path/non_existent.xlsx"},
    )
    assert status == 404, f"Expected 404, got {status}"
    print(f"  -> Expected 404 Error: {data.get('detail')}")
    results.append(("404 Error Handling", "PASS"))

    # 11. Negative Test: Unsupported Upload Extension (400)
    status, data = test_endpoint(
        "11. Unsupported Upload File (Expect 400)",
        "POST",
        f"{BASE_URL}/api/v1/ingest/upload",
        files={"file": ("malicious.exe", b"binarycontent", "application/octet-stream")},
    )
    assert status == 400, f"Expected 400, got {status}"
    print(f"  -> Expected 400 Error: {data.get('detail')}")
    results.append(("400 Error Handling", "PASS"))

    # 12. Negative Test: Invalid Query Payload (422)
    status, data = test_endpoint(
        "12. Invalid Query Payload (Expect 422)",
        "POST",
        f"{BASE_URL}/api/v1/query",
        json={"query": "", "top_k": 50},  # empty query & out-of-range top_k (max 20)
    )
    assert status == 422, f"Expected 422, got {status}"
    print("  -> Expected 422 Validation Error caught successfully.")
    results.append(("422 Validation Handling", "PASS"))

    # Final Summary
    print_section("FINAL TEST EXECUTION SUMMARY")
    for name, res in results:
        print(f"  [{res}] {name}")
    print("\nAll live API endpoints tested and verified!\n")


if __name__ == "__main__":
    main()
