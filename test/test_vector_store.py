import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.engine import DataExtractionEngine
from src.vector_store import VectorStoreManager

load_dotenv()


def main():
    print("=" * 60)
    print("VECTOR STORE & EMBEDDING ENGINE TEST")
    print("=" * 60)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("GEMINI_API_KEY or GOOGLE_API_KEY is not configured.")
        return

    collection_name = "sop_test_collection"
    test_file = os.path.join(PROJECT_ROOT, "io", "SOP", "01_test.XLSX")
    with VectorStoreManager(collection_name=collection_name) as vector_db:
        vector_db.create_collection(collection_name=collection_name)
        doc = DataExtractionEngine().extract(test_file)
        uploaded_count = vector_db.upsert_chunks(doc.chunks)
        print(f"Successfully uploaded {uploaded_count} points to Qdrant!")

        query = "\u0915\u0902\u092a\u094d\u0930\u0947\u0938\u0930 \u0915\u093e \u0930\u0916\u0930\u0916\u093e\u0935 \u0915\u0948\u0938\u0947 \u0915\u0930\u0947\u0902"
        results = vector_db.search(query, limit=3)
        for idx, result in enumerate(results, 1):
            print(f"Result {idx} (Score: {result['score']:.4f})")
            print(result["content"][:200])


if __name__ == "__main__":
    main()
