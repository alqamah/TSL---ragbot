import argparse
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.rag_engine import DataExtractionEngine, VectorStoreManager

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run the Gemini-backed RAG ingestion and query test.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join(PROJECT_ROOT, "io", "SOP", "01_test.XLSX"),
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="\u0915\u0902\u092a\u094d\u0930\u0947\u0938\u0930 \u0915\u094b \u0938\u094d\u091f\u093e\u0930\u094d\u091f \u0915\u0948\u0938\u0947 \u0915\u0930\u0947\u0902\u0964",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GEMINI VECTOR EMBEDDING & RAG PIPELINE TEST")
    print("=" * 60)

    extractor = DataExtractionEngine()
    vector_db = VectorStoreManager()
    print(f"VectorStore active mode: {vector_db.mode}")

    test_file = args.file_path
    if not os.path.isabs(test_file):
        test_file = os.path.join(PROJECT_ROOT, test_file)
    if not os.path.exists(test_file):
        alternate = os.path.join(PROJECT_ROOT, "io", "SOP", os.path.basename(test_file))
        if os.path.exists(alternate):
            test_file = alternate

    print(f"\n1. Extracting and chunking test file: {test_file}...")
    doc = extractor.extract(test_file)
    chunks = doc.chunks
    print(f"Extracted {len(chunks)} chunks from {doc.file_name}.")

    print(f"\n2. Upserting {len(chunks)} chunks to Qdrant using Gemini...")
    stored = vector_db.upsert_chunks(chunks)
    print(f"Stored {stored} chunks.")

    print(f"\n3. Performing similarity search for query: '{args.query}'...")
    hits = vector_db.search(args.query, limit=2)

    print(f"\nTop Search Results for query: '{args.query}':\n")
    for idx, hit in enumerate(hits, 1):
        print(f"--- Result {idx} (Score: {hit['score']:.4f}) ---")
        print(f"Source : {hit['source']}")
        print(f"Content Snippet:\n{hit['content'][:300]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
