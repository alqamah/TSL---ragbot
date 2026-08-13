# test/test_vector_store.py
import os
import sys
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine import DataExtractionEngine
from src.vector_store import VectorStore

load_dotenv()

def main():
    print("=" * 60)
    print("VECTOR STORE & EMBEDDING ENGINE TEST")
    print("=" * 60)

    # 1. Initialize VectorStore
    vs = VectorStore()
    print(f"VectorStore active mode: {vs.mode}")

    collection_name = "sop_test_collection"
    print(f"Creating/Ensuring Qdrant collection '{collection_name}'...")
    vs.create_collection(collection_name=collection_name, recreate=True)
    print("Collection initialized successfully!")

    # 2. Parse a test document
    test_file = os.path.join("io", "SOP", "01_test.XLSX")
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    print(f"\nExtracting and chunking test file: {test_file}...")
    engine = DataExtractionEngine()
    doc = engine.extract(test_file)
    print(f"Extracted {len(doc.chunks)} chunks from {doc.file_name}.")

    # 3. Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[NOTE] OPENAI_API_KEY is not set in environment or .env file.")
        print("To run vector embedding and Qdrant upsert, add OPENAI_API_KEY to your .env file.")
        print("Test verified: Qdrant client connection and schema initialization passed successfully.")
        return

    print(f"\nEmbedding and upserting {len(doc.chunks)} chunks into Qdrant...")
    uploaded_count = vs.upsert_document_chunks(collection_name=collection_name, chunks=doc.chunks)
    print(f"Successfully uploaded {uploaded_count} points to Qdrant!")

    # 4. Perform a test similarity query
    query = "कंप्रेसर का रखरखाव कैसे करें"
    print(f"\nPerforming similarity search for query: '{query}'...")
    results = vs.search(collection_name=collection_name, query_text=query, top_k=3)

    print("\nTop Search Results:")
    for idx, res in enumerate(results, 1):
        print(f"\nResult {idx} (Score: {res['score']:.4f}):")
        print(f"Source   : {res['metadata'].get('source')}")
        print(f"Sheet    : {res['metadata'].get('sheet_name')}")
        print(f"Content  :\n{res['content'][:200]}...")

if __name__ == "__main__":
    main()
