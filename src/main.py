import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.engine import DataExtractionEngine
from src.generator import RAGGenerator
from src.vector_store import VectorStoreManager

load_dotenv()


class RAGPipeline:
    """End-to-end RAG pipeline managing Ingestion, Retrieval, and Generation."""

    def __init__(
        self,
        collection_name: str = "industrial_sops_gemini",
        generation_model: Optional[str] = None,
        embedding_model: str = "models/gemini-embedding-001",
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = (
            generation_model
            or os.getenv("GEMINI_GENERATION_MODEL")
            or "gemini-3.7-flash"
        )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing!")

        self.ai_client = genai.Client(api_key=api_key)
        self.extractor = DataExtractionEngine()
        self.vector_store = VectorStoreManager(
            collection_name=self.collection_name,
            model_name=self.embedding_model,
        )
        self.generator = RAGGenerator(
            collection_name=self.collection_name,
            generation_model=self.generation_model,
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
        )

    def count_indexed_points(self) -> int:
        """Return the count of vector points in the active collection."""
        try:
            if self.vector_store.qdrant_client.collection_exists(self.collection_name):
                info = self.vector_store.qdrant_client.get_collection(self.collection_name)
                return info.points_count or 0
        except Exception:
            pass
        return 0

    def ingest_file(self, file_path: str) -> int:
        """Extract chunks from a single document and index into Qdrant."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"\n[Ingest] Extracting document: {file_path}...")
        extracted_doc = self.extractor.extract(file_path)
        chunks = extracted_doc.chunks
        print(f"[Ingest] Extracted {len(chunks)} chunks from '{extracted_doc.file_name}'.")

        if not chunks:
            print("[Ingest] No chunks found to index.")
            return 0

        print(f"[Ingest] Upserting to collection '{self.collection_name}'...")
        uploaded = self.vector_store.upsert_chunks(chunks)
        print(f"[Ingest] Successfully indexed {uploaded} chunks.\n")
        return uploaded

    def ingest_directory(self, dir_path: str) -> int:
        """Scan directory and index all supported documents."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        supported_extensions = set(self.extractor.parsers.keys())
        all_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]

        if not all_files:
            print(f"[Ingest] No supported files found in '{dir_path}'.")
            return 0

        print(f"[Ingest] Found {len(all_files)} supported files in '{dir_path}'. Starting batch indexing...")
        total_indexed = 0
        for file_path in all_files:
            try:
                total_indexed += self.ingest_file(file_path)
            except Exception as e:
                print(f"[Ingest Error] Failed to process '{file_path}': {e}")

        print(f"[Ingest Complete] Total {total_indexed} chunks indexed into '{self.collection_name}'.\n")
        return total_indexed

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks using semantic vector search."""
        return self.generator.retrieve_with_metrics(query=query, top_k=top_k)["sources"]

    def generate_answer(
        self,
        query: str,
        contexts: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3,
    ) -> str:
        """Synthesize an answer using Gemini conditioned on retrieved context."""
        result = self.generator.generate_answer(query=query, top_k=top_k, verbose=False)
        return result["answer"]

    def ask(self, query: str, top_k: int = 3, show_sources: bool = True) -> Dict[str, Any]:
        """Complete RAG step: Retrieve context, synthesize answer, and log layer metrics."""
        return self.generator.generate_answer(query=query, top_k=top_k, verbose=show_sources)

    def close(self) -> None:
        self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def print_banner(pipeline: RAGPipeline):
    points_count = pipeline.count_indexed_points()
    print("=" * 70)
    print("      INDUSTRIAL SOP RAG ASSISTANT (Gemini + Qdrant)       ")
    print("=" * 70)
    print(f"Collection    : {pipeline.collection_name}")
    print(f"Storage Mode  : {pipeline.vector_store.mode}")
    print(f"Indexed Chunks: {points_count}")
    print(f"LLM Model     : {pipeline.generation_model}")
    print(f"Embed Model   : {pipeline.embedding_model}")
    print("=" * 70)
    print("Commands:")
    print("  Type your question and press Enter.")
    print("  ':ingest <path>' to index a file or folder.")
    print("  ':status'        to check indexed chunks count.")
    print("  'exit' / 'quit'  to exit.")
    print("=" * 70)


def run_interactive_session(pipeline: RAGPipeline, top_k: int = 3, show_sources: bool = True):
    print_banner(pipeline)

    # Check if empty collection and default SOPs exist
    points_count = pipeline.count_indexed_points()
    if points_count == 0:
        default_sop_dir = os.path.join(PROJECT_ROOT, "io", "SOP")
        if os.path.isdir(default_sop_dir):
            print("\n[Notice] The vector database collection is currently empty.")
            choice = input(f"Would you like to ingest documents from '{default_sop_dir}' now? (y/n): ").strip().lower()
            if choice in ["y", "yes"]:
                pipeline.ingest_directory(default_sop_dir)

    while True:
        try:
            query = input("\nAsk Query > ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if query.startswith(":ingest "):
                target_path = query[8:].strip()
                if not os.path.isabs(target_path):
                    target_path = os.path.join(PROJECT_ROOT, target_path)
                if os.path.isdir(target_path):
                    pipeline.ingest_directory(target_path)
                elif os.path.isfile(target_path):
                    pipeline.ingest_file(target_path)
                else:
                    print(f"Path not found: {target_path}")
                continue

            if query.lower() == ":status":
                print(f"Currently indexed points in '{pipeline.collection_name}': {pipeline.count_indexed_points()}")
                continue

            print("\n[1/2] Retrieving relevant SOP context...")
            result = pipeline.ask(query, top_k=top_k, show_sources=show_sources)

            if show_sources and result["sources"]:
                print(f"[2/2] Found {len(result['sources'])} relevant chunks:")
                for idx, src in enumerate(result["sources"], 1):
                    doc_source = src.get("source") or "Unknown"
                    score = src.get("score", 0.0)
                    snippet = src.get("content", "").replace("\n", " ")[:150]
                    print(f"  [{idx}] (Score: {score:.3f}) | {doc_source} -> \"{snippet}...\"")

            print("\n" + "-" * 70)
            print("ASSISTANT RESPONSE:")
            print("-" * 70)
            print(result["answer"])
            print("-" * 70)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as err:
            print(f"\n[Error processing request]: {err}")


def main():
    parser = argparse.ArgumentParser(description="Industrial SOP RAG Pipeline CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query to run without interactive prompt")
    parser.add_argument("--ingest-dir", "-i", type=str, help="Path to directory of SOP documents to ingest")
    parser.add_argument("--ingest-file", type=str, help="Path to single document to ingest")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--collection", type=str, default="industrial_sops_gemini", help="Qdrant collection name")
    parser.add_argument("--model", "-m", type=str, default="gemini-3.7-flash", help="Gemini generation model")
    parser.add_argument("--no-sources", action="store_true", help="Hide retrieved sources in output")

    args = parser.parse_args()

    pipeline = RAGPipeline(
        collection_name=args.collection,
        generation_model=args.model,
    )

    try:
        if args.ingest_dir:
            pipeline.ingest_directory(args.ingest_dir)

        if args.ingest_file:
            pipeline.ingest_file(args.ingest_file)

        if args.query:
            result = pipeline.ask(args.query, top_k=args.top_k, show_sources=not args.no_sources)
            print("-" * 75)
            print("ASSISTANT RESPONSE:")
            print("-" * 75)
            print(result["answer"])
            print("-" * 75)
        else:
            run_interactive_session(pipeline, top_k=args.top_k, show_sources=not args.no_sources)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
