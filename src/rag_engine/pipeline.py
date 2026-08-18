import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.rag_engine.engine import DataExtractionEngine
from src.rag_engine.generator import RAGGenerator
from src.rag_engine.vector_store import VectorStoreManager

load_dotenv()


class RAGPipeline:
    """End-to-end RAG pipeline managing Ingestion, Retrieval, and Generation."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        generation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or "industrial_sops_gemini"
        )
        self.embedding_model = (
            embedding_model
            or os.getenv("GEMINI_EMBEDDING_MODEL")
            or "models/gemini-embedding-001"
        )
        self.generation_model = (
            generation_model
            or os.getenv("GEMINI_GENERATION_MODEL")
            or "gemini-3.7-flash"
        )

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
