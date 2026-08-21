import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure standard streams are configured with UTF-8 and replace error handler
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Print helper that guarantees no UnicodeEncodeError on Windows charmap consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    sanitized_args.append(arg.encode(encoding, errors="replace").decode(encoding))
                else:
                    sanitized_args.append(arg)
            print(*sanitized_args, **kwargs)
        except Exception:
            pass


from src.rag_engine.engine import DataExtractionEngine
from src.rag_engine.generator import RAGGenerator
from src.rag_engine.metadata import DocumentMetadataGenerator
from src.rag_engine.vector_store import VectorStoreManager

load_dotenv()


class RAGPipeline:
    """End-to-end RAG pipeline managing Ingestion, Retrieval, and Generation."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        generation_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        db_mode: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_path: Optional[str] = None,
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
        self.last_ingest_metadata: Optional[Dict[str, Any]] = None
        self.last_ingest_results: List[Dict[str, Any]] = []
        store_kwargs = {
            "collection_name": self.collection_name,
            "model_name": self.embedding_model,
            "db_mode": db_mode,
            "qdrant_url": qdrant_url,
            "qdrant_api_key": qdrant_api_key,
        }
        if qdrant_path:
            store_kwargs["qdrant_path"] = qdrant_path

        self.vector_store = VectorStoreManager(**store_kwargs)
        self.metadata_generator = DocumentMetadataGenerator()
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

    def list_indexed_files(self) -> List[str]:
        """Return the distinct source files currently indexed in Qdrant."""
        return self.vector_store.list_indexed_files(self.collection_name)

    @property
    def llm_models(self) -> List[str]:
        """Return all generation models this pipeline can use, in selection order."""
        return self.generator.candidate_models

    def ingest_file(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_table_rows: Optional[int] = None,
    ) -> int:
        """Extract chunks from a single document and index into Qdrant."""
        file_name = os.path.basename(file_path)
        document_metadata: Optional[Dict[str, Any]] = None
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            safe_print(f"\n[Ingest] Extracting document: {file_path} (chunk_size={chunk_size or 600}, overlap={chunk_overlap or 100}, table_rows={max_table_rows or 4})...")
            extracted_doc = self.extractor.extract(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_table_rows=max_table_rows,
            )
            chunks = extracted_doc.chunks
            safe_print(f"[Ingest] Extracted {len(chunks)} chunks from '{extracted_doc.file_name}'.")

            extension = os.path.splitext(extracted_doc.file_name)[1].lower()
            parser_cls = self.extractor.parsers.get(extension)
            parser_name = parser_cls.__name__ if parser_cls else "UnknownParser"
            document_metadata = self.metadata_generator.generate(
                file_path=file_path,
                document=extracted_doc,
                parser_name=parser_name,
            )
            semantic_metadata = self.metadata_generator.generate_semantic(
                extracted_text=extracted_doc.full_text
                or "\n\n".join(chunk.content for chunk in chunks),
                client_manager=getattr(self.vector_store, "client_manager", None),
                model_name=getattr(self, "generation_model", ""),
            )
            if semantic_metadata.get("summary"):
                document_metadata["extractive_summary"] = document_metadata["summary"]
            document_metadata.update(semantic_metadata)
            extracted_doc.metadata = document_metadata.copy()
            document_metadata["status"] = "processing"
            self.last_ingest_metadata = document_metadata

            if not chunks:
                raise ValueError("No extractable text or chunks were found in the document.")

            enriched_chunks = self.metadata_generator.attach_to_chunks(
                chunks,
                document_metadata,
            )
            safe_print(f"[Ingest] Upserting to collection '{self.collection_name}'...")
            uploaded = self.vector_store.upsert_chunks(enriched_chunks)
            if uploaded != len(enriched_chunks):
                raise RuntimeError(
                    f"Qdrant acknowledged {uploaded} of {len(enriched_chunks)} chunks."
                )

            document_metadata.update(
                {
                    "status": "success",
                    "chunks_indexed": uploaded,
                    "indexed_collection": self.collection_name,
                }
            )
            self.last_ingest_metadata = document_metadata
            self._print_upload_success(document_metadata)
            return uploaded
        except Exception as error:
            if document_metadata is None:
                document_metadata = {
                    "file_name": file_name,
                    "status": "failed",
                }
            document_metadata.update(
                {
                    "status": "failed",
                    "error": str(error),
                }
            )
            self.last_ingest_metadata = document_metadata
            safe_print(f"[Upload Failed] File: {file_name}")
            safe_print(f"[Upload Failed] Reason: {error}\n")
            raise

    @staticmethod
    def _print_upload_success(document_metadata: Dict[str, Any]) -> None:
        """Print the terminal confirmation and summary for a completed upload safely."""
        safe_print("[Upload Successful]")
        safe_print(f"  File: {document_metadata.get('file_name', 'Unknown')}")
        safe_print(f"  Type: {document_metadata.get('file_type', 'Unknown')}")
        safe_print(f"  Document ID: {document_metadata.get('document_id', 'Unknown')}")
        safe_print(f"  Size: {document_metadata.get('file_size_bytes', 0)} bytes")
        safe_print(f"  Extracted words: {document_metadata.get('extracted_word_count', 0)}")
        safe_print(f"  Chunks indexed: {document_metadata.get('chunks_indexed', 0)}")
        safe_print(f"  Summary: {document_metadata.get('summary', 'No summary available.')}")
        if document_metadata.get("category"):
            safe_print(f"  Category: {document_metadata['category']}")
        if document_metadata.get("topics"):
            safe_print(f"  Topics: {', '.join(document_metadata['topics'])}")
        keywords = document_metadata.get("keywords", [])
        if keywords:
            safe_print(f"  Keywords: {', '.join(keywords)}")
        for warning in document_metadata.get("metadata_warnings", []):
            safe_print(f"  Metadata warning: {warning}")
        safe_print()

    def ingest_directory(
        self,
        dir_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        max_table_rows: Optional[int] = None,
    ) -> int:
        """Scan directory and index all supported documents."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        self.last_ingest_results = []

        supported_extensions = set(self.extractor.parsers.keys())
        all_files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]

        if not all_files:
            safe_print(f"[Ingest] No supported files found in '{dir_path}'.")
            return 0

        safe_print(f"[Ingest] Found {len(all_files)} supported files in '{dir_path}'. Starting batch indexing...")
        total_indexed = 0
        for file_path in all_files:
            try:
                total_indexed += self.ingest_file(
                    file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_table_rows=max_table_rows,
                )
            except Exception as e:
                safe_print(f"[Ingest Error] Failed to process '{file_path}': {e}")
            finally:
                if self.last_ingest_metadata:
                    self.last_ingest_results.append(self.last_ingest_metadata.copy())

        safe_print(f"[Ingest Complete] Total {total_indexed} chunks indexed into '{self.collection_name}'.\n")
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

    def ask(
        self,
        query: str,
        top_k: int = 3,
        show_sources: bool = True,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete RAG step: Retrieve context, synthesize answer, and log layer metrics."""
        return self.generator.generate_answer(
            query=query,
            top_k=top_k,
            verbose=show_sources,
            model=model,
        )

    def list_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """List generation-capable Gemini models available to the configured API key."""
        return self.generator.list_available_models(force_refresh=force_refresh)

    def reset_database(self) -> Dict[str, Any]:
        """Delete and recreate the active collection to completely reset indexed vectors."""
        safe_print(f"[Reset] Resetting collection '{self.collection_name}'...")
        self.vector_store.create_collection(self.collection_name, recreate=True)
        self.last_ingest_metadata = None
        self.last_ingest_results = []
        safe_print(f"[Reset] Collection '{self.collection_name}' reset successfully.")
        return {
            "status": "success",
            "message": f"Database collection '{self.collection_name}' has been completely reset.",
            "collection_name": self.collection_name,
            "indexed_points": self.count_indexed_points(),
        }

    def close(self) -> None:
        self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
