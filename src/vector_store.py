import hashlib
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.schema import DocumentChunk

load_dotenv()


class GeminiClientManager:
    """Manages a pool of Gemini API keys with round-robin cycling and rate-limit recovery."""

    def __init__(self, api_keys: Optional[List[str]] = None):
        if api_keys:
            self.api_keys = [k.strip() for k in api_keys if k.strip()]
        else:
            keys: List[str] = []
            raw_multi = os.getenv("GEMINI_API_KEYS", "")
            if raw_multi:
                keys.extend([k.strip() for k in raw_multi.split(",") if k.strip()])
            single = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if single and single.strip() not in keys:
                keys.append(single.strip())
            for idx in range(1, 10):
                k = os.getenv(f"GEMINI_API_KEY_{idx}")
                if k and k.strip() and k.strip() not in keys:
                    keys.append(k.strip())
            self.api_keys = keys

        if not self.api_keys:
            raise ValueError("No GEMINI_API_KEY or GEMINI_API_KEYS configured in environment!")

        self.clients = [genai.Client(api_key=key) for key in self.api_keys]
        self._current_idx = 0

    @property
    def current_client(self) -> genai.Client:
        return self.clients[self._current_idx]

    @property
    def current_key_masked(self) -> str:
        k = self.api_keys[self._current_idx]
        if len(k) > 12:
            return f"{k[:7]}...{k[-5:]}"
        return "***"

    @property
    def key_count(self) -> int:
        return len(self.api_keys)

    def rotate(self) -> genai.Client:
        """Rotate to next available API key in round-robin fashion."""
        self._current_idx = (self._current_idx + 1) % len(self.clients)
        return self.current_client


class VectorStoreManager:
    """Gemini embedding store backed by either Qdrant Server or local Qdrant."""

    def __init__(
        self,
        collection_name: str = "industrial_sops_gemini",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_path: str = "./qdrant_db",
        model_name: str = "models/gemini-embedding-001",
        client_manager: Optional[GeminiClientManager] = None,
    ):
        self.collection_name = collection_name
        self.model_name = model_name
        self.vector_size = 3072

        self.client_manager = client_manager or GeminiClientManager()
        self.ai_client = self.client_manager.current_client

        configured_path = os.getenv("QDRANT_PATH", qdrant_path)
        configured_url = os.getenv("QDRANT_URL")
        try:
            if configured_url:
                client = QdrantClient(
                    url=configured_url,
                    timeout=3.0,
                    check_compatibility=False,
                )
                client.get_collections()
                self.qdrant_client = client
                self.mode = f"server ({configured_url})"
            else:
                client = QdrantClient(
                    host=qdrant_host,
                    port=qdrant_port,
                    timeout=3.0,
                    check_compatibility=False,
                )
                client.get_collections()
                self.qdrant_client = client
                self.mode = f"server ({qdrant_host}:{qdrant_port})"
        except Exception as server_error:
            print(
                f"Warning: Qdrant server unavailable ({server_error}). "
                f"Using local storage at '{configured_path}'."
            )
            try:
                self.qdrant_client = QdrantClient(path=configured_path)
            except Exception as local_error:
                raise RuntimeError(
                    f"Qdrant server is unavailable and local storage '{configured_path}' "
                    "is unavailable or locked. Use QDRANT_URL or a different QDRANT_PATH."
                ) from local_error
            self.mode = f"local_path ({configured_path})"

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        if self.qdrant_client.collection_exists(self.collection_name):
            info = self.qdrant_client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size
            if existing_size != self.vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' has vector size {existing_size}, "
                    f"but model '{self.model_name}' returns {self.vector_size}. "
                    "Use a new collection or explicitly recreate the old one."
                )
            return

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created Qdrant Collection: '{self.collection_name}' ({self.vector_size} dims)")

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ) -> None:
        """Create a collection, optionally replacing it when explicitly requested."""
        name = collection_name or self.collection_name
        size = vector_size or self.vector_size
        if recreate and self.qdrant_client.collection_exists(name):
            self.qdrant_client.delete_collection(name)
        if self.qdrant_client.collection_exists(name):
            info = self.qdrant_client.get_collection(name)
            existing_size = info.config.params.vectors.size
            if existing_size != size:
                raise ValueError(
                    f"Collection '{name}' has vector size {existing_size}; expected {size}."
                )
        else:
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=distance),
            )

    def close(self) -> None:
        self.qdrant_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        max_retries: int = 6,
    ) -> List[List[float]]:
        input_texts = [texts] if isinstance(texts, str) else list(texts)
        cleaned_texts = [text.replace("\n", " ") if text else " " for text in input_texts]
        if not cleaned_texts:
            return []
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        key_pool_size = self.client_manager.key_count
        for attempt in range(max_retries):
            client = self.client_manager.current_client
            try:
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=cleaned_texts,
                )
                embeddings = [list(embedding.values) for embedding in response.embeddings]
                if len(embeddings) != len(cleaned_texts):
                    raise RuntimeError(
                        f"Gemini returned {len(embeddings)} embeddings for "
                        f"{len(cleaned_texts)} inputs."
                    )
                if any(len(embedding) != self.vector_size for embedding in embeddings):
                    raise RuntimeError(
                        f"Gemini returned an embedding with the wrong size; expected {self.vector_size}."
                    )
                return embeddings
            except Exception as error:
                error_text = str(error)
                if "429" not in error_text and "RESOURCE_EXHAUSTED" not in error_text:
                    raise
                if attempt == max_retries - 1:
                    break

                masked_key = self.client_manager.current_key_masked
                self.client_manager.rotate()
                next_key = self.client_manager.current_key_masked

                if attempt < key_pool_size:
                    print(
                        f"[Embedding] Rate limit (429) on Key [{masked_key}]. "
                        f"Instantly rotating to Key [{next_key}]..."
                    )
                    time.sleep(0.5)
                else:
                    wait_time = 3 * (attempt - key_pool_size + 1)
                    print(
                        f"[Embedding] All keys rate limited. Rotating to [{next_key}] & pausing {wait_time}s... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)

        raise RuntimeError("Exceeded maximum retries for Gemini API embedding due to rate limits across all keys.")

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings(text)[0]

    @staticmethod
    def _point_id(chunk: DocumentChunk) -> str:
        payload = chunk.metadata.copy() if chunk.metadata else {}
        identity = json.dumps(
            {"content": chunk.content, "metadata": payload},
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 15,
        collection_name: Optional[str] = None,
    ) -> int:
        if not chunks:
            return 0
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        target_collection = collection_name or self.collection_name
        if not self.qdrant_client.collection_exists(target_collection):
            raise ValueError(f"Qdrant collection does not exist: '{target_collection}'")

        print(
            f"Embedding {len(chunks)} chunks using Gemini '{self.model_name}' "
            f"(batch_size={batch_size})..."
        )
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        uploaded = 0
        for batch_idx, start in enumerate(range(0, len(chunks), batch_size), 1):
            batch_chunks = chunks[start : start + batch_size]
            embeddings = self.get_embeddings([chunk.content for chunk in batch_chunks])
            points = []
            for chunk, vector in zip(batch_chunks, embeddings):
                payload = chunk.metadata.copy() if chunk.metadata else {}
                payload["content"] = chunk.content
                points.append(
                    PointStruct(
                        id=self._point_id(chunk),
                        vector=vector,
                        payload=payload,
                    )
                )
            self.qdrant_client.upsert(collection_name=target_collection, points=points)
            uploaded += len(points)
            print(f"Embedded batch {batch_idx}/{total_batches} ({uploaded}/{len(chunks)} stored)...")
            if batch_idx < total_batches:
                time.sleep(3.5)
        return uploaded

    def upsert_document_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        batch_size: int = 15,
    ) -> int:
        return self.upsert_chunks(chunks, batch_size=batch_size, collection_name=collection_name)

    def search(
        self,
        query: Optional[str] = None,
        limit: int = 3,
        *,
        collection_name: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        search_text = query if query is not None else query_text
        if not search_text:
            raise ValueError("A non-empty query is required")
        result_limit = top_k if top_k is not None else limit
        target_collection = collection_name or self.collection_name
        query_vector = self.get_embedding(search_text)

        if hasattr(self.qdrant_client, "query_points"):
            hits = self.qdrant_client.query_points(
                collection_name=target_collection,
                query=query_vector,
                limit=result_limit,
            ).points
        else:
            hits = self.qdrant_client.search(
                collection_name=target_collection,
                query_vector=query_vector,
                limit=result_limit,
            )

        results = []
        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}
            results.append(
                {
                    "score": hit.score,
                    "content": payload.get("content", ""),
                    "source": payload.get("source"),
                    "metadata": payload,
                }
            )
        return results


# Kept as a compatibility alias for existing callers.
VectorStore = VectorStoreManager
