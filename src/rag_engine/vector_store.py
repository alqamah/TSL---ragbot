import hashlib
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.rag_engine.schema import DocumentChunk

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
    def current_key_index(self) -> int:
        return self._current_idx

    @property
    def current_key_label(self) -> str:
        return f"Key {self._current_idx + 1}"

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

    @staticmethod
    def is_key_rotation_error(error_text: str) -> bool:
        """Return whether an API error can be recovered by trying another key."""
        normalized_error = error_text.upper()
        return any(
            marker in normalized_error
            for marker in (
                "429",
                "RESOURCE_EXHAUSTED",
                "401",
                "403",
                "UNAUTHENTICATED",
                "PERMISSION_DENIED",
                "API_KEY_INVALID",
            )
        )


class VectorStoreManager:
    """Gemini embedding store backed by either Qdrant Server or local Qdrant."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        db_mode: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_path: str = "./qdrant_db",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        client_manager: Optional[GeminiClientManager] = None,
    ):
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or "industrial_sops_gemini"
        )
        self.model_name = (
            model_name
            or os.getenv("GEMINI_EMBEDDING_MODEL")
            or "models/gemini-embedding-001"
        )
        self.vector_size = 3072

        self.client_manager = client_manager or GeminiClientManager()
        self.ai_client = self.client_manager.current_client

        selected_mode = (
            db_mode
            or os.getenv("QDRANT_MODE")
            or "auto"
        ).strip().lower()

        configured_path = os.getenv("QDRANT_PATH", qdrant_path)
        configured_url = (
            qdrant_url
            or os.getenv("QDRANT_URL")
            or os.getenv("QDRANT_ENDPOINT")
            or os.getenv("QDRANT_CLOUD_URL")
        )
        configured_api_key = (
            qdrant_api_key
            or os.getenv("QDRANT_API_KEY")
            or os.getenv("QDRANT_API")
        )

        def _mask_url(url: str) -> str:
            if not url:
                return ""
            parts = url.split("://")
            scheme = parts[0] + "://" if len(parts) > 1 else ""
            host = parts[-1]
            if len(host) > 28:
                return f"{scheme}{host[:12]}...{host[-12:]}"
            return url

        if selected_mode in ["cloud", "online"]:
            if not configured_url:
                raise ValueError("Qdrant Cloud mode requested, but no QDRANT_URL or QDRANT_ENDPOINT is configured.")
            print(f"[VectorStore] Connecting to Qdrant Cloud at {_mask_url(configured_url)}...")
            client = QdrantClient(
                url=configured_url,
                api_key=configured_api_key,
                timeout=20.0,
                check_compatibility=False,
            )
            client.get_collections()
            self.qdrant_client = client
            self.mode = f"cloud ({_mask_url(configured_url)})"

        elif selected_mode in ["local", "embedded", "disk"]:
            print(f"[VectorStore] Initializing local Qdrant storage at '{configured_path}'...")
            self.qdrant_client = QdrantClient(path=configured_path)
            self.mode = f"local_path ({configured_path})"

        elif selected_mode in ["server", "remote"]:
            print(f"[VectorStore] Connecting to Qdrant server at {qdrant_host}:{qdrant_port}...")
            client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                api_key=configured_api_key,
                timeout=5.0,
                check_compatibility=False,
            )
            client.get_collections()
            self.qdrant_client = client
            self.mode = f"server ({qdrant_host}:{qdrant_port})"

        else:
            # Auto mode: Try configured Cloud URL first, then Server, then fallback to Local Path
            connected = False
            if configured_url:
                try:
                    client = QdrantClient(
                        url=configured_url,
                        api_key=configured_api_key,
                        timeout=10.0,
                        check_compatibility=False,
                    )
                    client.get_collections()
                    self.qdrant_client = client
                    self.mode = f"cloud ({_mask_url(configured_url)})"
                    connected = True
                except Exception as cloud_err:
                    print(f"Warning: Cloud Qdrant connection failed ({cloud_err}).")

            if not connected:
                try:
                    client = QdrantClient(
                        host=qdrant_host,
                        port=qdrant_port,
                        api_key=configured_api_key,
                        timeout=3.0,
                        check_compatibility=False,
                    )
                    client.get_collections()
                    self.qdrant_client = client
                    self.mode = f"server ({qdrant_host}:{qdrant_port})"
                    connected = True
                except Exception:
                    pass

            if not connected:
                print(f"[VectorStore] Using local storage at '{configured_path}'.")
                try:
                    self.qdrant_client = QdrantClient(path=configured_path)
                    self.mode = f"local_path ({configured_path})"
                except Exception as local_err:
                    raise RuntimeError(
                        f"Qdrant server/cloud is unavailable and local storage '{configured_path}' "
                        "is unavailable or locked. Specify a different QDRANT_PATH or valid QDRANT_URL."
                    ) from local_err

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
            # If local embedded client on Windows, close persistence file handles first so SQLite files can be removed
            local_inner = getattr(self.qdrant_client, "_client", None)
            if local_inner and hasattr(local_inner, "collections") and name in local_inner.collections:
                try:
                    local_inner.collections[name].close()
                except Exception:
                    pass
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
        max_retries: int = 15,
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
                if not self.client_manager.is_key_rotation_error(error_text):
                    raise
                if attempt == max_retries - 1:
                    break

                masked_key = self.client_manager.current_key_masked
                self.client_manager.rotate()
                next_key = self.client_manager.current_key_masked

                if attempt < key_pool_size:
                    safe_print(
                        f"[Embedding] Key failure or quota limit on Key [{masked_key}]. "
                        f"Instantly rotating to Key [{next_key}]..."
                    )
                    time.sleep(0.5)
                else:
                    wait_time = min(15, 3 * (attempt - key_pool_size + 1))
                    safe_print(
                        f"[Embedding] All keys rate limited. Rotating to [{next_key}] & pausing {wait_time}s... "
                        f"(Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)

        raise RuntimeError("Exceeded maximum retries for Gemini API embedding due to rate limits across all keys.")

    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings(text)[0]

    def list_indexed_files(self, collection_name: Optional[str] = None) -> List[str]:
        """Return distinct source file names stored in the collection payloads."""
        target_collection = collection_name or self.collection_name
        files = set()
        offset = None

        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=target_collection,
                limit=256,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for point in points:
                source = (getattr(point, "payload", {}) or {}).get("source")
                if isinstance(source, str) and source.strip():
                    files.add(source.strip())
            if offset is None:
                break

        return sorted(files, key=str.casefold)

    @staticmethod
    def _point_id(chunk: DocumentChunk) -> str:
        payload = chunk.metadata.copy() if chunk.metadata else {}
        document_metadata = payload.get("document_metadata")
        if isinstance(document_metadata, dict):
            document_metadata = document_metadata.copy()
            document_metadata.pop("processed_at", None)
            payload["document_metadata"] = document_metadata
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
            self.qdrant_client.upsert(
                collection_name=target_collection,
                points=points,
                wait=True,
            )
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
