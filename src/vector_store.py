# vector_store.py
import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.schema import DocumentChunk

load_dotenv()

class VectorStore:
    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initializes the VectorStore with OpenAI embedding client and Qdrant vector database.
        
        Args:
            qdrant_url: Qdrant server URL (e.g., http://localhost:6333). If None, checks env or defaults to local host.
            qdrant_path: Path for local embedded Qdrant database if server is unavailable.
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            embedding_model: OpenAI embedding model name.
        """
        self.embedding_model = embedding_model
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

        # Initialize Qdrant Client (attempt server connection first, fallback to path if specified)
        target_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        
        try:
            client = QdrantClient(url=target_url, timeout=3.0, check_compatibility=False)
            client.get_collections() # Test connection
            self.qdrant_client = client
            self.mode = f"server ({target_url})"
        except Exception as e:
            # Fallback to local persistent database path if server is down/not running
            local_path = qdrant_path or os.getenv("QDRANT_PATH", "./qdrant_db")
            print(f"Warning: Could not connect to Qdrant server at '{target_url}' ({e}). Falling back to local storage path: '{local_path}'.")
            self.qdrant_client = QdrantClient(path=local_path)
            self.mode = f"local_path ({local_path})"

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> None:
        """
        Creates or recreates a collection in Qdrant.
        """
        if recreate and self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.delete_collection(collection_name)

        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of texts using OpenAI embedding API.
        """
        if not self.openai_client:
            raise ValueError("OpenAI client is not initialized. Please provide a valid OPENAI_API_KEY.")
            
        # Clean texts to remove empty string embeddings
        cleaned_texts = [t.replace("\n", " ") if t else " " for t in texts]
        
        response = self.openai_client.embeddings.create(
            input=cleaned_texts,
            model=self.embedding_model
        )
        return [item.embedding for item in response.data]

    def upsert_document_chunks(
        self,
        collection_name: str,
        chunks: List[DocumentChunk],
        batch_size: int = 64
    ) -> int:
        """
        Embeds DocumentChunks and uploads them to the specified Qdrant collection.
        
        Args:
            collection_name: Name of Qdrant collection.
            chunks: List of DocumentChunk dataclass objects.
            batch_size: Number of chunks per embedding & upload batch.
            
        Returns:
            Total number of chunks uploaded.
        """
        if not chunks:
            return 0

        self.create_collection(collection_name=collection_name)

        total_uploaded = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            # 1. Generate embeddings for batch
            embeddings = self.get_embeddings(batch_texts)

            # 2. Build Qdrant PointStructs
            points = []
            for chunk, vector in zip(batch_chunks, embeddings):
                point_id = str(uuid.uuid4())
                payload = {
                    "content": chunk.content,
                    **chunk.metadata
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            # 3. Upsert into Qdrant
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            total_uploaded += len(points)

        return total_uploaded

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches Qdrant vector collection for closest matching document chunks.
        
        Args:
            collection_name: Target Qdrant collection.
            query_text: Natural language user query.
            top_k: Number of top search results to return.
            filter_dict: Optional exact-match payload filter key-values (e.g. {"source": "01_test.XLSX"}).
            
        Returns:
            List of matching results with score and payload metadata.
        """
        # Embed query text
        query_vector = self.get_embeddings([query_text])[0]

        # Construct optional payload filter
        qdrant_filter = None
        if filter_dict:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            qdrant_filter = Filter(must=must_conditions)

        # Query Qdrant
        results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k
        )

        output = []
        for point in results.points:
            output.append({
                "id": point.id,
                "score": point.score,
                "content": point.payload.get("content", ""),
                "metadata": {k: v for k, v in point.payload.items() if k != "content"}
            })

        return output
