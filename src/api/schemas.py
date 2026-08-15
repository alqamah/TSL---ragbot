from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question in English, Hindi, or Hinglish")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of context chunks to retrieve")
    show_sources: bool = Field(default=True, description="Whether to include retrieved source chunks in response")


class SourceCitation(BaseModel):
    source: Optional[str] = Field(default="Unknown", description="Source document file name")
    score: float = Field(default=0.0, description="Cosine similarity score")
    content: str = Field(default="", description="Snippet or text of the retrieved chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata including sheet name, row range, or section")


class LayerMetrics(BaseModel):
    embedding_model: Optional[str] = None
    query_embedding_ms: float = 0.0
    qdrant_search_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    prompt_construction_ms: float = 0.0
    final_llm_model: Optional[str] = None
    llm_generation_ms: float = 0.0
    total_e2e_latency_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    metrics: LayerMetrics


class IngestDirectoryRequest(BaseModel):
    directory_path: str = Field(..., description="Absolute or relative path to directory containing documents")


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to a single document")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int
    total_points: int


class StatusResponse(BaseModel):
    collection_name: str
    indexed_points: int
    storage_mode: str
    generation_model: str
    embedding_model: str
    status: str = "online"


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
