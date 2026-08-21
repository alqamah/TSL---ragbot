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
    gemini_api_key_used: Optional[str] = None
    llm_generation_ms: float = 0.0
    total_e2e_latency_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    metrics: LayerMetrics


class IngestDirectoryRequest(BaseModel):
    directory_path: str = Field(..., description="Absolute or relative path to directory containing documents")
    chunk_size: Optional[int] = Field(default=None, ge=50, le=5000, description="Custom chunk size in characters")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000, description="Custom chunk overlap in characters")
    max_table_rows: Optional[int] = Field(default=None, ge=1, le=50, description="Custom table rows per chunk")


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to a single document")
    chunk_size: Optional[int] = Field(default=None, ge=50, le=5000, description="Custom chunk size in characters")
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=1000, description="Custom chunk overlap in characters")
    max_table_rows: Optional[int] = Field(default=None, ge=1, le=50, description="Custom table rows per chunk")


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int
    total_points: int
    summary: Optional[str] = None
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    documents: List[Dict[str, Any]] = Field(default_factory=list)


class StatusResponse(BaseModel):
    collection_name: str
    indexed_points: int
    storage_mode: str
    generation_model: str
    embedding_model: str
    llm_models: List[str] = Field(default_factory=list, description="Generation models available to the pipeline")
    status: str = "online"


class IndexedFilesResponse(BaseModel):
    files: List[str] = Field(default_factory=list, description="Distinct source files indexed in the vector database")
    total_files: int = 0


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"


class LogEntry(BaseModel):
    id: int = Field(..., description="Monotonically increasing cursor of the log line")
    ts: str = Field(..., description="Timestamp of the line as HH:MM:SS")
    message: str = Field(..., description="Raw terminal line printed by the backend")


class LogsResponse(BaseModel):
    logs: List[LogEntry] = Field(default_factory=list, description="Terminal lines newer than the requested cursor")
    cursor: int = Field(0, description="Latest cursor to use for subsequent polls")
    busy: bool = Field(False, description="True when backend logged activity within the last few seconds")
