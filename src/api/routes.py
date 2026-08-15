import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from src.api.schemas import (
    HealthResponse,
    IngestDirectoryRequest,
    IngestFileRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from src.rag_engine.pipeline import RAGPipeline

router = APIRouter()


def get_pipeline(request: Request) -> RAGPipeline:
    """Dependency to retrieve the active RAGPipeline singleton from app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized.",
        )
    return pipeline


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check endpoint."""
    return HealthResponse(status="healthy", version="1.0.0")


@router.get("/api/v1/status", response_model=StatusResponse, tags=["Telemetry & Status"])
async def get_system_status(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Retrieve vector database index stats, active storage mode, and model configurations."""
    try:
        points_count = pipeline.count_indexed_points()
        return StatusResponse(
            collection_name=pipeline.collection_name,
            indexed_points=points_count,
            storage_mode=pipeline.vector_store.mode,
            generation_model=pipeline.generation_model,
            embedding_model=pipeline.embedding_model,
            status="online",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch status: {str(e)}",
        )


@router.post("/api/v1/query", response_model=QueryResponse, tags=["RAG Inference"])
async def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Execute a semantic search and LLM grounded synthesis over industrial SOP documents."""
    try:
        result = pipeline.ask(
            query=request.query,
            top_k=request.top_k,
            show_sources=request.show_sources,
        )
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=result["sources"] if request.show_sources else [],
            metrics=result["metrics"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing query: {str(e)}",
        )


@router.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_file_by_path(
    request: IngestFileRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Index a single document specified by local path (.xlsx, .docx, .doc, .txt, .csv)."""
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found at path: {request.file_path}",
        )
    try:
        indexed_count = pipeline.ingest_file(request.file_path)
        total_points = pipeline.count_indexed_points()
        return IngestResponse(
            status="success",
            message=f"Successfully indexed '{os.path.basename(request.file_path)}'.",
            chunks_indexed=indexed_count,
            total_points=total_points,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest file: {str(e)}",
        )


@router.post("/api/v1/ingest/upload", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_uploaded_file(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Upload and index a document via multipart/form-data."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    supported = set(pipeline.extractor.parsers.keys())
    if suffix not in supported:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Supported: {list(supported)}",
        )

    # Save to a temporary file for parsing
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename or f"upload{suffix}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        indexed_count = pipeline.ingest_file(temp_file_path)
        total_points = pipeline.count_indexed_points()

        return IngestResponse(
            status="success",
            message=f"Uploaded and indexed {indexed_count} chunks from '{file.filename}'.",
            chunks_indexed=indexed_count,
            total_points=total_points,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process upload: {str(e)}",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/api/v1/ingest/directory", response_model=IngestResponse, tags=["Document Ingestion"])
async def ingest_directory(
    request: IngestDirectoryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Index all supported documents from a directory."""
    if not os.path.isdir(request.directory_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {request.directory_path}",
        )
    try:
        indexed_count = pipeline.ingest_directory(request.directory_path)
        total_points = pipeline.count_indexed_points()
        return IngestResponse(
            status="success",
            message=f"Indexed {indexed_count} total chunks from directory '{request.directory_path}'.",
            chunks_indexed=indexed_count,
            total_points=total_points,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest directory: {str(e)}",
        )
