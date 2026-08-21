import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status

from src.api import log_stream
from src.api.schemas import (
    HealthResponse,
    IndexedFilesResponse,
    IngestDirectoryRequest,
    IngestFileRequest,
    IngestResponse,
    LogsResponse,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
    StatusResponse,
)
from src.rag_engine.pipeline import RAGPipeline, safe_print

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
def get_system_status(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Retrieve vector database index stats, active storage mode, and model configurations."""
    try:
        points_count = pipeline.count_indexed_points()
        return StatusResponse(
            collection_name=pipeline.collection_name,
            indexed_points=points_count,
            storage_mode=pipeline.vector_store.mode,
            generation_model=pipeline.generation_model,
            embedding_model=pipeline.embedding_model,
            llm_models=pipeline.llm_models,
            status="online",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch status: {str(e)}",
        )


@router.get("/api/v1/logs", response_model=LogsResponse, tags=["Telemetry & Status"])
def stream_terminal_logs(
    since: int = Query(0, ge=0, description="Only return lines with cursor greater than this value"),
):
    """Stream backend terminal output (ingest progress, uploads, resets, warnings)."""
    entries, cursor, _ = log_stream.get_logs(since=since)
    return LogsResponse(
        logs=entries,
        cursor=cursor,
        busy=log_stream.is_busy(),
    )


@router.get("/api/v1/models", response_model=ModelsResponse, tags=["Telemetry & Status"])
def list_available_models(
    force: bool = Query(False, description="Bypass the 10-minute cache and re-query the Gemini API"),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """List generation-capable Gemini models available via the configured API key (free tier)."""
    try:
        models = pipeline.list_available_models(force_refresh=force)
        return ModelsResponse(models=models, total_models=len(models))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get("/api/v1/files", response_model=IndexedFilesResponse, tags=["Telemetry & Status"])
def list_indexed_files(pipeline: RAGPipeline = Depends(get_pipeline)):
    """List the distinct source files currently present in the vector database."""
    try:
        files = pipeline.list_indexed_files()
        return IndexedFilesResponse(files=files, total_files=len(files))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indexed files: {str(e)}",
        )


@router.post("/api/v1/reset", tags=["Database Management"])
@router.delete("/api/v1/reset", tags=["Database Management"])
def reset_database(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Wipe and reset the vector database collection."""
    try:
        result = pipeline.reset_database()
        return result
    except Exception as e:
        safe_print(f"[Reset Error] Failed to reset database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset database: {str(e)}",
        )


@router.post("/api/v1/query", tags=["RAG Inference"], response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Execute a semantic search and LLM grounded synthesis over industrial SOP documents."""
    try:
        result = pipeline.ask(
            query=request.query,
            top_k=request.top_k,
            show_sources=request.show_sources,
            model=request.model,
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
def ingest_file_by_path(
    request: IngestFileRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Index a single document specified by local path (.xlsx, .docx, .doc, .txt, .csv)."""
    if not os.path.exists(request.file_path):
        safe_print(f"[Upload Failed] File: {os.path.basename(request.file_path)}")
        safe_print(f"[Upload Failed] Reason: File not found at path: {request.file_path}\n")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found at path: {request.file_path}",
        )
    try:
        indexed_count = pipeline.ingest_file(
            request.file_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_table_rows=request.max_table_rows,
        )
        total_points = pipeline.count_indexed_points()
        document_metadata = pipeline.last_ingest_metadata or {}
        return IngestResponse(
            status="success",
            message=f"Successfully indexed '{os.path.basename(request.file_path)}'.",
            chunks_indexed=indexed_count,
            total_points=total_points,
            summary=document_metadata.get("summary"),
            document_metadata=document_metadata,
            documents=[document_metadata] if document_metadata else [],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest file: {str(e)}",
        )


@router.post("/api/v1/ingest/upload", response_model=IngestResponse, tags=["Document Ingestion"])
def ingest_uploaded_file(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    max_table_rows: Optional[int] = Form(None),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Upload and index a document via multipart/form-data."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    supported = set(pipeline.extractor.parsers.keys())
    if suffix not in supported:
        safe_print(f"[Upload Failed] File: {file.filename or 'unnamed'}")
        safe_print(f"[Upload Failed] Reason: Unsupported file type '{suffix}'.\n")
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

        indexed_count = pipeline.ingest_file(
            temp_file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_table_rows=max_table_rows,
        )
        total_points = pipeline.count_indexed_points()
        document_metadata = pipeline.last_ingest_metadata or {}

        return IngestResponse(
            status="success",
            message=(
                f"Upload successful: '{file.filename}' is indexed in the vector database "
                f"({indexed_count} chunks)."
            ),
            chunks_indexed=indexed_count,
            total_points=total_points,
            summary=document_metadata.get("summary"),
            document_metadata=document_metadata,
            documents=[document_metadata] if document_metadata else [],
        )
    except Exception as e:
        safe_print(f"[Upload Failed] File: {file.filename or 'unnamed'}")
        safe_print(f"[Upload Failed] Reason: {e}\n")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process upload: {str(e)}",
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/api/v1/ingest/directory", response_model=IngestResponse, tags=["Document Ingestion"])
def ingest_directory(
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
        indexed_count = pipeline.ingest_directory(
            request.directory_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_table_rows=request.max_table_rows,
        )
        total_points = pipeline.count_indexed_points()
        return IngestResponse(
            status="success",
            message=f"Indexed {indexed_count} total chunks from directory '{request.directory_path}'.",
            chunks_indexed=indexed_count,
            total_points=total_points,
            documents=pipeline.last_ingest_results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest directory: {str(e)}",
        )
