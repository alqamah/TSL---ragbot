import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.routes import router
from src.rag_engine.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle RAGPipeline startup and shutdown."""
    print("[API] Initializing RAGPipeline instance...")
    collection_name = os.getenv("QDRANT_COLLECTION", "industrial_sops_gemini")
    generation_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.7-flash")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    db_mode = os.getenv("QDRANT_MODE")

    pipeline = RAGPipeline(
        collection_name=collection_name,
        generation_model=generation_model,
        embedding_model=embedding_model,
        db_mode=db_mode,
    )
    app.state.pipeline = pipeline
    print(f"[API] RAGPipeline initialized (Storage mode: {pipeline.vector_store.mode}).")

    yield

    print("[API] Shutting down RAGPipeline...")
    pipeline.close()
    print("[API] Cleanup complete.")


app = FastAPI(
    title="Industrial SOP RAG Assistant API",
    description=(
        "Production-ready REST API for multilingual Industrial SOP retrieval and "
        "grounded generation powered by Google Gemini and Qdrant."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for external frontend or microservice integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    return JSONResponse(
        content={
            "service": "Industrial SOP RAG Assistant API",
            "version": "1.0.0",
            "documentation": "/docs",
            "health": "/health",
            "status": "/api/v1/status",
        }
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Industrial SOP RAG FastAPI Server")
    parser.add_argument("--host", type=str, default=os.getenv("API_HOST", "0.0.0.0"), help="Host IP (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", 8000)), help="Port (default: 8000)")
    parser.add_argument(
        "--db-mode",
        "--mode",
        dest="db_mode",
        type=str,
        choices=["auto", "cloud", "local", "server"],
        default=None,
        help="Vector DB storage mode: 'cloud', 'local', 'server', 'auto'.",
    )
    parser.add_argument("--cloud", action="store_true", help="Force Qdrant Cloud online vector DB mode.")
    parser.add_argument("--local", action="store_true", help="Force local embedded vector DB storage.")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload on code change.")
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload.")

    cli_args, _ = parser.parse_known_args()

    if cli_args.cloud:
        os.environ["QDRANT_MODE"] = "cloud"
    elif cli_args.local:
        os.environ["QDRANT_MODE"] = "local"
    elif cli_args.db_mode:
        os.environ["QDRANT_MODE"] = cli_args.db_mode

    uvicorn.run("src.api.app:app", host=cli_args.host, port=cli_args.port, reload=cli_args.reload)
