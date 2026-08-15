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

    pipeline = RAGPipeline(
        collection_name=collection_name,
        generation_model=generation_model,
        embedding_model=embedding_model,
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
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)
