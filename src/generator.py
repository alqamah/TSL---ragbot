import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vector_store import VectorStoreManager

load_dotenv()


class RAGGenerator:
    """RAG Synthesis and Generation Engine with fine-grained layer latency logging."""

    def __init__(
        self,
        collection_name: str = "industrial_sops_gemini",
        generation_model: Optional[str] = None,
        embedding_model: str = "models/gemini-embedding-001",
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = (
            generation_model
            or os.getenv("GEMINI_GENERATION_MODEL")
            or "gemini-3.7-flash"
        )

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing!")

        self.ai_client = genai.Client(api_key=api_key)
        self.vector_store = vector_store or VectorStoreManager(
            collection_name=self.collection_name,
            model_name=self.embedding_model,
        )

    def retrieve_with_metrics(
        self, query: str, top_k: int = 3
    ) -> Dict[str, Any]:
        """Retrieve relevant context while benchmarking vector search layers."""
        t_start = time.perf_counter()

        # Step 1: Query embedding
        t_embed_start = time.perf_counter()
        query_vector = self.vector_store.get_embedding(query)
        embed_duration = (time.perf_counter() - t_embed_start) * 1000

        # Step 2: Vector database lookup
        t_search_start = time.perf_counter()
        if hasattr(self.vector_store.qdrant_client, "query_points"):
            hits = self.vector_store.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            ).points
        else:
            hits = self.vector_store.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
            )
        qdrant_duration = (time.perf_counter() - t_search_start) * 1000

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

        total_retrieval_duration = (time.perf_counter() - t_start) * 1000

        return {
            "sources": results,
            "metrics": {
                "query_embedding_ms": round(embed_duration, 2),
                "qdrant_search_ms": round(qdrant_duration, 2),
                "total_retrieval_ms": round(total_retrieval_duration, 2),
            },
        }

    def generate_answer(
        self,
        query: str,
        top_k: int = 3,
        max_retries: int = 3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute full RAG generation with per-layer latency tracking."""
        total_start = time.perf_counter()

        # Layer 1 & 2: Retrieval
        retrieval_res = self.retrieve_with_metrics(query=query, top_k=top_k)
        contexts = retrieval_res["sources"]
        retrieval_metrics = retrieval_res["metrics"]

        # Layer 3: Prompt Construction
        t_prompt_start = time.perf_counter()
        formatted_context_blocks = []
        for idx, item in enumerate(contexts, 1):
            source = item.get("source") or "Unknown Document"
            meta = item.get("metadata", {})
            sheet = meta.get("sheet", "")
            row_range = meta.get("row_range", "")
            section = meta.get("section", "")

            location_details = []
            if sheet:
                location_details.append(f"Sheet: {sheet}")
            if section:
                location_details.append(f"Section: {section}")
            if row_range:
                location_details.append(f"Rows: {row_range}")

            loc_str = f" ({', '.join(location_details)})" if location_details else ""
            formatted_context_blocks.append(
                f"[Source {idx}]: {source}{loc_str} (Similarity Score: {item.get('score', 0):.3f})\n"
                f"{item.get('content', '').strip()}"
            )

        context_text = "\n\n---\n\n".join(formatted_context_blocks)

        prompt = f"""You are an expert Industrial Standard Operating Procedure (SOP) and Technical Safety Assistant.
Your task is to provide accurate, reliable, and step-by-step answers based strictly on the provided context below.

Context:
{context_text}

Question:
{query}

Instructions:
1. Answer strictly using the provided context. If the context does not contain sufficient details to answer, state clearly what is missing.
2. If the user's question is in Hindi (or Hinglish), respond in clear and professional Hindi. If the question is in English, respond in English.
3. Structure your response clearly with bullet points or numbered steps for operational/safety procedures.
4. Mention which source/document your answer is derived from.

Answer:"""

        prompt_build_duration = (time.perf_counter() - t_prompt_start) * 1000

        # Layer 4: LLM Generation (Synthesis)
        candidate_models = [self.generation_model]
        for fallback in ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        active_model = None
        answer_text = ""
        llm_duration = 0.0
        last_error = None

        t_gen_start = time.perf_counter()
        for model in candidate_models:
            for attempt in range(max_retries):
                try:
                    response = self.ai_client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    active_model = model
                    answer_text = response.text.strip() if response.text else "No response generated."
                    llm_duration = (time.perf_counter() - t_gen_start) * 1000
                    break
                except Exception as err:
                    err_str = str(err)
                    last_error = err
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait_time = 2 * (attempt + 1)
                        if verbose:
                            print(f"[{model}] Temporary load ({err_str[:40]}...). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    break
            if active_model:
                break

        if not active_model:
            raise RuntimeError(f"Failed to generate answer across candidate models: {last_error}")

        total_elapsed = (time.perf_counter() - total_start) * 1000

        metrics = {
            "embedding_model": self.embedding_model,
            "query_embedding_ms": retrieval_metrics["query_embedding_ms"],
            "qdrant_search_ms": retrieval_metrics["qdrant_search_ms"],
            "total_retrieval_ms": retrieval_metrics["total_retrieval_ms"],
            "prompt_construction_ms": round(prompt_build_duration, 2),
            "final_llm_model": active_model,
            "llm_generation_ms": round(llm_duration, 2),
            "total_e2e_latency_ms": round(total_elapsed, 2),
        }

        if verbose:
            self._log_layer_summary(metrics, contexts)

        return {
            "query": query,
            "answer": answer_text,
            "sources": contexts,
            "metrics": metrics,
        }

    @staticmethod
    def _log_layer_summary(metrics: Dict[str, Any], sources: List[Dict[str, Any]]):
        """Format and print timing breakdown and output source telemetry."""
        print("\n" + "=" * 75)
        print("                   RAG PIPELINE EXECUTION TELEMETRY                    ")
        print("=" * 75)
        print(f"1. Retrieval Layer:")
        print(f"   - Embedding Model    : {metrics['embedding_model']}")
        print(f"   - Query Embedding    : {metrics['query_embedding_ms']:.2f} ms")
        print(f"   - Qdrant Vector Search: {metrics['qdrant_search_ms']:.2f} ms")
        print(f"   - Total Retrieval    : {metrics['total_retrieval_ms']:.2f} ms")
        print("-" * 75)
        print(f"2. Generation Layer:")
        print(f"   - Prompt Assembly    : {metrics['prompt_construction_ms']:.2f} ms")
        print(f"   - Final LLM Model    : {metrics['final_llm_model']}")
        print(f"   - LLM Synthesis Time : {metrics['llm_generation_ms']:.2f} ms ({metrics['llm_generation_ms']/1000:.2f}s)")
        print("-" * 75)
        print(f"3. Overall Performance:")
        print(f"   - Total E2E Latency  : {metrics['total_e2e_latency_ms']:.2f} ms ({metrics['total_e2e_latency_ms']/1000:.2f}s)")
        print("=" * 75)
        print("4. Retrieved Source Attributions:")
        for idx, src in enumerate(sources, 1):
            doc = src.get("source") or "Unknown"
            score = src.get("score", 0.0)
            meta = src.get("metadata", {})
            sheet = meta.get("sheet", "N/A")
            rows = meta.get("row_range", "N/A")
            print(f"   [{idx}] File: {doc} | Sheet: {sheet} | Rows: {rows} | Similarity: {score:.4f}")
        print("=" * 75 + "\n")

    def close(self) -> None:
        self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
