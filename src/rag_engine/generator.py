import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_engine.vector_store import GeminiClientManager, VectorStoreManager

load_dotenv()


class RAGGenerator:
    """RAG Synthesis and Generation Engine with fine-grained layer latency logging and persistent file logging."""

    def __init__(
        self,
        collection_name: str = "industrial_sops_gemini",
        generation_model: Optional[str] = None,
        embedding_model: str = "models/gemini-embedding-001",
        vector_store: Optional[VectorStoreManager] = None,
        client_manager: Optional[GeminiClientManager] = None,
        log_dir: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = (
            generation_model
            or os.getenv("GEMINI_GENERATION_MODEL")
            or "gemini-3.7-flash"
        )
        self.log_dir = log_dir or os.path.join(PROJECT_ROOT, "log")
        os.makedirs(self.log_dir, exist_ok=True)

        self.vector_store = vector_store or VectorStoreManager(
            collection_name=self.collection_name,
            model_name=self.embedding_model,
            client_manager=client_manager,
        )
        self.client_manager = (
            client_manager
            or getattr(self.vector_store, "client_manager", None)
            or GeminiClientManager()
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
        """Execute full RAG generation with per-layer latency tracking and automated persistent logging."""
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

        # Layer 4: LLM Generation (Synthesis) with Gemini fallback pool
        candidate_models = [self.generation_model]
        for fallback in ["gemini-2.0-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        active_model = None
        answer_text = ""
        llm_duration = 0.0
        last_error = None

        t_gen_start = time.perf_counter()
        for model in candidate_models:
            for attempt in range(max_retries):
                client = self.client_manager.current_client
                try:
                    response = client.models.generate_content(
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
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        masked_key = self.client_manager.current_key_masked
                        self.client_manager.rotate()
                        next_key = self.client_manager.current_key_masked
                        if verbose:
                            print(f"[{model}] Rate limit (429) on Key [{masked_key}]. Rotating to Key [{next_key}]...")
                        time.sleep(0.5)
                        continue
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        wait_time = 2 * (attempt + 1)
                        if verbose:
                            print(f"[{model}] Temporary server load ({err_str[:40]}...). Retrying in {wait_time}s...")
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

        # 1. Console telemetry summary
        if verbose:
            self._log_layer_summary(metrics, contexts)

        # 2. Persistent file logging to log/ folder
        self._write_persistent_log(query, answer_text, metrics, contexts)

        return {
            "query": query,
            "answer": answer_text,
            "sources": contexts,
            "metrics": metrics,
        }

    def _write_persistent_log(
        self,
        query: str,
        answer: str,
        metrics: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> None:
        """Write human-readable and structured logs to the log directory."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        iso_str = datetime.now().isoformat()

        # Write to log/rag_pipeline.log
        readable_log_file = os.path.join(self.log_dir, "rag_pipeline.log")
        try:
            with open(readable_log_file, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"TIMESTAMP : {now_str}\n")
                f.write(f"QUESTION  : {query}\n")
                f.write("-" * 80 + "\n")
                f.write("LAYER TIMINGS & MODELS:\n")
                f.write(f"  * Query Embedding   : {metrics['query_embedding_ms']} ms  (Model: {metrics['embedding_model']})\n")
                f.write(f"  * Qdrant Lookup     : {metrics['qdrant_search_ms']} ms\n")
                f.write(f"  * Total Retrieval   : {metrics['total_retrieval_ms']} ms\n")
                f.write(f"  * Prompt Assembly   : {metrics['prompt_construction_ms']} ms\n")
                f.write(f"  * Final LLM Model   : {metrics['final_llm_model']}\n")
                f.write(f"  * LLM Synthesis     : {metrics['llm_generation_ms']} ms ({metrics['llm_generation_ms']/1000:.2f}s)\n")
                f.write(f"  * Total Latency     : {metrics['total_e2e_latency_ms']} ms ({metrics['total_e2e_latency_ms']/1000:.2f}s)\n")
                f.write("-" * 80 + "\n")
                f.write("RETRIEVED SOURCES:\n")
                for idx, src in enumerate(sources, 1):
                    doc = src.get("source") or "Unknown"
                    score = src.get("score", 0.0)
                    meta = src.get("metadata", {})
                    sheet = meta.get("sheet", "N/A")
                    rows = meta.get("row_range", "N/A")
                    f.write(f"  [{idx}] {doc} | Sheet: {sheet} | Rows: {rows} | Similarity: {score:.4f}\n")
                f.write("-" * 80 + "\n")
                f.write(f"RESPONSE:\n{answer}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"[Warning] Failed to write readable log: {e}")

        # Write to log/runs.jsonl
        jsonl_log_file = os.path.join(self.log_dir, "runs.jsonl")
        try:
            record = {
                "timestamp": iso_str,
                "query": query,
                "answer": answer,
                "metrics": metrics,
                "sources": [
                    {
                        "source": s.get("source"),
                        "score": s.get("score"),
                        "metadata": s.get("metadata", {}),
                    }
                    for s in sources
                ],
            }
            with open(jsonl_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Warning] Failed to write JSONL log: {e}")

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
