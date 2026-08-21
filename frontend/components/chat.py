from typing import Any, Dict, List
import streamlit as st


def render_message_metrics(metrics: Dict[str, Any]):
    """Render latency breakdown pills for query embedding, vector search, LLM generation, and active API key."""
    if not metrics:
        return

    q_emb = metrics.get("query_embedding_ms", 0.0)
    q_search = metrics.get("qdrant_search_ms", 0.0)
    llm_gen = metrics.get("llm_generation_ms", 0.0)
    total = metrics.get("total_e2e_latency_ms", 0.0)
    model_name = metrics.get("final_llm_model") or "gemini-3.7-flash"
    api_key_used = (
        metrics.get("gemini_api_key_used")
        or metrics.get("gemini_key_used")
        or metrics.get("api_key_used")
        or "Key 1"
    )

    top_k_val = metrics.get("top_k")
    top_k_badge = f'<span class="latency-pill">🎯 Top-K: {top_k_val}</span>' if top_k_val is not None else ""

    st.markdown(
        f"""
        <div style="margin-top: 10px; margin-bottom: 6px;">
            <span class="latency-pill">⚡ Total: {total:.0f} ms</span>
            {top_k_badge}
            <span class="latency-pill">🔍 Qdrant: {q_search:.1f} ms</span>
            <span class="latency-pill">🧠 Gemini ({model_name}): {llm_gen:.0f} ms</span>
            <span class="latency-pill">📐 Embedding: {q_emb:.1f} ms</span>
            <span class="latency-pill">🔑 Gemini API Key: {api_key_used}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: List[Dict[str, Any]]):
    """Render collapsible source citation drawers with similarity scores and chunk previews."""
    if not sources:
        return

    with st.expander(f"📑 Grounded Sources ({len(sources)} Chunks Retreived)", expanded=False):
        for i, src in enumerate(sources, 1):
            source_file = src.get("source", "Unknown Document")
            score = src.get("score", 0.0)
            content = src.get("content", "")
            meta = src.get("metadata", {})

            meta_details = []
            if "sheet_name" in meta:
                meta_details.append(f"Sheet: {meta['sheet_name']}")
            if "row_range" in meta:
                meta_details.append(f"Rows: {meta['row_range']}")
            elif "row" in meta:
                meta_details.append(f"Row: {meta['row']}")

            meta_str = f" | {', '.join(meta_details)}" if meta_details else ""

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <span>📄 Chunk #{i}: <strong>{source_file}</strong>{meta_str}</span>
                        <span class="score-badge">Similarity: {score:.3f}</span>
                    </div>
                    <div class="citation-content">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_chat_history():
    """Render all previous chat messages stored in session state."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👷‍♂️" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("metrics"):
                render_message_metrics(msg["metrics"])
            if msg.get("sources"):
                render_sources(msg["sources"])
