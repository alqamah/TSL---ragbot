import os
import streamlit as st
from frontend.api_client import RAGAPIClient


def render_sidebar(client: RAGAPIClient):
    """Render the sidebar controls, telemetry dashboard, and document ingestion tools."""
    with st.sidebar:
        st.markdown("### ⚙️ System & Controls")

        # 1. Connection & Health Check
        is_healthy, health_data = client.check_health()
        if is_healthy:
            st.markdown(
                '<div class="status-pill-online"><span class="pulse-dot"></span> API ONLINE (v'
                + str(health_data.get("version", "1.0"))
                + ")</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-pill-offline"><span class="pulse-dot"></span> BACKEND OFFLINE</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Could not connect to `{client.base_url}`. Make sure uvicorn is running.")

        st.markdown("---")

        # 2. Vector DB & System Telemetry
        st.markdown("#### 📊 Vector DB Telemetry")
        is_status_ok, status_data = client.get_status()

        if is_status_ok:
            st.markdown(
                f"""
                <div class="telemetry-card">
                    <div class="telemetry-row">
                        <span class="telemetry-label">Collection</span>
                        <span class="telemetry-val">{status_data.get('collection_name', 'industrial_sops_v2')}</span>
                    </div>
                    <div class="telemetry-row">
                        <span class="telemetry-label">Indexed Points</span>
                        <span class="telemetry-val">{status_data.get('indexed_points', 0)} chunks</span>
                    </div>
                    <div class="telemetry-row">
                        <span class="telemetry-label">Storage Mode</span>
                        <span class="telemetry-val">{status_data.get('storage_mode', 'local')}</span>
                    </div>
                    <div class="telemetry-row">
                        <span class="telemetry-label">LLM Model</span>
                        <span class="telemetry-val">{status_data.get('generation_model', 'gemini-3.7-flash')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Telemetry unavailable (Backend offline)")

        if st.button("🔄 Refresh Telemetry", use_container_width=True):
            st.rerun()

        st.markdown("---")

        # 3. Document Ingestion Panel
        st.markdown("#### 📥 Document Ingestion")
        ingest_tab1, ingest_tab2 = st.tabs(["Upload File", "Server Path"])

        with ingest_tab1:
            uploaded_file = st.file_uploader(
                "Upload SOP Document",
                type=["xlsx", "xls", "docx", "doc", "txt", "csv"],
                help="Upload industrial SOPs in Excel, Word, Text, or CSV formats.",
            )
            if uploaded_file is not None:
                if st.button("🚀 Index Uploaded File", use_container_width=True, type="primary"):
                    with st.spinner(f"Parsing and indexing '{uploaded_file.name}'..."):
                        file_bytes = uploaded_file.getvalue()
                        ok, resp = client.upload_file(file_bytes, uploaded_file.name)
                        if ok:
                            st.success(f"Indexed {resp.get('chunks_indexed', 0)} chunks! (Total: {resp.get('total_points', 0)})")
                            st.rerun()
                        else:
                            st.error(resp.get("error", "Upload failed"))

        with ingest_tab2:
            default_dir = os.path.join(os.getcwd(), "io", "SOP")
            dir_path = st.text_input("Directory Path", value=default_dir)
            if st.button("📂 Ingest Directory", use_container_width=True):
                with st.spinner(f"Scanning & indexing documents in '{dir_path}'..."):
                    ok, resp = client.ingest_directory(dir_path)
                    if ok:
                        st.success(f"Indexed {resp.get('chunks_indexed', 0)} chunks!")
                        st.rerun()
                    else:
                        st.error(resp.get("error", "Directory ingestion failed"))

        st.markdown("---")

        # 4. Search Hyperparameters
        st.markdown("#### 🎛️ Retrieval Settings")
        top_k = st.slider("Context Chunks (top_k)", min_value=1, max_value=10, value=3, step=1)
        show_sources = st.checkbox("Include Source Citations", value=True)

        return {
            "top_k": top_k,
            "show_sources": show_sources,
            "is_backend_online": is_healthy,
        }
