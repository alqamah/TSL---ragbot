import os

import streamlit as st

from frontend.api_client import RAGAPIClient

MAX_STREAM_LINES = 400


def _init_status_state():
    if "backend_logs" not in st.session_state:
        st.session_state.backend_logs = []
    if "backend_log_cursor" not in st.session_state:
        st.session_state.backend_log_cursor = 0
    if "gemini_models" not in st.session_state:
        st.session_state.gemini_models = []


@st.fragment
def render_status_stream(client: RAGAPIClient):
    """Show backend terminal output inside the Current Status dropdown.

    No background polling: the stream only updates when a full app rerun is
    triggered manually (e.g. via the 🔄 Refresh button or any other action).
    """
    _init_status_state()

    ok, data = client.get_logs(st.session_state.backend_log_cursor)
    busy = False

    if ok:
        remote_cursor = int(data.get("cursor", 0))
        if remote_cursor < st.session_state.backend_log_cursor:
            # Backend restarted and reset its cursor: resync full history.
            st.session_state.backend_logs = []
            st.session_state.backend_log_cursor = 0
            ok, data = client.get_logs(0)

        if ok:
            busy = bool(data.get("busy", False))
            new_entries = data.get("logs", [])
            if new_entries:
                st.session_state.backend_logs.extend(new_entries)
                st.session_state.backend_logs = st.session_state.backend_logs[-MAX_STREAM_LINES:]
            st.session_state.backend_log_cursor = int(data.get("cursor", st.session_state.backend_log_cursor))

    pulse = "🟢 LIVE" if busy else "⚪ IDLE"
    title = f"{pulse} · Backend Terminal"
    if st.session_state.backend_logs:
        title += f" ({len(st.session_state.backend_logs)} lines)"

    with st.expander(title, expanded=False):
        if not ok:
            st.caption(f"Terminal stream unavailable — cannot reach `{client.base_url}`.")
            return

        st.caption("Mirrors the backend terminal · updates on 🔄 Refresh click (no background pinging)")

        if st.session_state.backend_logs:
            lines = [f"{entry['ts']} | {entry['message']}" for entry in st.session_state.backend_logs]
            st.code("\n".join(lines), language="text")
        else:
            st.caption("No backend output yet — waiting for activity (ingest, query, reset)...")


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
            llm_models = status_data.get("llm_models", [])
            model_list = ", ".join(llm_models) if llm_models else status_data.get("generation_model", "gemini-3.7-flash")
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
                        <span class="telemetry-label">LLM Models</span>
                        <span class="telemetry-val">{model_list}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Telemetry unavailable (Backend offline)")

        # 2b. Live Backend Terminal Stream (Current Status)
        st.markdown("#### 🛰️ Current Status")
        render_status_stream(client)

        files_ok, files_data = client.list_indexed_files()
        if files_ok:
            indexed_files = files_data.get("files", [])
            with st.expander(f"Indexed Files ({files_data.get('total_files', len(indexed_files))})", expanded=False):
                if indexed_files:
                    for file_name in indexed_files:
                        st.write(file_name)
                else:
                    st.caption("No files are indexed yet.")

        col_ref, col_reset = st.columns(2)
        with col_ref:
            if st.button("🔄 Refresh", use_container_width=True, help="Re-fetch telemetry, backend terminal output, and the list of Gemini models available to the API key."):
                _init_status_state()
                models_ok, models_data = client.list_models(force=True)
                if models_ok:
                    st.session_state.gemini_models = models_data.get("models", [])
                else:
                    st.toast(f"Model list unavailable: {models_data.get('error', 'unknown error')}", icon="⚠️")
                st.rerun()
        with col_reset:
            if st.button("🗑️ Reset DB", use_container_width=True, help="Wipe all indexed points and reset the collection."):
                with st.spinner("Resetting vector database..."):
                    ok, resp = client.reset_database()
                    if ok:
                        st.session_state.messages = []
                        st.toast("✅ Database reset! All vectors cleared.", icon="🗑️")
                        st.rerun()
                    else:
                        st.error(resp.get("error", "Reset failed"))

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

            upload_chunk_size = st.slider(
                "Chunk Size (characters)",
                min_value=100,
                max_value=2000,
                value=600,
                step=50,
                help="Target text length per semantic chunk. Smaller values (300-500) yield finer retrieval precision; larger values (800-1500) preserve broader section context.",
            )

            with st.expander("⚙️ Advanced Chunking Settings", expanded=False):
                upload_chunk_overlap = st.slider(
                    "Chunk Overlap (characters)",
                    min_value=0,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Character overlap between consecutive chunks to preserve contextual flow.",
                )
                upload_table_rows = st.slider(
                    "Table Rows per Chunk",
                    min_value=1,
                    max_value=20,
                    value=4,
                    step=1,
                    help="Number of table rows per chunk. Table headers are preserved automatically across splits.",
                )

            if uploaded_file is not None:
                if st.button("🚀 Index Uploaded File", use_container_width=True, type="primary"):
                    with st.spinner(f"Parsing and indexing '{uploaded_file.name}' (chunk_size={upload_chunk_size})..."):
                        file_bytes = uploaded_file.getvalue()
                        ok, resp = client.upload_file(
                            file_bytes,
                            uploaded_file.name,
                            chunk_size=upload_chunk_size,
                            chunk_overlap=upload_chunk_overlap,
                            max_table_rows=upload_table_rows,
                        )
                        if ok:
                            st.toast(f"✅ Indexed {resp.get('chunks_indexed', 0)} chunks!", icon="🎉")
                            st.success(
                                resp.get(
                                    "message",
                                    f"Upload successful: '{uploaded_file.name}' is indexed in the vector database.",
                                )
                            )
                            if resp.get("summary"):
                                st.info(f"Summary: {resp['summary']}")
                            document_metadata = resp.get("document_metadata", {})
                            if document_metadata:
                                with st.expander("Uploaded file metadata", expanded=True):
                                    st.json(document_metadata)
                        else:
                            st.error(resp.get("error", "Upload failed"))

        with ingest_tab2:
            default_dir = os.path.join(os.getcwd(), "io", "SOP")
            dir_path = st.text_input("Directory Path", value=default_dir)
            if st.button("📂 Ingest Directory", use_container_width=True):
                with st.spinner(f"Scanning & indexing documents in '{dir_path}'..."):
                    ok, resp = client.ingest_directory(
                        dir_path,
                        chunk_size=upload_chunk_size,
                        chunk_overlap=upload_chunk_overlap,
                        max_table_rows=upload_table_rows,
                    )
                    if ok:
                        st.toast(f"✅ Directory indexed ({resp.get('chunks_indexed', 0)} chunks)!", icon="🎉")
                        st.success(f"Indexed {resp.get('chunks_indexed', 0)} chunks!")
                        documents = resp.get("documents", [])
                        if documents:
                            with st.expander("Uploaded document metadata", expanded=False):
                                st.json(documents)
                        st.rerun()
                    else:
                        st.error(resp.get("error", "Directory ingestion failed"))

        st.markdown("---")

        # 4. Search Hyperparameters
        st.markdown("#### 🎛️ Retrieval Settings")

        _init_status_state()
        model_options = ["auto (pipeline default)"] + [
            m["name"] for m in st.session_state.gemini_models
        ]
        selected_model = st.selectbox(
            "🤖 Gemini Model",
            options=model_options,
            index=0,
            help=(
                "Model used to synthesize answers. 'auto' lets the pipeline use its default "
                "with automatic fallback. Click 🔄 Refresh to fetch the list of models "
                "available to the configured API key."
            ),
        )
        if not st.session_state.gemini_models:
            st.caption("Click 🔄 Refresh above to list all models verified available via the Gemini API (takes ~30s on first fetch).")

        top_k = st.slider("Context Chunks (top_k)", min_value=1, max_value=10, value=3, step=1)
        show_sources = st.checkbox("Include Source Citations", value=True)

        return {
            "top_k": top_k,
            "show_sources": show_sources,
            "model": None if selected_model.startswith("auto") else selected_model,
            "is_backend_online": is_healthy,
        }
