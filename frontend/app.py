import os
import sys
import streamlit as st

# Ensure frontend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.api_client import RAGAPIClient
from frontend.components.styles import apply_custom_styles
from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat_history, render_message_metrics, render_sources

# Streamlit Page Configuration
st.set_page_config(
    page_title="TSL-Infra RAG Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Modern Custom CSS
apply_custom_styles()

# Initialize API Client (defaulting to local backend or environment override)
# BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_URL = os.getenv("RAG_BACKEND_URL", "https://ragbot-backend-q5wj.onrender.com")

client = RAGAPIClient(base_url=BACKEND_URL)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# Render Sidebar (Telemetry, Health, Ingestion, Parameters)
settings = render_sidebar(client)

# Main Dashboard & Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            <span> TSL-Infra RAG Assistant</span>
            <span class="hero-badge">Google Gemini + Qdrant</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# # Quick Prompts / Demo Chips
# st.markdown("##### 💡 Multilingual Quick Queries:")
# col1, col2, col3 = st.columns(3)

# with col1:
#     if st.button("🇬🇧 Compressor Maintenance Checklist", use_container_width=True):
#         st.session_state.quick_prompt = "What safety precautions and PPEs are mandatory before starting air compressor maintenance?"

# with col2:
#     if st.button("🇮🇳 कंप्रेसर चालू करने की प्रक्रिया", use_container_width=True):
#         st.session_state.quick_prompt = "कंप्रेसर चालू करने से पहले कौन-कौन से सुरक्षा उपाय और जांच करना जरूरी है?"

# with col3:
#     if st.button("🗣️ Hinglish: Tower Light Isolation", use_container_width=True):
#         st.session_state.quick_prompt = "Tower light servicing karte waqt electrical hazard se bachne ke liye positive isolation kaise kare?"

# st.markdown("---")

# Render Existing Chat History
render_chat_history()

# Handle User Input (via text input or quick prompt button)
user_query = None
if st.session_state.quick_prompt:
    user_query = st.session_state.quick_prompt
    st.session_state.quick_prompt = None
else:
    user_query = st.chat_input("Ask a question about the uploaded files (English / हिंदी)...")

if user_query:
    # Append User Message to State
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👷‍♂️"):
        st.markdown(user_query)

    # Process via RAG API
    with st.chat_message("assistant", avatar="🤖"):
        if not settings.get("is_backend_online", True):
            error_msg = f"⚠️ **Backend API is currently unreachable at `{client.base_url}`.** If hosted on a free tier, it may be waking up from sleep mode (takes ~30s). Please wait a moment and retry."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("Searching SOP knowledge base and synthesizing grounded response with Gemini..."):
                success, response_data = client.query(
                    query_text=user_query,
                    top_k=settings.get("top_k", 3),
                    show_sources=settings.get("show_sources", True),
                    model=settings.get("model"),
                )

                if success:
                    answer = response_data.get("answer", "No answer provided.")
                    sources = response_data.get("sources", [])
                    metrics = response_data.get("metrics", {})

                    st.markdown(answer)
                    render_message_metrics(metrics)
                    if sources:
                        render_sources(sources)

                    # Append to state
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "metrics": metrics,
                        }
                    )
                else:
                    error_msg = f"❌ **Query Execution Failed:** {response_data.get('error', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
