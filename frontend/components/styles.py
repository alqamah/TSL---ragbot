import streamlit as st


def apply_custom_styles():
    """Inject custom CSS for an ultra-modern, professional industrial UI theme."""
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Banner Header */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0369a1 100%);
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
    }

    .hero-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-badge {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 3px 10px;
        border-radius: 20px;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        font-weight: 600;
    }

    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        margin-bottom: 0px;
        line-height: 1.5;
    }

    /* Telemetry Card in Sidebar */
    .telemetry-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 15px;
    }

    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 13px;
    }

    .telemetry-row:last-child {
        border-bottom: none;
    }

    .telemetry-label {
        color: #94a3b8;
    }

    .telemetry-val {
        font-weight: 600;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Status Indicator Pills */
    .status-pill-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }

    .status-pill-offline {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: currentColor;
    }

    /* Metric Chips */
    .latency-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 8px;
        margin-top: 8px;
    }

    /* Source Citation Card */
    .citation-card {
        background: rgba(30, 41, 59, 0.5);
        border-left: 3px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 13px;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 6px;
    }

    .score-badge {
        background: rgba(14, 165, 233, 0.2);
        color: #38bdf8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
    }

    .citation-content {
        font-size: 12.5px;
        color: #cbd5e1;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
