"""Industrial SOP RAG Assistant package."""

import sys

# Ensure UTF-8 output across standard streams on all platforms (especially Windows)
for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if stream and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
