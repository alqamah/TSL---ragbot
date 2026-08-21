"""Terminal output capture: tees backend stdout into a thread-safe ring buffer.

All status lines printed to the backend terminal (ingest progress, upload
results, resets, vector store activity, LLM key rotation warnings, ...) are
recorded here so the frontend can stream them via GET /api/v1/logs.
"""

import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_LOCK = threading.Lock()
_BUFFER: deque = deque(maxlen=400)
_CURSOR = 0
_LAST_ACTIVITY_TS = 0.0

_original_stdout: Optional[object] = None
_installed = False


def _record_line(line: str) -> None:
    global _CURSOR, _LAST_ACTIVITY_TS
    text = _ANSI_RE.sub("", line.rstrip("\r\n"))
    if not text.strip():
        return
    with _LOCK:
        _CURSOR += 1
        _LAST_ACTIVITY_TS = time.time()
        _BUFFER.append(
            {
                "id": _CURSOR,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "message": text,
            }
        )


class TeeStream:
    """Wraps the real stdout, forwarding writes and recording full lines."""

    def __init__(self, original):
        self._original = original
        self._pending = ""

    def write(self, text):
        try:
            self._original.write(text)
        except Exception:
            pass
        if text:
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                _record_line(line)
        return len(text)

    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        if self._pending.strip():
            _record_line(self._pending)
            self._pending = ""

    def isatty(self):
        try:
            return self._original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._original, name)


def install() -> None:
    """Replace sys.stdout with a tee that mirrors terminal output into the buffer."""
    global _original_stdout, _installed
    if _installed:
        return
    _original_stdout = sys.stdout
    sys.stdout = TeeStream(_original_stdout)
    _installed = True


def get_logs(since: int = 0) -> Tuple[List[Dict], int, float]:
    """Return entries newer than `since`, the latest cursor, and last activity ts."""
    with _LOCK:
        entries = [dict(entry) for entry in _BUFFER if entry["id"] > since]
        latest_cursor = _BUFFER[-1]["id"] if _BUFFER else _CURSOR
        return entries, latest_cursor, _LAST_ACTIVITY_TS


def is_busy(idle_seconds: float = 5.0) -> bool:
    """True when a log line was recorded within the last `idle_seconds`."""
    with _LOCK:
        return (time.time() - _LAST_ACTIVITY_TS) < idle_seconds
