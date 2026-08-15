# schema.py
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class DocumentChunk:
    """Represents a single extracted section, paragraph, or table."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractedDocument:
    """The master output object returned by any parser."""
    file_name: str
    file_type: str
    chunks: List[DocumentChunk]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
