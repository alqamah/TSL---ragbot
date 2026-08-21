"""Document-level metadata generation for extracted and parsed documents."""

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from src.rag_engine.schema import DocumentChunk, ExtractedDocument


class DocumentMetadataGenerator:
    """Build stable, serializable metadata without requiring another model call."""

    _STOP_WORDS = {
        "about", "after", "again", "also", "before", "being", "between", "could",
        "from", "have", "into", "more", "other", "over", "such", "that", "their",
        "there", "these", "they", "this", "those", "through", "under", "using", "what",
        "when", "where", "which", "while", "with", "would", "your",
    }

    def generate(
        self,
        file_path: str,
        document: ExtractedDocument,
        parser_name: str,
    ) -> Dict[str, Any]:
        """Generate document metadata from the source file and extracted chunks."""
        extracted_text = document.full_text or "\n\n".join(
            chunk.content for chunk in document.chunks if chunk.content
        )
        normalized_text = re.sub(r"\s+", " ", extracted_text).strip()
        file_bytes = self._read_file_bytes(file_path)
        extension = os.path.splitext(document.file_name)[1].lower()
        chunk_types = sorted(
            {
                str(chunk.metadata.get("type"))
                for chunk in document.chunks
                if chunk.metadata.get("type")
            }
        )
        sheets = sorted(
            {
                str(chunk.metadata.get("sheet_name"))
                for chunk in document.chunks
                if chunk.metadata.get("sheet_name")
            },
            key=str.casefold,
        )
        paragraph_count = sum(
            1 for chunk in document.chunks if "paragraph_index" in chunk.metadata
        )
        table_count = sum(
            1
            for chunk in document.chunks
            if "table_index" in chunk.metadata or chunk.metadata.get("type") == "docx_table"
        )

        return {
            "metadata_version": 1,
            "document_id": hashlib.sha256(file_bytes).hexdigest(),
            "file_name": document.file_name,
            "file_extension": extension,
            "file_type": document.file_type,
            "parser": parser_name,
            "file_size_bytes": len(file_bytes),
            "content_sha256": hashlib.sha256(file_bytes).hexdigest(),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "extracted_character_count": len(extracted_text),
            "extracted_word_count": len(re.findall(r"\S+", extracted_text)),
            "chunk_count": len(document.chunks),
            "chunk_types": chunk_types,
            "sheet_names": sheets,
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "keywords": self._keywords(normalized_text),
            "summary": self._summary(normalized_text),
        }

    def generate_semantic(
        self,
        extracted_text: str,
        client_manager: Any = None,
        model_name: str = "",
    ) -> Dict[str, Any]:
        """Best-effort structured semantic metadata from the configured LLM."""
        if client_manager is None or not model_name or not extracted_text.strip():
            return {}

        prompt = f"""Read the extracted document text below and return ONLY valid JSON.
Do not infer facts that are not present in the text. Use empty strings or arrays when a
field cannot be determined.

JSON fields:
title (string), summary (string, maximum 500 characters), category (string),
topics (array of strings), equipment (array of strings), hazards (array of strings),
ppe (array of strings), standards (array of strings), language (string),
intended_audience (string).

Extracted document text:
{extracted_text[:12000]}
"""
        attempts = max(1, int(getattr(client_manager, "key_count", 1)))
        last_error = None
        for _attempt in range(attempts):
            try:
                chat = client_manager.current_client.chats.create(model=model_name)
                response = chat.send_message(prompt)
                parsed = self._parse_json_response(response.text if response else "")
                if not isinstance(parsed, dict):
                    raise ValueError("The metadata model did not return a JSON object.")
                semantic = self._normalize_semantic_metadata(parsed)
                if getattr(client_manager, "key_count", 1) > 1:
                    client_manager.rotate()
                return semantic
            except Exception as error:
                last_error = error
                is_key_error = getattr(client_manager, "is_key_rotation_error", None)
                if not callable(is_key_error) or not is_key_error(str(error)):
                    break
                if getattr(client_manager, "key_count", 1) > 1:
                    client_manager.rotate()

        warning = f"Semantic metadata unavailable: {last_error}"
        return {"metadata_warnings": [warning]}

    @staticmethod
    def attach_to_chunks(
        chunks: Iterable[DocumentChunk],
        document_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Add document metadata and stable chunk details to every chunk payload."""
        enriched_chunks = []
        for chunk_index, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata.copy() if chunk.metadata else {}
            indexed_document_metadata = document_metadata.copy()
            indexed_document_metadata.pop("status", None)
            indexed_document_metadata.pop("error", None)
            metadata.update(
                {
                    "document_id": document_metadata["document_id"],
                    "document_summary": document_metadata["summary"],
                    "document_metadata": indexed_document_metadata,
                    "chunk_index": chunk_index,
                    "chunk_word_count": len(re.findall(r"\S+", chunk.content)),
                }
            )
            enriched_chunks.append(DocumentChunk(content=chunk.content, metadata=metadata))
        return enriched_chunks

    @staticmethod
    def _read_file_bytes(file_path: str) -> bytes:
        with open(file_path, "rb") as source_file:
            return source_file.read()

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object was found in the metadata model response.")
        return json.loads(cleaned[start : end + 1])

    @staticmethod
    def _normalize_semantic_metadata(parsed: Dict[str, Any]) -> Dict[str, Any]:
        list_fields = {"topics", "equipment", "hazards", "ppe", "standards"}
        normalized: Dict[str, Any] = {}
        for field in ("title", "summary", "category", "language", "intended_audience"):
            value = parsed.get(field)
            if isinstance(value, str) and value.strip():
                clean_value = value.strip()
                if field == "summary" and len(clean_value) > 500:
                    clean_value = clean_value[:499].rsplit(" ", 1)[0] + "…"
                normalized[field] = clean_value
        for field in list_fields:
            value = parsed.get(field)
            if isinstance(value, list):
                normalized[field] = [str(item).strip() for item in value if str(item).strip()]
            else:
                normalized[field] = []
        return normalized

    def _keywords(self, text: str, limit: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u0900-\u097F]{3,}", text.lower())
        counts = Counter(
            token for token in tokens if token not in self._STOP_WORDS and not token.isdigit()
        )
        return [token for token, _count in counts.most_common(limit)]

    @staticmethod
    def _summary(text: str, max_length: int = 500) -> str:
        if not text:
            return "No extractable text was found in this document."
        sentences = [part.strip() for part in re.split(r"(?<=[.!?।])\s+", text) if part.strip()]
        summary = " ".join(sentences[:2]) if sentences else text
        if len(summary) > max_length:
            summary = summary[: max_length - 1].rsplit(" ", 1)[0] + "…"
        return summary
