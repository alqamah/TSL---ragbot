# chunker.py
from typing import List
from src.schema import DocumentChunk, ExtractedDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SmartChunker:
    def __init__(self, text_chunk_size: int = 600, text_chunk_overlap: int = 100, max_table_rows: int = 4):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_chunk_size,
            chunk_overlap=text_chunk_overlap,
            separators=["\n\n", "\n", "।", ".", " ", ""] # Included Devanagari danda '।' for Hindi sentences!
        )
        self.max_table_rows = max_table_rows

    def process_document(self, doc: ExtractedDocument) -> List[DocumentChunk]:
        final_chunks: List[DocumentChunk] = []

        for raw_chunk in doc.chunks:
            chunk_type = raw_chunk.metadata.get("type", "")

            if chunk_type in ["excel", "docx_table"]:
                # Table chunking strategy: split long table texts into smaller row groups
                sub_chunks = self._chunk_table_text(raw_chunk)
                final_chunks.extend(sub_chunks)
            else:
                # Text chunking strategy: Standard semantic text splitter
                split_texts = self.text_splitter.split_text(raw_chunk.content)
                for idx, t in enumerate(split_texts):
                    if not t.strip():
                        continue
                    meta = raw_chunk.metadata.copy()
                    meta["sub_chunk_id"] = idx
                    final_chunks.append(DocumentChunk(content=t, metadata=meta))

        return final_chunks

    def _chunk_table_text(self, table_chunk: DocumentChunk) -> List[DocumentChunk]:
        """Splits long table text into smaller contextual blocks while preserving header context."""
        lines = table_chunk.content.split("\n")
        chunk_type = table_chunk.metadata.get("type", "")

        # Determine header line count based on table type / markdown format
        if chunk_type == "excel" and len(lines) >= 4 and lines[0].startswith("###") and lines[2].startswith("|"):
            # Markdown table: Line 0 (Sheet title), Line 1 (blank), Line 2 (Headers), Line 3 (Separator)
            header_count = 4
        elif chunk_type == "docx_table" and len(lines) >= 2 and lines[0].startswith("---"):
            # Docx table: Line 0 (Table title), Line 1 (Header row)
            header_count = 2
        else:
            header_count = 2

        if len(lines) <= (header_count + self.max_table_rows):
            return [table_chunk]

        headers = lines[:header_count]
        data_rows = lines[header_count:]

        sub_chunks = []
        for idx, i in enumerate(range(0, len(data_rows), self.max_table_rows)):
            batch_rows = data_rows[i : i + self.max_table_rows]
            chunk_text = "\n".join(headers + batch_rows)
            
            meta = table_chunk.metadata.copy()
            meta["sub_chunk_id"] = idx
            meta["row_range"] = f"{i+1}-{i+len(batch_rows)}"
            
            sub_chunks.append(DocumentChunk(content=chunk_text, metadata=meta))

        return sub_chunks