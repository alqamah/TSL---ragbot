# chunker.py
import pandas as pd
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
                    meta = raw_chunk.metadata.copy()
                    meta["sub_chunk_id"] = idx
                    final_chunks.append(DocumentChunk(content=t, metadata=meta))

        return final_chunks

    def _chunk_table_text(self, table_chunk: DocumentChunk) -> List[DocumentChunk]:
        """Splits long table text into smaller contextual blocks."""
        lines = table_chunk.content.split("\n")
        
        # If the table is short enough, keep it as is
        if len(lines) <= (self.max_table_rows + 4): 
            return [table_chunk]

        # Extract Header (Usually first 2 lines in Markdown tables)
        headers = lines[:2]
        data_rows = lines[2:]

        sub_chunks = []
        # Group rows into batches of `max_table_rows`
        for i in range(0, len(data_rows), self.max_table_rows):
            batch_rows = data_rows[i : i + self.max_table_rows]
            
            # Re-build table with headers + row batch
            chunk_text = "\n".join(headers + batch_rows)
            
            meta = table_chunk.metadata.copy()
            meta["row_range"] = f"{i+1}-{i+len(batch_rows)}"
            
            sub_chunks.append(DocumentChunk(content=chunk_text, metadata=meta))

        return sub_chunks