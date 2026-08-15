# engine.py
import os
from typing import Optional
from src.rag_engine.parsers import TextParser, DocxParser, DocParser, ExcelParser
from src.rag_engine.chunker import SmartChunker
from src.rag_engine.schema import ExtractedDocument

class DataExtractionEngine:
    def __init__(self, chunker: Optional[SmartChunker] = None):
        self.parsers = {
            ".txt": TextParser,
            ".docx": DocxParser,
            ".doc": DocParser,
            ".xlsx": ExcelParser,
            ".xls": ExcelParser,
            ".csv": ExcelParser,
        }
        self.chunker = chunker or SmartChunker()

    def extract(self, file_path: str, enable_chunking: bool = True) -> ExtractedDocument:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext not in self.parsers:
            raise ValueError(
                f"Unsupported file format: '{ext}'. Supported formats: {list(self.parsers.keys())}"
            )

        parser_cls = self.parsers[ext]
        raw_doc = parser_cls.parse(file_path, file_name)

        if enable_chunking and self.chunker:
            processed_chunks = self.chunker.process_document(raw_doc)
            return ExtractedDocument(
                file_name=raw_doc.file_name,
                file_type=raw_doc.file_type,
                chunks=processed_chunks,
                full_text=raw_doc.full_text
            )

        return raw_doc
