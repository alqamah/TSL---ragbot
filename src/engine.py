# engine.py
import os
from src.parsers import TextParser, DocxParser, ExcelParser
from src.schema import ExtractedDocument


class DataExtractionEngine:
    def __init__(self):
        self.parsers = {
            ".txt": TextParser,
            ".docx": DocxParser,
            ".xlsx": ExcelParser,
            ".xls": ExcelParser,
            ".csv": ExcelParser,
        }

    def extract(self, file_path: str) -> ExtractedDocument:
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
        return parser_cls.parse(file_path, file_name)