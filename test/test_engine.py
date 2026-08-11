# test_engine.py
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to support UTF-8 on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from src.engine import DataExtractionEngine


def main():
    engine = DataExtractionEngine()

    # Point to a sample file in io/SOP
    file_to_test = os.path.join("io", "SOP", "01_test.XLSX")

    print(f"Extracting file: {file_to_test}...")
    doc = engine.extract(file_to_test)

    print("\nSUCCESS!")
    print(f"File Name: {doc.file_name}")
    print(f"File Type: {doc.file_type}")
    print(f"Extracted Chunks: {len(doc.chunks)}")

    if doc.chunks:
        print("\n--- FIRST CHUNK CONTENT SAMPLE ---")
        print(doc.chunks[0].content[:400])

        print("\n--- FIRST CHUNK METADATA ---")
        print(doc.chunks[0].metadata)
    else:
        print("\n[WARNING] No chunks were extracted from this document.")


if __name__ == "__main__":
    main()