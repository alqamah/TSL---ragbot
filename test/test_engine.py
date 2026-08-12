import argparse
import os
import sys
from datetime import datetime

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout to support UTF-8 on Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

from src.engine import DataExtractionEngine


def main():
    parser = argparse.ArgumentParser(
        description="Data Extraction Engine CLI Test Tool"
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join("io", "SOP", "02_test.doc"),
        help="Path to file to extract (default: io/SOP/02_test.doc)",
    )
    args = parser.parse_args()

    engine = DataExtractionEngine()

    file_to_test = args.file_path

    print(f"Extracting file: {file_to_test}...")
    doc = engine.extract(file_to_test)

    # Prepare timestamped output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(file_to_test)
    output_dir = os.path.join("test", "op")
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{timestamp}_{base_name}.txt"
    output_path = os.path.join(output_dir, output_filename)

    # Construct extraction report
    lines = [
        "=" * 60,
        "DATA EXTRACTION REPORT",
        f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input File  : {doc.file_name}",
        f"File Type   : {doc.file_type}",
        f"Total Chunks: {len(doc.chunks)}",
        "=" * 60,
        "",
        "--- FULL EXTRACTED TEXT ---",
        doc.full_text,
        "",
        "=" * 60,
        "--- CHUNK DETAILS ---",
    ]

    for idx, chunk in enumerate(doc.chunks, 1):
        lines.append(f"\n[CHUNK {idx}] Metadata: {chunk.metadata}")
        lines.append("-" * 40)
        lines.append(chunk.content)

    report_content = "\n".join(lines)

    # Write output report to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\nSUCCESS!")
    print(f"File Name: {doc.file_name}")
    print(f"File Type: {doc.file_type}")
    print(f"Extracted Chunks: {len(doc.chunks)}")
    print(f"Output saved to : {output_path}")


if __name__ == "__main__":
    main()
