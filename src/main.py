import argparse
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.rag_engine import RAGPipeline

load_dotenv()


def print_banner(pipeline: RAGPipeline):
    points_count = pipeline.count_indexed_points()
    print("=" * 70)
    print("      INDUSTRIAL SOP RAG ASSISTANT (Gemini + Qdrant)       ")
    print("=" * 70)
    print(f"Collection    : {pipeline.collection_name}")
    print(f"Storage Mode  : {pipeline.vector_store.mode}")
    print(f"Indexed Chunks: {points_count}")
    print(f"LLM Model     : {pipeline.generation_model}")
    print(f"Embed Model   : {pipeline.embedding_model}")
    print("=" * 70)
    print("Commands:")
    print("  Type your question and press Enter.")
    print("  ':ingest <path>' to index a file or folder.")
    print("  ':status'        to check indexed chunks count.")
    print("  'exit' / 'quit'  to exit.")
    print("=" * 70)


def run_interactive_session(pipeline: RAGPipeline, top_k: int = 3, show_sources: bool = True):
    print_banner(pipeline)

    # Check if empty collection and default SOPs exist
    points_count = pipeline.count_indexed_points()
    if points_count == 0:
        default_sop_dir = os.path.join(PROJECT_ROOT, "io", "SOP")
        if os.path.isdir(default_sop_dir):
            print("\n[Notice] The vector database collection is currently empty.")
            choice = input(f"Would you like to ingest documents from '{default_sop_dir}' now? (y/n): ").strip().lower()
            if choice in ["y", "yes"]:
                pipeline.ingest_directory(default_sop_dir)

    while True:
        try:
            query = input("\nAsk Query > ").strip()
            if not query:
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            if query.startswith(":ingest "):
                target_path = query[8:].strip()
                if not os.path.isabs(target_path):
                    target_path = os.path.join(PROJECT_ROOT, target_path)
                if os.path.isdir(target_path):
                    pipeline.ingest_directory(target_path)
                elif os.path.isfile(target_path):
                    pipeline.ingest_file(target_path)
                else:
                    print(f"Path not found: {target_path}")
                continue

            if query.lower() == ":status":
                print(f"Currently indexed points in '{pipeline.collection_name}': {pipeline.count_indexed_points()}")
                continue

            print("\n[1/2] Retrieving relevant SOP context...")
            result = pipeline.ask(query, top_k=top_k, show_sources=show_sources)

            if show_sources and result["sources"]:
                print(f"[2/2] Found {len(result['sources'])} relevant chunks:")
                for idx, src in enumerate(result["sources"], 1):
                    doc_source = src.get("source") or "Unknown"
                    score = src.get("score", 0.0)
                    snippet = src.get("content", "").replace("\n", " ")[:150]
                    print(f"  [{idx}] (Score: {score:.3f}) | {doc_source} -> \"{snippet}...\"")

            print("\n" + "-" * 70)
            print("ASSISTANT RESPONSE:")
            print("-" * 70)
            print(result["answer"])
            print("-" * 70)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as err:
            print(f"\n[Error processing request]: {err}")


def main():
    parser = argparse.ArgumentParser(description="Industrial SOP RAG Pipeline CLI")
    parser.add_argument("--query", "-q", type=str, help="Single query to run without interactive prompt")
    parser.add_argument("--ingest-dir", "-i", type=str, help="Path to directory of SOP documents to ingest")
    parser.add_argument("--ingest-file", type=str, help="Path to single document to ingest")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--collection", type=str, default="industrial_sops_gemini", help="Qdrant collection name")
    parser.add_argument("--model", "-m", type=str, default="gemini-3.7-flash", help="Gemini generation model")
    parser.add_argument("--no-sources", action="store_true", help="Hide retrieved sources in output")

    args = parser.parse_args()

    pipeline = RAGPipeline(
        collection_name=args.collection,
        generation_model=args.model,
    )

    try:
        if args.ingest_dir:
            pipeline.ingest_directory(args.ingest_dir)

        if args.ingest_file:
            pipeline.ingest_file(args.ingest_file)

        if args.query:
            result = pipeline.ask(args.query, top_k=args.top_k, show_sources=not args.no_sources)
            print("-" * 75)
            print("ASSISTANT RESPONSE:")
            print("-" * 75)
            print(result["answer"])
            print("-" * 75)
        else:
            run_interactive_session(pipeline, top_k=args.top_k, show_sources=not args.no_sources)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
