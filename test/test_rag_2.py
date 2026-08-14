import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.generator import RAGGenerator

def main():
    rag = RAGGenerator()

    # Test Query 1 (Hindi Query)
    query_1 = "कंप्रेसर का एयर प्रेशर कैसे सेट करते हैं और स्क्रू घुमाने पर कितना प्रेशर बढ़ता है?"
    print(f"================ USER QUERY (HINDI) ================")
    print(f"Q: {query_1}\n")
    
    result_1 = rag.generate_answer(query_1, top_k=3)
    
    print("🤖 GEMINI RESPONSE:")
    print(result_1["answer"])
    print("\n📚 RETRIEVED SOURCES:")
    for src in result_1["sources"]:
        print(f" - {src['source']} (Score: {src['score']:.4f})")

    print("\n" + "="*60 + "\n")

    # Test Query 2 (English Query)
    query_2 = "What safety precautions are required before marching the mobile crane?"
    print(f"================ USER QUERY (ENGLISH) ================")
    print(f"Q: {query_2}\n")
    
    result_2 = rag.generate_answer(query_2, top_k=3)
    
    print("🤖 GEMINI RESPONSE:")
    print(result_2["answer"])

if __name__ == "__main__":
    main()