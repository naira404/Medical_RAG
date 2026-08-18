import json
import os
import sys
from pathlib import Path

# ضبط مسار المشروع الأساسي
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.retrieval.vector_store import MedicalVectorStore


def main():
    chunks_path = BASE_DIR / "data" / "processed" / "chunks.json"
    chroma_dir = BASE_DIR / "data" / "processed" / "chroma_db"

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found! Please run 'scripts/ingest_documents.py' first.")
        return

    # 1. قراءة الـ Chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # 2. تهيئة الـ Vector Store وبناء الفهرس
    store = MedicalVectorStore(
        persist_directory=str(chroma_dir),
        collection_name="uspstf_guidelines",
        model_name="BAAI/bge-small-en-v1.5"
    )

    store.add_chunks(chunks)

    # 3. اختبار استرجاع سريع (Sanity Check)
    test_query = "What is the recommendation for depression screening in adults?"
    query_emb = [list(store.model.embed([test_query]))[0].tolist()]    
    results = store.collection.query(
        query_embeddings=query_emb,
        n_results=2
    )

    print("\n" + "=" * 60)
    print("Quick Retrieval Test:")
    print(f"Query: '{test_query}'\n")
    for idx, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][idx].get("source")
        topic = results["metadatas"][0][idx].get("topic")
        grade = results["metadatas"][0][idx].get("recommendation_grade")
        print(f"[Match #{idx+1}] Source: {source} | Grade: {grade}")
        print(f"Topic: {topic}")
        print(f"Text Snippet: {doc[:160]}...\n")
    print("=" * 60)


if __name__ == "__main__":
    main()