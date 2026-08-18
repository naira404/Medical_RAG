import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.retrieval.hybrid_retriever import MedicalHybridRetriever
from src.generation.generator import MedicalGenerator


def run_pipeline():
    print("=" * 70)
    print("Initializing Medical Hybrid Retriever & Generator...")
    print("=" * 70)

    retriever = MedicalHybridRetriever(
        vector_store_dir="data/processed/chroma_db",
        chunks_json_path="data/processed/chunks.json",
        collection_name="uspstf_guidelines"
    )
    generator = MedicalGenerator(model_name="gemini-3.6-flash")

    test_queries = [
        "What is the USPSTF recommendation for folic acid supplementation?",
        "Should young children be screened for autism spectrum disorder?"
    ]

    for q in test_queries:
        print("\n" + "#" * 70)
        print(f"CLINICAL QUESTION: {q}")
        print("#" * 70)

        contexts = retriever.retrieve(query=q, top_k=3, alpha=0.6)
        print(f"\n[Retrieved {len(contexts)} Evidence Sources via Hybrid Search]")

        result = generator.generate_response(query=q, contexts=contexts)

        print("\n--- GENERATED MEDICAL RESPONSE ---\n")
        print(result["answer"])

        print("\n--- CITED SOURCES ---")
        for idx, src in enumerate(result["sources"]):
            print(
                f"[{idx + 1}] Source: {src.get('source')} | "
                f"Topic: {src.get('topic')} | "
                f"Grade: {src.get('recommendation_grade')}"
            )


if __name__ == "__main__":
    run_pipeline()