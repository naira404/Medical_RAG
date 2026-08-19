import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# ضبط مسار المشروع الأساسي
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.retrieval.hybrid_retriever import MedicalHybridRetriever
from src.generation.generator import MedicalGenerator

# -------------------------------------------------------------------------
# 1. بنك الأسئلة المرجعية (15 Benchmark Questions داخل النطاق)
# -------------------------------------------------------------------------
EVALUATION_QUESTIONS = [
    {
        "id": "Q01",
        "question": "What is the recommended daily dosage of folic acid to prevent neural tube defects?",
        "expected_doc": "folic-acid-supplementation-final-rec-statement.pdf"
    },
    {
        "id": "Q02",
        "question": "Does the USPSTF recommend universal screening for autism in children aged 18 to 30 months?",
        "expected_doc": "autismfinalrs.pdf"
    },
    {
        "id": "Q03",
        "question": "What is the USPSTF recommendation grade for depression screening in adults?",
        "expected_doc": "depression-suicide-risk-adults-rs.pdf"
    },
    {
        "id": "Q04",
        "question": "What is the evidence regarding primary care interventions to prevent child maltreatment?",
        "expected_doc": "child-maltreatment-interventions-final-rec-statement.pdf"
    },
    {
        "id": "Q05",
        "question": "Should asymptomatic adults and adolescents be screened for eating disorders?",
        "expected_doc": "eating-disorders-screening-adults-adolescents-final-recommendation.pdf"
    },
    {
        "id": "Q06",
        "question": "What behavioral counseling is recommended for cardiovascular disease prevention in high-risk adults?",
        "expected_doc": "healthy-diet-phys-activity-high-risk-final-rec.pdf"
    },
    {
        "id": "Q07",
        "question": "What are the USPSTF recommendations for healthy weight and weight gain during pregnancy?",
        "expected_doc": "healthy-weight-gain-pregnancy-final-rec-statement.pdf"
    },
    {
        "id": "Q08",
        "question": "Is there sufficient evidence to recommend primary care interventions for illicit drug use in children?",
        "expected_doc": "illicit-drug-use-children-final-rec.pdf"
    },
    {
        "id": "Q09",
        "question": "What is the recommendation grade for screening anxiety in children and adolescents aged 8 to 18?",
        "expected_doc": "screening-anxiety-children-final-recommendation.pdf"
    },
    {
        "id": "Q10",
        "question": "What does USPSTF recommend regarding screening for Major Depressive Disorder (MDD) in adolescents?",
        "expected_doc": "screening-depression-suicide-risk-children-final-recommendation.pdf"
    },
    {
        "id": "Q11",
        "question": "What is the recommendation for speech and language delay screening in asymptomatic preschool children?",
        "expected_doc": "speech-language-delay-screening-children-final-recommendation.pdf"
    },
    {
        "id": "Q12",
        "question": "What behavioral interventions and pharmacotherapy are recommended for tobacco cessation in adults?",
        "expected_doc": "tobacco-cessation-adults-final-rec-statement.pdf"
    },
    {
        "id": "Q13",
        "question": "What primary care interventions prevent tobacco and e-cigarette use among school-aged children?",
        "expected_doc": "tobacco-use-children-final-rec-statement.pdf"
    },
    {
        "id": "Q14",
        "question": "What screening tools and recommendations apply to unhealthy alcohol use in primary care adults?",
        "expected_doc": "unhealthy-alcohol-use-adults-final-rec-statement.pdf"
    },
    {
        "id": "Q15",
        "question": "What is the USPSTF statement on screening for unhealthy drug use in asymptomatic adults?",
        "expected_doc": "unhealthy-drug-use-screening-interventions-final-rec.pdf"
    }
]

# -------------------------------------------------------------------------
# 2. أسئلة خارج النطاق تماماً (Negative / Out-of-Domain Questions)
# -------------------------------------------------------------------------
NEGATIVE_EVALUATION_QUESTIONS = [
    {
        "id": "NEG_01",
        "question": "What is the USPSTF recommendation for screening for skin cancer using whole-body visual examination?",
        "expected": "No Evidence Available"
    },
    {
        "id": "NEG_02",
        "question": "What does USPSTF recommend regarding routine screening for osteoporosis in postmenopausal women?",
        "expected": "No Evidence Available"
    },
    {
        "id": "NEG_03",
        "question": "What is the starting age recommended by USPSTF for colorectal cancer screening using colonoscopy?",
        "expected": "No Evidence Available"
    }
]


def display_retrieval_results(query: str, results: List[Dict[str, Any]], k: int):
    """عرض تفاصيل كل Chunk مسترجع بدقة (النص، السكور، المصدر، الصفحة، الـ ID)."""
    print(f"\n--- Top-{k} Retrieval Details for: '{query}' ---")
    for i, res in enumerate(results[:k]):
        meta = res.get("metadata", {})
        print(f"[{i+1}] Chunk ID: {res.get('id', 'N/A')} | Score: {res.get('score', 0.0):.4f}")
        print(f"    Document: {meta.get('source', 'Unknown')}")
        print(f"    Pages: {meta.get('pages', 'N/A')} | Topic: {meta.get('topic', 'N/A')} | Grade: {meta.get('recommendation_grade', 'N/A')}")
        print(f"    Snippet: {res.get('text', '')[:140]}...\n")


def calculate_precision_at_k(results: List[Dict[str, Any]], expected_doc: str, k: int) -> float:
    if not results or k == 0:
        return 0.0
    top_k_results = results[:k]
    relevant_count = sum(1 for r in top_k_results if r.get("metadata", {}).get("source") == expected_doc)
    return relevant_count / k


def run_full_lab_evaluation():
    print("=" * 80)
    print("HANDS-ON LAB: RETRIEVAL & GENERATION EVALUATION")
    print("=" * 80)

    retriever = MedicalHybridRetriever()
    generator = MedicalGenerator(model_name="gemini-3.6-flash")
    
    p3_scores = []
    p5_scores = []
    
    print(f"\nEvaluating on {len(EVALUATION_QUESTIONS)} In-Domain Benchmark Questions...\n")

    for item in EVALUATION_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        expected = item["expected_doc"]

        hybrid_results = retriever.retrieve(query=q, top_k=10, alpha=0.6)

        p3 = calculate_precision_at_k(hybrid_results, expected, k=3)
        p5 = calculate_precision_at_k(hybrid_results, expected, k=5)
        
        p3_scores.append(p3)
        p5_scores.append(p5)

        if qid in ["Q01", "Q02", "Q03"]:
            print("=" * 80)
            print(f"[{qid}] {q}")
            print(f"Expected Source: {expected}")
            print("=" * 80)
            display_retrieval_results(q, hybrid_results, k=3)
            display_retrieval_results(q, hybrid_results, k=5)
            print(f"-> Precision@3: {p3:.2f} | Precision@5: {p5:.2f}\n")

    # -------------------------------------------------------------------------
    # مقارنة طرق البحث (BM25 vs Vector vs Hybrid)
    # -------------------------------------------------------------------------
    test_q = EVALUATION_QUESTIONS[0]["question"]
    print("=" * 80)
    print("RETRIEVAL METHOD COMPARISON (BM25 vs Vector vs Hybrid Search)")
    print(f"Query: '{test_q}'")
    print("=" * 80)
    
    res_keyword = retriever.retrieve(query=test_q, top_k=3, alpha=0.0)
    res_vector = retriever.retrieve(query=test_q, top_k=3, alpha=1.0)
    res_hybrid = retriever.retrieve(query=test_q, top_k=3, alpha=0.6)

    print("\n[BM25 Keyword Search Only]:")
    for r in res_keyword:
        print(f" - ID: {r['id']} | Doc: {r['metadata'].get('source')} | Score: {r['score']:.4f}")

    print("\n[Dense Vector Search Only]:")
    for r in res_vector:
        print(f" - ID: {r['id']} | Doc: {r['metadata'].get('source')} | Score: {r['score']:.4f}")

    print("\n[Hybrid Search (BM25 + Dense Vector RRF)]:")
    for r in res_hybrid:
        print(f" - ID: {r['id']} | Doc: {r['metadata'].get('source')} | Score: {r['score']:.4f}")

    # -------------------------------------------------------------------------
    # ملخص الإحصائيات
    # -------------------------------------------------------------------------
    avg_p3 = sum(p3_scores) / len(p3_scores)
    avg_p5 = sum(p5_scores) / len(p5_scores)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION METRICS SUMMARY (IN-DOMAIN)")
    print("=" * 80)
    print(f"Total In-Domain Questions : {len(EVALUATION_QUESTIONS)}")
    print(f"Average Precision@3       : {avg_p3 * 100:.2f}%")
    print(f"Average Precision@5       : {avg_p5 * 100:.2f}%")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 3. اختبار الأسئلة غير الموجودة (Out-of-Domain & Anti-Hallucination Test)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OUT-OF-DOMAIN / NEGATIVE CONSTRAINTS TEST (ANTI-HALLUCINATION)")
    print("Testing queries with NO ground truth documents in our dataset")
    print("=" * 80)

    for item in NEGATIVE_EVALUATION_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        print(f"\n[{qid}] Query: {q}")
        
        # استرجاع السياق المقارب
        contexts = retriever.retrieve(query=q, top_k=3, alpha=0.6)
        
        # توليد الإجابة لاختبار الهلوسة
        res = generator.generate_response(query=q, contexts=contexts)
        
        print("\n--- LLM Response Output ---")
        print(res["answer"])
        print("\n--- Retrieved Irrelevant Context Sources ---")
        for s in res["sources"]:
            print(f"• Retrieved: {s.get('source')} (Topic: {s.get('topic')})")
        print("-" * 80)


if __name__ == "__main__":
    run_full_lab_evaluation()