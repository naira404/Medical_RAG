from typing import Dict, Any, List
from src.retrieval.hybrid_retriever import MedicalHybridRetriever
from src.generation.generator import Generator
from src.generation.prompt import build_clinical_rag_prompt
from src.safety.input_guard import InputGuard
from src.safety.medical_disclaimer import MedicalDisclaimer


class ClinicalRAGPipeline:
    def __init__(self):
        self.input_guard = InputGuard()
        self.retriever = MedicalHybridRetriever()
        self.generator = Generator()

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # الخطوة 1: فحص أمان السؤال (Input Safety)
        guard_result = self.input_guard.check_safety(query)
        if not guard_result["is_safe"]:
            return {
                "query": query,
                "answer": guard_result["message"],
                "contexts": [],
                "status": "unsafe_emergency_blocked"
            }

        # الخطوة 2: استرجاع الأدلة (Hybrid Retrieval with RRF)
        contexts: List[Dict[str, Any]] = self.retriever.retrieve(query, top_k=top_k)

        # الخطوة 3: التحقق من وجود أدلة كافية (Insufficient Evidence Fallback)
        if not contexts:
            fallback_msg = "The provided USPSTF guidelines do not contain sufficient evidence to answer this question."
            return {
                "query": query,
                "answer": fallback_msg + MedicalDisclaimer.get_disclaimer(),
                "contexts": [],
                "status": "insufficient_evidence_fallback"
            }

        # الخطوة 4: بناء الـ Grounded Prompt والتوليد
        prompt = build_clinical_rag_prompt(query, contexts)
        raw_answer = self.generator.generate_response(query=query, prompt=prompt)

        # الخطوة 5: إلحاق التنبيه الطبي النهائي
        final_answer = raw_answer + MedicalDisclaimer.get_disclaimer()

        return {
            "query": query,
            "answer": final_answer,
            "contexts": contexts,
            "status": "success"
        }