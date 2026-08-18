import os
from typing import Dict, Any, List
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


class MedicalGenerator:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        # استخدام الاسم الكامل للموديل المعتمد في مكتبة google-genai
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)

    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        formatted_blocks = []
        for i, ctx in enumerate(contexts):
            meta = ctx.get("metadata", {})
            source = meta.get("source", "Unknown Guideline")
            topic = meta.get("topic", "N/A")
            grade = meta.get("recommendation_grade", "Not Specified")
            pop = meta.get("target_population", "General Population")
            pages = meta.get("pages", "N/A")
            pub_year = meta.get("publication_year", "N/A")

            block = (
                f"--- [EVIDENCE SOURCE {i+1}] ---\n"
                f"Document: {source}\n"
                f"Topic: {topic}\n"
                f"USPSTF Grade: {grade}\n"
                f"Target Population: {pop}\n"
                f"Publication Year: {pub_year}\n"
                f"Page(s): {pages}\n"
                f"Clinical Evidence Content:\n{ctx.get('text', '')}\n"
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    def generate_response(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        context_text = self._format_context(contexts)

        system_instruction = (
            "You are an authoritative Evidence-Based Clinical Assistant specialized in US Preventive Services Task Force (USPSTF) guidelines.\n\n"
            "STRICT COMPLIANCE DIRECTIVES:\n"
            "1. Grounding: Answer ONLY based on the provided Evidence Sources below. Do NOT extrapolate or inject outside knowledge.\n"
            "2. Insufficient Context: If the evidence does not clearly contain the answer, state: 'The provided USPSTF guidelines do not contain sufficient evidence to answer this question.'\n"
            "3. Structured Recommendation Citation: In your response, explicitly state:\n"
            "   - Document Source (PDF file name).\n"
            "   - USPSTF Recommendation Grade (e.g., Grade A, Grade B, Grade C, Grade D, or I Statement).\n"
            "   - Publication Year / Release Date.\n"
            "   - Target Population & Key Clinical Rationale.\n"
            "4. Inline Citations: Reference every clinical finding with its exact document source and page number(s) (e.g., [depression-suicide-risk-adults-rs.pdf, Page 2, 2023]).\n"
        )

        user_content = f"""Evidence Sources:
{context_text}

Clinical Question:
{query}"""

        # استخدام الإعدادات الصحيحة للـ SDK الجديد مع تمرير system_instruction بشكل منفصل
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )

        return {
            "query": query,
            "answer": response.text,
            "sources": [ctx.get("metadata", {}) for ctx in contexts]
        }