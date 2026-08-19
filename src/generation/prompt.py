from typing import List, Dict, Any

def build_clinical_rag_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    يبني Prompt صارم يمنع الـ Hallucination ويلزم النموذج بذكر المصادر وأرقام الصفحات.
    """
    formatted_contexts = []
    for idx, ctx in enumerate(contexts, start=1):
        meta = ctx.get("metadata", {})
        doc_name = meta.get("source", meta.get("file_name", "Unknown Document"))
        pages = meta.get("page", meta.get("pages", "N/A"))
        topic = meta.get("topic", "Clinical Guideline")
        text = ctx.get("text", "").strip()
        
        formatted_contexts.append(
            f"[Source {idx}]: Document: {doc_name} | Pages: {pages} | Topic: {topic}\n"
            f"Content: {text}\n"
        )

    context_str = "\n".join(formatted_contexts)

    system_prompt = f"""You are a specialized Clinical Decision Support AI Assistant grounded strictly in USPSTF (US Preventive Services Task Force) guidelines.

Rules:
1. Answer the query using ONLY the provided contexts below.
2. If the context does not contain sufficient evidence to answer the query, reply EXACTLY with:
   "The provided USPSTF guidelines do not contain sufficient evidence to answer this question."
3. Every claim must have an inline citation referring to the source number, document name, and page (e.g., [Source 1, Page 4]).
4. Structure your response into:
   - **Clinical Recommendation:** (Summary and Grade: A, B, C, D, or I)
   - **Target Population:** (Who this applies to)
   - **Evidence & Clinical Considerations:** (Key points from evidence)
   - **Citations:** (List of referenced source documents)

Contexts:
{context_str}

User Query: {query}

Structured Clinical Answer:"""

    return system_prompt   