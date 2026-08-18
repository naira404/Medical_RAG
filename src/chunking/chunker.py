import re
import tiktoken
from typing import List, Dict, Any

def find_page_numbers(chunk_start: int, chunk_end: int, page_offsets: List[Dict[str, Any]]) -> List[int]:
    """تحديد أرقام الصفحات التي يغطيها الـ Chunk."""
    pages = []
    for p in page_offsets:
        if max(chunk_start, p["start"]) < min(chunk_end, p["end"]):
            pages.append(p["page"])
    return pages

def chunk_document_by_tokens(
    doc_data: dict,
    max_tokens: int = 500,
    overlap_tokens: int = 100,
    encoding_name: str = "cl100k_base",
) -> List[Dict[str, Any]]:
    """تقطيع المستند حسب التوكنز مع الحفاظ على حدود الجمل وتتبع الصفحات."""
    encoding = tiktoken.get_encoding(encoding_name)
    full_text = doc_data["full_text"]
    page_offsets = doc_data["page_offsets"]

    if not full_text:
        return []

    sentences = re.split(r"(?<=[.?!])\s+", full_text)
    chunks = []
    current_sentences = []
    current_tokens = 0
    search_cursor = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens = len(encoding.encode(sentence))
        if sentence_tokens > max_tokens:
            sentence_tokens = max_tokens

        if current_tokens + sentence_tokens <= max_tokens:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        else:
            if current_sentences:
                chunk_str = " ".join(current_sentences)
                chunk_start = full_text.find(chunk_str, search_cursor)
                chunk_end = chunk_start + len(chunk_str) if chunk_start != -1 else search_cursor
                chunk_pages = find_page_numbers(chunk_start, chunk_end, page_offsets)

                chunks.append({
                    "text": chunk_str,
                    "token_count": current_tokens,
                    "metadata": {
                        "source": doc_data["source"],
                        "pages": chunk_pages,
                        "page_label": (
                            f"Page {chunk_pages[0]}"
                            if len(chunk_pages) == 1
                            else f"Pages {chunk_pages[0]}-{chunk_pages[-1]}"
                        ) if chunk_pages else "Unknown",
                    },
                })
                if chunk_start != -1:
                    search_cursor = chunk_start

            overlap_sentences = []
            accumulated_tokens = 0
            for prev_sentence in reversed(current_sentences):
                prev_len = len(encoding.encode(prev_sentence))
                if accumulated_tokens + prev_len <= overlap_tokens:
                    overlap_sentences.insert(0, prev_sentence)
                    accumulated_tokens += prev_len
                else:
                    break

            current_sentences = overlap_sentences + [sentence]
            current_tokens = accumulated_tokens + sentence_tokens

    if current_sentences:
        chunk_str = " ".join(current_sentences)
        chunk_start = full_text.find(chunk_str, search_cursor)
        chunk_end = chunk_start + len(chunk_str) if chunk_start != -1 else len(full_text)
        chunk_pages = find_page_numbers(chunk_start, chunk_end, page_offsets)

        chunks.append({
            "text": chunk_str,
            "token_count": current_tokens,
            "metadata": {
                "source": doc_data["source"],
                "pages": chunk_pages,
                "page_label": (
                    f"Page {chunk_pages[0]}"
                    if len(chunk_pages) == 1
                    else f"Pages {chunk_pages[0]}-{chunk_pages[-1]}"
                ) if chunk_pages else "Unknown",
            },
        })

    return chunks