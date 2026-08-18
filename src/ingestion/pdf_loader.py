import fitz
from pathlib import Path
from src.ingestion.cleaner import clean_medical_text

def extract_document_with_page_offsets(pdf_path: Path) -> dict:
    """استخراج نص الـ PDF مع خريطة بمواضع بداية ونهاية كل صفحة."""
    doc = fitz.open(pdf_path)
    full_text_parts = []
    page_offsets = []
    current_char_pos = 0

    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()
        cleaned_page = clean_medical_text(raw_text)

        if not cleaned_page:
            continue

        start_offset = current_char_pos
        full_text_parts.append(cleaned_page)
        current_char_pos += len(cleaned_page) + 1
        end_offset = current_char_pos - 1

        page_offsets.append({
            "page": page_num + 1,
            "start": start_offset,
            "end": end_offset,
        })

    doc.close()
    return {
        "source": pdf_path.name,
        "full_text": " ".join(full_text_parts),
        "page_offsets": page_offsets,
    }