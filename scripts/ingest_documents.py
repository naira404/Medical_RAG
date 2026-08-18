import json
import os
from pathlib import Path
import sys
from tqdm import tqdm

# Set project root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.chunking.chunker import chunk_document_by_tokens
from src.ingestion.pdf_loader import extract_document_with_page_offsets
from src.ingestion.metadata import extract_document_metadata


def run_ingestion(raw_dir: Path, output_file: Path):
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"Warning: No PDF files found in: {raw_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files. Extracting, adding metadata, and chunking...")

    all_chunks = []
    chunk_id = 0

    for pdf_path in tqdm(pdf_files, desc="Processing files"):
        # 1. استخراج الـ Metadata الأساسية للملف
        doc_meta = extract_document_metadata(pdf_path)

        # 2. استخراج النص والصفحات
        doc_data = extract_document_with_page_offsets(pdf_path)
        
        # 3. التقطيع
        chunks = chunk_document_by_tokens(doc_data)

        # 4. دمج الـ Metadata التفصيلية مع كل Chunk
        for chunk in chunks:
            chunk_metadata = {
                **doc_meta,
                "pages": chunk["metadata"].get("pages", [1]),
                "page_label": chunk["metadata"].get("page_label", "Page 1"),
                "chunk_index": len(all_chunks)
            }
            
            all_chunks.append({
                "id": chunk_id,
                "text": chunk["text"],
                "token_count": chunk["token_count"],
                "metadata": chunk_metadata
            })
            chunk_id += 1

    # 5. الحفظ في مسار data/processed/chunks.json
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Successfully saved {len(all_chunks)} chunks with metadata to:")
    print(f"Path: {output_file}")
    print("=" * 70)

    # طباعة عينة توضح شكل الـ Metadata المحفوظة
    if all_chunks:
        sample = all_chunks[0]
        print("\nMetadata Sample for Chunk #0:")
        print(json.dumps(sample["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "uspstf"
    OUTPUT_DATA_PATH = BASE_DIR / "data" / "processed" / "chunks.json"

    run_ingestion(RAW_DATA_PATH, OUTPUT_DATA_PATH)