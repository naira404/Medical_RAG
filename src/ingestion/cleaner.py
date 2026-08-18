import re

def clean_medical_text(text: str) -> str:
    """إصلاح الكلمات المقسومة بشرطة عبر الأسطر وتنظيف المسافات الزائدة."""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()