import re
from typing import Dict, Any

class InputGuard:
    def __init__(self):
        # كلمات مفتاحية لحالات الطوارئ والأزمات الحادة
        self.emergency_patterns = [
            r"\b(suicid|kill myself|end my life|want to die|overdose)\b",
            r"\b(انتحار|اموت نفسي|انهي حياتي|جرعة زائدة)\b",
            r"\b(chest pain|cannot breathe|severe bleeding|heart attack)\b",
            r"\b(الم في الصدر|صعوبة في التنفس|نزيف حاد|جلطة)\b"
        ]

    def check_safety(self, query: str) -> Dict[str, Any]:
        """
        يفحص السؤال للتأكد من عدم وجود حالات طوارئ أو أزمات فورية.
        """
        for pattern in self.emergency_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "reason": "emergency_detected",
                    "message": (
                        "⚠️ **Emergency Protocol Activated:** If you or someone you know is experiencing "
                        "a medical emergency or mental health crisis, please call your local emergency services "
                        "(e.g., 911 or national suicide helpline 988) immediately. This system cannot handle emergency care."
                    )
                }
        
        return {"is_safe": True, "reason": None, "message": None}