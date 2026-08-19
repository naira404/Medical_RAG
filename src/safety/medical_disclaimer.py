class MedicalDisclaimer:
    @staticmethod
    def get_disclaimer() -> str:
        return (
            "\n\n---\n"
            "🔍 **Clinical Decision Support Disclaimer:** This information is strictly grounded on published "
            "USPSTF recommendation guidelines for educational and clinical decision support purposes only. "
            "It does not constitute direct medical advice, diagnosis, or treatment planning."
        )  