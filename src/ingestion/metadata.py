import re
from pathlib import Path
from typing import Dict, Any

# خريطة سريعة لربط الملفات بمواضيعها وتصنيفاتها الطبية
DOCUMENT_REGISTRY = {
    "autismfinalrs.pdf": {
        "topic": "Autism Spectrum Disorder Screening in Young Children",
        "target_population": "Children aged 18 to 30 months",
        "recommendation_grade": "I",
        "publication_year": 2016
    },
    "child-maltreatment-interventions-final-rec-statement.pdf": {
        "topic": "Primary Care Interventions to Prevent Child Maltreatment",
        "target_population": "Children and adolescents younger than 18 years",
        "recommendation_grade": "I",
        "publication_year": 2024
    },
    "depression-suicide-risk-adults-rs.pdf": {
        "topic": "Screening for Depression and Suicide Risk in Adults",
        "target_population": "Asymptomatic adults 19 years or older, including pregnant/postpartum",
        "recommendation_grade": "B (Depression) / I (Suicide Risk)",
        "publication_year": 2023
    },
    "eating-disorders-screening-adults-adolescents-final-recommendation.pdf": {
        "topic": "Screening for Eating Disorders in Adolescents and Adults",
        "target_population": "Asymptomatic adolescents and adults 10 years or older",
        "recommendation_grade": "I",
        "publication_year": 2022
    },
    "folic-acid-supplementation-final-rec-statement.pdf": {
        "topic": "Folic Acid Supplementation to Prevent Neural Tube Defects",
        "target_population": "Persons planning to or who could become pregnant",
        "recommendation_grade": "A",
        "publication_year": 2023
    },
    "healthy-diet-phys-activity-high-risk-final-rec.pdf": {
        "topic": "Behavioral Counseling to Promote a Healthy Diet and Physical Activity for CVD Prevention",
        "target_population": "Adults 18 years or older with CVD risk factors",
        "recommendation_grade": "B",
        "publication_year": 2020
    },
    "healthy-weight-gain-pregnancy-final-rec-statement.pdf": {
        "topic": "Behavioral Counseling Interventions for Healthy Weight and Weight Gain in Pregnancy",
        "target_population": "Pregnant adolescents and adults",
        "recommendation_grade": "B",
        "publication_year": 2021
    },
    "illicit-drug-use-children-final-rec.pdf": {
        "topic": "Primary Care–Based Interventions to Prevent Illicit Drug Use in Children and Young Adults",
        "target_population": "Children, adolescents, and young adults (up to 25 years)",
        "recommendation_grade": "I",
        "publication_year": 2020
    },
    "screening-anxiety-children-final-recommendation.pdf": {
        "topic": "Screening for Anxiety in Children and Adolescents",
        "target_population": "Children and adolescents aged 8 to 18 years",
        "recommendation_grade": "B (8-18 yrs) / I (<= 7 yrs)",
        "publication_year": 2022
    },
    "screening-depression-suicide-risk-children-final-recommendation.pdf": {
        "topic": "Screening for Depression and Suicide Risk in Children and Adolescents",
        "target_population": "Adolescents aged 12 to 18 years / Children 11 or younger",
        "recommendation_grade": "B (MDD 12-18) / I (MDD <=11, Suicide Risk)",
        "publication_year": 2022
    },
}

def extract_document_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Returns rich metadata for citation and filtering based on document filename.
    """
    file_name = file_path.name
    
    if file_name in DOCUMENT_REGISTRY:
        meta = DOCUMENT_REGISTRY[file_name].copy()
        meta["source"] = file_name
        return meta
    
    # Fallback if unknown file is added
    clean_topic = file_path.stem.replace("-", " ").replace("_", " ").title()
    return {
        "source": file_name,
        "topic": clean_topic,
        "target_population": "General",
        "recommendation_grade": "Not Specified",
        "publication_year": None
    }