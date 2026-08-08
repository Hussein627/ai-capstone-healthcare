# ============================================================
# MODULE 8: Natural Language Processing
# Covers: Text Parsing, Tokenization, and Keyword Extraction
# ============================================================

import re
from typing import Dict, List

class MedicalNLPProcessor:
    """
    Natural Language Processing module.
    Uses rule-based tokenization and keyword extraction
    to parse free-text patient complaints into structured symptoms.
    """
    def __init__(self):
        # Master list of accepted system features
        self.valid_symptoms = [
            'fever', 'cough', 'fatigue', 'headache',
            'body_aches', 'loss_of_smell', 'chest_pain',
            'rash', 'joint_pain', 'shortness_of_breath',
            'sweating', 'frequent_urination', 'excessive_thirst',
            'blurred_vision', 'night_sweats', 'weight_loss',
            'stiff_neck', 'light_sensitivity'
        ]
        
        # A dictionary mapping common layperson terms to strict medical symptoms
        self.symptom_synonyms = {
            'hot': 'fever',
            'temperature': 'fever',
            'coughing': 'cough',
            'tired': 'fatigue',
            'exhausted': 'fatigue',
            'head hurts': 'headache',
            'migraine': 'headache',
            'body hurts': 'body_aches',
            'muscle pain': 'body_aches',
            'can\'t smell': 'loss_of_smell',
            'no taste': 'loss_of_smell',
            'chest hurts': 'chest_pain',
            'heart hurts': 'chest_pain',
            'hard to breathe': 'shortness_of_breath',
            'panting': 'shortness_of_breath',
            'cant breathe': 'shortness_of_breath',
            'sweaty': 'sweating',
            'peeing a lot': 'frequent_urination',
            'always thirsty': 'excessive_thirst',
            'cant see well': 'blurred_vision',
            'neck hurts': 'stiff_neck'
        }

    def process_text(self, text: str) -> List[str]:
        """Extract standard symptoms from raw patient text."""
        # Lowercase everything for uniform processing
        text = text.lower()
        
        # FIX: Clean underscores from terminal inputs so they match dictionary keys
        text = text.replace('_', ' ')
        
        # Regex: Remove punctuation but keep alphanumeric characters and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        extracted = set()
        
        # 1. Direct match with standard symptoms
        for symptom in self.valid_symptoms:
            clean_symptom = symptom.replace('_', ' ')
            if clean_symptom in text:
                extracted.add(symptom)
                
        # 2. Match synonyms and slang to their standard definitions
        for slang, standard in self.symptom_synonyms.items():
            if slang in text:
                extracted.add(standard)
                
        return list(extracted)

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        
        raw_input = getattr(percept, 'raw_text', " ".join(percept.symptoms))
        extracted_symptoms = self.process_text(raw_input)
        
        if extracted_symptoms:
            percept.symptoms = extracted_symptoms
            
        return {
            'summary': f"NLP Extracted {len(extracted_symptoms)} valid symptoms from raw text",
            'extracted_symptoms': extracted_symptoms,
            'confidence': 0.95 if extracted_symptoms else 0.0
        }