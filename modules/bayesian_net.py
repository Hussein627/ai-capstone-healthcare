# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Covers: Week 7 (Bayesian Networks)
# ============================================================

import numpy as np
import csv
import os
from typing import Dict, List

class SimpleBayesianDiagnostics:
    """
    Bayesian diagnostic model using dynamically loaded 
    priors and conditional probabilities from CSV files.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, float]] = {}
        self._load_from_csv()

    def _load_from_csv(self):
        """Load priors and likelihoods dynamically from diseases.csv"""
        diseases_path = os.path.join(self.data_dir, "diseases.csv")
        
        raw_diseases = []
        try:
            with open(diseases_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d_name = row["disease_name"].strip().lower().replace(' ', '_')
                    symptoms = [s.strip().lower().replace(' ', '_') for s in row["common_symptoms"].split(",")]
                    severity = row.get("severity_level", "MEDIUM").upper()
                    
                    raw_diseases.append({
                        "name": d_name,
                        "symptoms": symptoms,
                        "severity": severity
                    })
        except FileNotFoundError:
            print(f"Warning: {diseases_path} not found. Falling back to default baseline profiles.")
            raw_diseases = [
                {"name": "flu", "symptoms": ["fever", "cough", "fatigue"], "severity": "MEDIUM"},
                {"name": "covid19", "symptoms": ["fever", "cough", "loss_of_smell", "fatigue"], "severity": "HIGH"}
            ]

        # 1. Calculate dynamic Priors P(Disease) based on severity weights
        severity_weights = {"LOW": 0.10, "MEDIUM": 0.20, "HIGH": 0.30, "CRITICAL": 0.40}
        total_weight = sum(severity_weights.get(d["severity"], 0.20) for d in raw_diseases)
        
        # Always include a baseline 'healthy' prior probability
        self.priors['healthy'] = 0.25
        remaining_prob = 0.75

        for d in raw_diseases:
            weight = severity_weights.get(d["severity"], 0.20)
            self.priors[d["name"]] = round(remaining_prob * (weight / total_weight), 4)

        # 2. Build Likelihoods P(Symptom | Disease)
        for d in raw_diseases:
            d_name = d["name"]
            common_symptoms = d["symptoms"]
            self.likelihoods[d_name] = {}
            
            # Assign high probabilities to symptoms listed for the disease
            for s in common_symptoms:
                self.likelihoods[d_name][s] = 0.85

        # Likelihoods for 'healthy' profile (low chance of displaying any disease symptoms)
        self.likelihoods['healthy'] = {
            'fever': 0.02, 'cough': 0.05, 'fatigue': 0.10,
            'headache': 0.08, 'rash': 0.01, 'chest_pain': 0.01,
            'joint_pain': 0.05, 'loss_of_smell': 0.01, 'body_aches': 0.05
        }

        # Fill missing symptom probabilities with a low background noise probability (smoothing)
        all_symptoms = set(s for d in raw_diseases for s in d["symptoms"]) | set(self.likelihoods['healthy'].keys())
        for d_name in self.priors:
            if d_name == 'healthy':
                continue
            if d_name not in self.likelihoods:
                self.likelihoods[d_name] = {}
            for s in all_symptoms:
                if s not in self.likelihoods[d_name]:
                    self.likelihoods[d_name][s] = 0.05  # background probability

    def compute_posterior(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Naïve Bayes posterior:
        P(D|S₁,...,Sₙ) ∝ P(D) × ∏ P(Sᵢ|D)
        """
        posteriors = {}
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]

        for disease, prior in self.priors.items():
            log_prob = np.log(prior)
            for symptom in symptoms_clean:
                # Fallback to background probability 0.01 if symptom not tracked
                p_s_given_d = self.likelihoods.get(disease, {}).get(symptom, 0.01)
                log_prob += np.log(p_s_given_d)
            posteriors[disease] = log_prob

        # Convert log-probabilities to probabilities
        max_log = max(posteriors.values())
        exp_probs = {d: np.exp(v - max_log) for d, v in posteriors.items()}
        total = sum(exp_probs.values())
        return {d: round(v/total, 4) for d, v in exp_probs.items()}

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        posteriors = self.compute_posterior(percept.symptoms)
        top_disease = max(posteriors, key=posteriors.get)
        top_prob = posteriors[top_disease]
        sorted_dx = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)

        return {
            'summary': f"Top: {top_disease} ({top_prob:.2%}) using dynamic CSV priors",
            'diagnosis': top_disease,
            'confidence': top_prob,
            'all_posteriors': posteriors,
            'ranked_diagnoses': sorted_dx[:5]
        }

    def explain(self, disease: str, symptoms: List[str]) -> str:
        symptoms_clean = [s.lower().replace(' ', '_') for s in symptoms]
        likelihoods = self.likelihoods.get(disease, {})
        evidence = [
            f"P({s}|{disease})={likelihoods.get(s, 0.01):.2f}"
            for s in symptoms_clean
        ]
        prior_val = self.priors.get(disease, 0.01)
        return f"P({disease}) = {prior_val} × " + " × ".join(evidence)