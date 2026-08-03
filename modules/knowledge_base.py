# ============================================================
# MODULE 2: FOL Knowledge Base + Inference Engine
# Covers: Week 5 (First-Order Logic & Inference)
# ============================================================

from typing import Set, List, Dict, Tuple, Optional
import csv
import os

class MedicalKnowledgeBase:
    """
    First-Order Logic based medical knowledge base.
    Supports forward chaining, backward chaining,
    and confidence-weighted inference, loaded dynamically from CSV files.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.facts: Set[str] = set()
        self.rules: List[Tuple] = []
        self.certainty_factors: Dict[str, float] = {}
        self.valid_symptoms: Set[str] = set()
        self._load_medical_knowledge()

    def _load_medical_knowledge(self):
        """Load domain medical knowledge from CSV files and define base inference rules"""
        symptoms_path = os.path.join(self.data_dir, "symptoms.csv")
        diseases_path = os.path.join(self.data_dir, "diseases.csv")

        # 1. Load valid symptom vocabulary from symptoms.csv
        try:
            with open(symptoms_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.valid_symptoms.add(row["symptom_name"].strip().lower().replace(' ', '_'))
        except FileNotFoundError:
            print(f"Warning: {symptoms_path} not found. Proceeding without strict vocabulary check.")

        # 2. Load diseases and map them into inference rules from diseases.csv
        try:
            with open(diseases_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    disease_name = row["disease_name"].strip().lower().replace(' ', '_') + "_suspected"
                    # Parse comma-separated symptoms
                    symptoms_list = [s.strip().lower().replace(' ', '_') for s in row["common_symptoms"].split(",")]
                    
                    # Assign a default certainty factor based on severity or standard weight
                    severity = row.get("severity_level", "MEDIUM").upper()
                    cf_map = {"LOW": 0.75, "MEDIUM": 0.82, "HIGH": 0.88, "CRITICAL": 0.95}
                    cf = cf_map.get(severity, 0.80)

                    # Add rule: [symptoms] -> disease_suspected
                    self.add_rule(symptoms_list, disease_name, cf)

                    # Add secondary urgency/action rule if critical
                    if severity == "CRITICAL":
                        self.add_rule([disease_name], "EMERGENCY", 0.98)
        except FileNotFoundError:
            print(f"Warning: {diseases_path} not found. Falling back to default baseline rules.")
            # Fallback hardcoded rules if CSV is missing
            fallback_rules = [
                (["fever", "cough", "fatigue"], "flu_suspected", 0.75),
                (["fever", "cough", "shortness_of_breath"], "pneumonia_suspected", 0.90),
                (["pneumonia_suspected"], "EMERGENCY", 0.95)
            ]
            for conditions, conclusion, cf in fallback_rules:
                self.add_rule(conditions, conclusion, cf)

        # 3. Add system-wide logical conclusion rules (e.g., confirmations and treatments)
        system_rules = [
            (["flu_suspected", "high_fever"], "flu_confirmed", 0.85),
            (["flu_confirmed"], "REST_AND_MEDICATE", 0.90),
            (["pneumonia_suspected"], "EMERGENCY", 0.95)
        ]
        for conditions, conclusion, cf in system_rules:
            self.add_rule(conditions, conclusion, cf)

    def add_fact(self, fact: str, certainty: float = 1.0):
        self.facts.add(fact)
        self.certainty_factors[fact] = certainty

    def add_rule(self, conditions: List[str], conclusion: str, certainty: float = 1.0):
        self.rules.append((conditions, conclusion, certainty))

    def load_patient_symptoms(self, symptoms: List[str]):
        """Load patient symptoms as facts"""
        for symptom in symptoms:
            formatted = symptom.lower().strip().replace(' ', '_')
            self.add_fact(formatted)

    def forward_chain(self, verbose: bool = False) -> Dict[str, float]:
        """Forward chaining with certainty factors"""
        inferred = {}
        changed = True
        iteration = 0

        while changed:
            changed = False
            iteration += 1
            for conditions, conclusion, rule_cf in self.rules:
                all_known = all(
                    c in self.facts or c in inferred for c in conditions
                )
                if all_known and conclusion not in inferred:
                    cond_cfs = [
                        self.certainty_factors.get(c, inferred.get(c, 1.0))
                        for c in conditions
                    ]
                    combined_cf = rule_cf * min(cond_cfs)
                    inferred[conclusion] = round(combined_cf, 4)

                    if verbose:
                        cond_str = " ∧ ".join(conditions)
                        print(f"  Iter {iteration}: {cond_str} → {conclusion} (CF={combined_cf:.3f})")
                    changed = True
        return inferred

    def backward_chain(self, goal: str, visited: Optional[Set] = None, depth: int = 0) -> Tuple[bool, float]:
        """Backward chaining — prove a goal"""
        visited = visited or set()

        if goal in self.facts:
            return True, self.certainty_factors.get(goal, 1.0)
        if goal in visited:
            return False, 0.0
        visited.add(goal)

        for conditions, conclusion, rule_cf in self.rules:
            if conclusion == goal:
                results = [
                    self.backward_chain(c, visited.copy(), depth+1)
                    for c in conditions
                ]
                if all(proved for proved, _ in results):
                    cf = rule_cf * min(cf for _, cf in results)
                    return True, round(cf, 4)
        return False, 0.0

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        self.facts = set()
        self.certainty_factors = {}
        self.load_patient_symptoms(percept.symptoms)

        # Add vitals as facts
        if hasattr(percept, 'temperature') and percept.temperature > 38.0:
            self.add_fact("fever", min(1.0, (percept.temperature - 37.0) / 3.0))
        if hasattr(percept, 'temperature') and percept.temperature > 39.5:
            self.add_fact("high_fever", 1.0)
        if hasattr(percept, 'heart_rate') and percept.heart_rate > 100:
            self.add_fact("tachycardia", 1.0)

        inferred = self.forward_chain()
        diseases = {k: v for k, v in inferred.items() if 'suspected' in k or 'confirmed' in k}

        top = max(diseases, key=diseases.get) if diseases else "Unknown"
        return {
            'summary': f"Inferred {len(inferred)} conclusions using dynamic CSV knowledge base",
            'diagnosis': top,
            'confidence': diseases.get(top, 0.5),
            'all_inferred': inferred
        }

    def get_explanation(self, diagnosis: str) -> str:
        """Explain how a diagnosis was reached"""
        for conditions, conclusion, cf in self.rules:
            if conclusion == diagnosis:
                return f"'{diagnosis}' derived from: {' + '.join(conditions)} (CF={cf})"
        return f"'{diagnosis}' is a base fact"