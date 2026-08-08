# ============================================================
# MODULE 9: Search Algorithms
# Covers: Breadth-First Search (BFS) & Graph Traversal
# ============================================================

from typing import Dict, List, Optional
from collections import deque

class MedicalSearchEngine:
    """
    Graph-based search module.
    Uses BFS to find pathways between symptoms, 
    diseases, and critical patient outcomes.
    """
    def __init__(self):
        # A simplified graph representing symptom/disease pathways
        self.medical_graph = {
            'fever': ['flu', 'covid19', 'meningitis', 'dengue'],
            'cough': ['flu', 'covid19', 'common_cold', 'tuberculosis'],
            'fatigue': ['flu', 'covid19', 'diabetes', 'dengue'],
            'chest_pain': ['cardiac_event', 'pneumonia'],
            'flu': ['respiratory_distress', 'recovery'],
            'covid19': ['respiratory_distress', 'pneumonia'],
            'dengue': ['hemorrhagic_fever'],
            'pneumonia': ['icu_admission'],
            'cardiac_event': ['icu_admission'],
            'respiratory_distress': ['icu_admission']
        }

    def bfs_path(self, start_node: str, target_node: str) -> Optional[List[str]]:
        """Find the shortest path between a symptom and an outcome using BFS."""
        if start_node not in self.medical_graph:
            return None
            
        queue = deque([[start_node]])
        visited = {start_node}
        
        while queue:
            path = queue.popleft()
            current = path[-1]
            
            if current == target_node:
                return path
                
            for neighbor in self.medical_graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return None

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        symptoms = percept.symptoms
        
        if not symptoms:
            return {'summary': 'No symptoms provided for search', 'risk_path': None}
        
        # Take the patient's first symptom and search for a path to the ICU
        primary_symptom = symptoms[0]
        risk_path = self.bfs_path(primary_symptom, 'icu_admission')
        
        if risk_path:
            summary = f"Risk pathway found: {' → '.join(risk_path)}"
            confidence = 0.85
        else:
            summary = f"No direct critical pathway found for {primary_symptom}"
            confidence = 0.50
            
        return {
            'summary': summary,
            'primary_symptom_tracked': primary_symptom,
            'critical_pathway': risk_path,
            'confidence': confidence
        }