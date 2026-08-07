# ============================================================
# MODULE 4: Machine Learning Diagnostic Classifier
# Covers: Week 9 (Supervised Learning, Decision Trees, Ensembles)
# ============================================================
"""
The data-driven pattern recognizer. Trains a Decision Tree
(ID3-style, criterion='entropy'), a Random Forest, and a
Gradient Boosting classifier on synthetic patient data, then
automatically selects the best-performing model via 5-fold CV.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix


class MLDiagnosticClassifier:

    # Fixed 18-symptom feature vector — every patient record must follow this order
    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache', 'rash', 'loss_of_smell',
        'chest_pain', 'joint_pain', 'shortness_of_breath', 'sweating',
        'body_aches', 'sore_throat', 'runny_nose', 'frequent_urination',
        'excessive_thirst', 'blurred_vision', 'nausea', 'chills',
    ]

    # Disease -> symptom probability profile, used to generate synthetic patients
    DISEASE_PROFILES = {
        'flu': {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88, 'headache': 0.70,
                'body_aches': 0.80, 'chills': 0.60, 'sore_throat': 0.35},
        'covid19': {'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90, 'loss_of_smell': 0.85,
                    'headache': 0.65, 'body_aches': 0.60, 'shortness_of_breath': 0.40},
        'dengue': {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85, 'headache': 0.90,
                   'fatigue': 0.80, 'body_aches': 0.88, 'nausea': 0.50},
        'cardiac': {'chest_pain': 0.92, 'shortness_of_breath': 0.88, 'fatigue': 0.70,
                    'sweating': 0.75, 'nausea': 0.30},
        'diabetes': {'fatigue': 0.82, 'frequent_urination': 0.95, 'excessive_thirst': 0.92,
                     'blurred_vision': 0.70},
        'common_cold': {'cough': 0.90, 'fever': 0.50, 'headache': 0.60, 'fatigue': 0.55,
                        'sore_throat': 0.75, 'runny_nose': 0.85},
        'healthy': {'fever': 0.02, 'cough': 0.05, 'fatigue': 0.10, 'headache': 0.08},
        'migraine': {'headache': 0.95, 'nausea': 0.70, 'fatigue': 0.40},
    }

    def __init__(self, samples_per_disease: int = 250, random_state: int = 42):
        self.samples_per_disease = samples_per_disease
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -1.0
        self.classes_ = sorted(self.DISEASE_PROFILES.keys())
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self._trained = False

    # ── Synthetic data generation ────────────────────────────
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.random_state)
        X, y = [], []
        for disease, profile in self.DISEASE_PROFILES.items():
            for _ in range(self.samples_per_disease):
                vector = [
                    1 if rng.random() < profile.get(sym, 0.03) else 0
                    for sym in self.SYMPTOM_FEATURES
                ]
                X.append(vector)
                y.append(disease)
        return np.array(X), np.array(y)

    # ── Training ──────────────────────────────────────────────
    def train(self, verbose: bool = True):
        X, y = self._generate_synthetic_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        candidates = {
            'DecisionTree': DecisionTreeClassifier(
                criterion='entropy', max_depth=8, random_state=self.random_state),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=self.random_state),
        }

        for name, model in candidates.items():
            model.fit(self.X_train, self.y_train)
            cv_scores = cross_val_score(model, X, y, cv=5)
            mean_score = cv_scores.mean()
            self.models[name] = {'model': model, 'cv_score': mean_score}
            if verbose:
                print(f"    {name:<18} 5-fold CV accuracy: {mean_score:.2%}")
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = model
                self.best_model_name = name

        self._trained = True
        if verbose:
            test_acc = accuracy_score(self.y_test, self.best_model.predict(self.X_test))
            print(f"    → Best model: {self.best_model_name} "
                  f"(CV={self.best_score:.2%}, held-out test={test_acc:.2%})")
        return self

    # ── Feature vector conversion ────────────────────────────
    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        clean = {s.lower().strip().replace(' ', '_') for s in symptoms}
        return np.array([[1 if sym in clean else 0 for sym in self.SYMPTOM_FEATURES]])

    # ── Prediction ────────────────────────────────────────────
    def predict(self, symptoms: List[str]) -> Dict:
        if not self._trained:
            raise RuntimeError("Model not trained yet — call .train() first")
        vector = self._symptoms_to_vector(symptoms)
        pred = self.best_model.predict(vector)[0]
        proba = self.best_model.predict_proba(vector)[0]
        class_order = list(self.best_model.classes_)
        confidence = float(proba[class_order.index(pred)])
        return {
            'diagnosis': pred,
            'confidence': round(confidence, 4),
            'model_used': self.best_model_name,
            'all_probabilities': dict(zip(class_order, proba.round(4))),
        }

    # ── Standard module interface (called by the Agent) ─────
    def analyze(self, patient) -> Dict:
        if not self._trained:
            self.train(verbose=False)
        return self.predict(patient.symptoms)

    # ── Evaluation plots ──────────────────────────────────────
    def plot_evaluation(self, save_path: str = "reports/ml_evaluation.png"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred, labels=self.best_model.classes_)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.best_model.classes_,
                    yticklabels=self.best_model.classes_, ax=axes[0])
        axes[0].set_title(f'Confusion Matrix — {self.best_model_name}')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            order = np.argsort(importances)[::-1]
            axes[1].bar(range(len(importances)),
                        importances[order], color='#3498DB')
            axes[1].set_xticks(range(len(importances)))
            axes[1].set_xticklabels(
                [self.SYMPTOM_FEATURES[i] for i in order], rotation=60, ha='right', fontsize=8)
            axes[1].set_title('Feature Importance')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✔ Saved ML evaluation plots -> {save_path}")


# ============================================================
# Quick self-test — run: python -m modules.ml_classifier
# ============================================================
if __name__ == "__main__":
    clf = MLDiagnosticClassifier()
    clf.train(verbose=True)
    result = clf.predict(["fever", "cough", "fatigue", "loss_of_smell"])
    print(f"Diagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Used: {result['model_used']}")
