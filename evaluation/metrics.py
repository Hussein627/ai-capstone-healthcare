# ============================================================
# EVALUATION: Metrics Calculation
# Covers: Model Performance Mathematics
# ============================================================

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Dict, List

class Evaluator:
    """
    Calculates standard machine learning performance metrics.
    """
    @staticmethod
    def calculate_classification_metrics(y_true: List, y_pred: List) -> Dict:
        """Returns accuracy, precision, recall, and f1-score."""
        accuracy = accuracy_score(y_true, y_pred)
        
        # 'weighted' accounts for class imbalance in medical data
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        report = classification_report(y_true, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report
        }

    @staticmethod
    def print_report(metrics_dict: Dict, model_name: str):
        """Formats and prints the evaluation report to the console."""
        print(f"\n{'='*50}")
        print(f" Performance Metrics: {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics_dict['accuracy']:.4f}")
        print(f"Precision: {metrics_dict['precision']:.4f}")
        print(f"Recall:    {metrics_dict['recall']:.4f}")
        print(f"F1-Score:  {metrics_dict['f1_score']:.4f}")
        print(f"\nDetailed Classification Report:\n")
        print(metrics_dict['classification_report'])
        print(f"{'='*50}\n")