# ============================================================
# EVALUATION: Visualizations
# Covers: Confusion Matrices & Loss/Accuracy Curves
# ============================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Dict

class Plotter:
    """
    Generates and saves performance charts for the Capstone report.
    """
    def __init__(self, save_dir: str = "evaluation"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true: List, y_pred: List, classes: List[str], filename: str = "ml_evaluation.png"):
        """Generates a heatmap of predicted vs actual diagnoses."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        
        plt.title('Machine Learning Diagnostic Confusion Matrix', pad=20, fontsize=14)
        plt.ylabel('True Diagnosis', fontsize=12)
        plt.xlabel('Predicted Diagnosis', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved Confusion Matrix to {save_path}")

    def plot_training_history(self, history: Dict, filename: str = "nn_training.png"):
        """Generates Accuracy and Loss curves for the Deep Neural Network."""
        plt.figure(figsize=(14, 5))
        
        # Plot 1: Accuracy Curve
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        plt.plot(history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        plt.title('Neural Network Accuracy over Epochs', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Accuracy', fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Loss Curve
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss', color='red', linewidth=2)
        plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title('Neural Network Loss over Epochs', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Loss (Categorical Crossentropy)', fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Saved Training Curves to {save_path}")