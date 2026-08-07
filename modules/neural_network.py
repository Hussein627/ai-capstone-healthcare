# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks, Deep Learning)
# ============================================================
"""
The deep-learning specialist. A Multi-Layer Perceptron trained
with TensorFlow/Keras on synthetic patient data, using batch
normalization, dropout, and early stopping.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # quiet TF startup logs

import numpy as np
from typing import Dict, List

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


class NeuralDiagnosticModel:

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache', 'rash', 'loss_of_smell',
        'chest_pain', 'joint_pain', 'shortness_of_breath', 'sweating',
        'body_aches', 'sore_throat', 'runny_nose', 'frequent_urination',
        'excessive_thirst', 'blurred_vision', 'nausea', 'chills',
    ]

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

    def __init__(self, samples_per_disease: int = 375, random_state: int = 42):
        self.samples_per_disease = samples_per_disease
        self.random_state = random_state
        self.classes_ = sorted(self.DISEASE_PROFILES.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.model = None
        self.history = None
        self._trained = False
        self._use_fallback = not _HAS_TF

    # ── Data generation ───────────────────────────────────────
    def _generate_data(self):
        rng = np.random.RandomState(self.random_state)
        X, y = [], []
        for disease, profile in self.DISEASE_PROFILES.items():
            for _ in range(self.samples_per_disease):
                vec = [1 if rng.random() < profile.get(s, 0.03) else 0
                       for s in self.SYMPTOM_FEATURES]
                X.append(vec)
                y.append(self.class_to_idx[disease])
        X = np.array(X, dtype='float32')
        y = np.array(y, dtype='int32')
        # shuffle
        perm = rng.permutation(len(X))
        return X[perm], y[perm]

    # ── Model architecture ─────────────────────────────────────
    def _build_model(self, n_features: int, n_classes: int):
        model = models.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(n_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # ── Training ────────────────────────────────────────────────
    def train(self, epochs: int = 30, verbose: int = 0):
        X, y = self._generate_data()
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if self._use_fallback:
            self._train_fallback(X_train, y_train, X_val, y_val)
            self._trained = True
            return self

        self.model = self._build_model(len(self.SYMPTOM_FEATURES), len(self.classes_))
        cb_list = [
            callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ]
        self.history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=32, callbacks=cb_list, verbose=verbose,
        )
        self._trained = True
        val_acc = self.history.history['val_accuracy'][-1]
        print(f"    Neural network trained — final val_accuracy: {val_acc:.2%}")
        return self

    # ── Lightweight numpy fallback (used only if TensorFlow is unavailable) ──
    def _train_fallback(self, X_train, y_train, X_val, y_val):
        """A tiny single-hidden-layer MLP trained via manual gradient descent,
        so the module still functions in environments without TensorFlow."""
        rng = np.random.RandomState(self.random_state)
        n_in, n_hidden, n_out = X_train.shape[1], 32, len(self.classes_)
        self.W1 = rng.randn(n_in, n_hidden) * 0.1
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_out) * 0.1
        self.b2 = np.zeros(n_out)

        def softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(axis=1, keepdims=True)

        y_onehot = np.eye(n_out)[y_train]
        lr = 0.05
        for epoch in range(300):
            h = np.maximum(0, X_train @ self.W1 + self.b1)      # ReLU
            out = softmax(h @ self.W2 + self.b2)
            grad_out = (out - y_onehot) / len(X_train)
            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ self.W2.T
            grad_h[h <= 0] = 0
            grad_W1 = X_train.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)
            self.W1 -= lr * grad_W1; self.b1 -= lr * grad_b1
            self.W2 -= lr * grad_W2; self.b2 -= lr * grad_b2

        val_h = np.maximum(0, X_val @ self.W1 + self.b1)
        val_out = softmax(val_h @ self.W2 + self.b2)
        val_acc = (val_out.argmax(axis=1) == y_val).mean()
        print(f"    [fallback MLP — TensorFlow unavailable] val_accuracy: {val_acc:.2%}")

    def _fallback_predict_proba(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ── Feature conversion & prediction ─────────────────────────
    def _symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        clean = {s.lower().strip().replace(' ', '_') for s in symptoms}
        return np.array([[1.0 if s in clean else 0.0 for s in self.SYMPTOM_FEATURES]], dtype='float32')

    def predict(self, symptoms: List[str]) -> Dict:
        if not self._trained:
            raise RuntimeError("Model not trained yet — call .train() first")
        vector = self._symptoms_to_vector(symptoms)
        if self._use_fallback:
            proba = self._fallback_predict_proba(vector)[0]
        else:
            proba = self.model.predict(vector, verbose=0)[0]
        idx = int(np.argmax(proba))
        return {
            'diagnosis': self.classes_[idx],
            'confidence': round(float(proba[idx]), 4),
            'all_probabilities': {c: round(float(p), 4) for c, p in zip(self.classes_, proba)},
        }

    # ── Standard module interface (called by the Agent) ─────
    def analyze(self, patient) -> Dict:
        if not self._trained:
            self.train(epochs=20, verbose=0)
        return self.predict(patient.symptoms)

    # ── Training curve plots ──────────────────────────────────
    def plot_training(self, save_path: str = "reports/nn_training.png"):
        import matplotlib.pyplot as plt
        if self._use_fallback or self.history is None:
            print("  ⚠ No Keras training history available (fallback mode) — skipping plot.")
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(self.history.history['accuracy'], label='train')
        axes[0].plot(self.history.history['val_accuracy'], label='validation')
        axes[0].set_title('Accuracy over Epochs'); axes[0].legend()
        axes[1].plot(self.history.history['loss'], label='train')
        axes[1].plot(self.history.history['val_loss'], label='validation')
        axes[1].set_title('Loss over Epochs'); axes[1].legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✔ Saved NN training curves -> {save_path}")


# ============================================================
# Quick self-test — run: python -m modules.neural_network
# ============================================================
if __name__ == "__main__":
    nn = NeuralDiagnosticModel()
    nn.train(epochs=30, verbose=0)
    result = nn.predict(["fever", "rash", "joint_pain", "headache"])
    print(f"Diagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
