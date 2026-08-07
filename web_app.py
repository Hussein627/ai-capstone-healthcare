# ============================================================
# WEB INTERFACE (optional, not a capstone requirement)
# A Flask front-end over the exact same agent/module pipeline
# used by app.py — nothing about the underlying AI logic changes.
# ============================================================
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import json
import pandas as pd
from flask import Flask, render_template, request, jsonify

from modules.agent            import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base   import MedicalKnowledgeBase
from modules.bayesian_net     import SimpleBayesianDiagnostics
from modules.ml_classifier    import MLDiagnosticClassifier
from modules.neural_network   import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner          import TreatmentPlanner
from evaluation.metrics       import compute_classification_metrics

app = Flask(__name__)

SYMPTOM_FEATURES = [
    'fever', 'cough', 'fatigue', 'headache', 'rash', 'loss_of_smell',
    'chest_pain', 'joint_pain', 'shortness_of_breath', 'sweating',
    'body_aches', 'sore_throat', 'runny_nose', 'frequent_urination',
    'excessive_thirst', 'blurred_vision', 'nausea', 'chills',
]

SAMPLE_PATIENTS = [
    {"id": "P001", "symptoms": ["fever", "cough", "fatigue", "loss_of_smell"],
     "age": 34, "temperature": 38.9, "heart_rate": 98, "blood_pressure": "120/80"},
    {"id": "P002", "symptoms": ["fever", "rash", "joint_pain", "headache"],
     "age": 27, "temperature": 39.6, "heart_rate": 112, "blood_pressure": "110/70"},
    {"id": "P003", "symptoms": ["cough", "sore_throat", "runny_nose"],
     "age": 45, "temperature": 37.4, "heart_rate": 80, "blood_pressure": "118/76"},
    {"id": "P004", "symptoms": ["chest_pain", "shortness_of_breath", "sweating", "fatigue"],
     "age": 61, "temperature": 37.6, "heart_rate": 118, "blood_pressure": "150/95"},
    {"id": "P005", "symptoms": ["fatigue", "frequent_urination", "excessive_thirst", "blurred_vision"],
     "age": 52, "temperature": 37.0, "heart_rate": 76, "blood_pressure": "130/85"},
]

# ── Build the agent ONCE at server startup (models train once, then stay in memory) ──
_agent = None
_eval_cache = None


def get_agent() -> HealthcareDiagnosticAgent:
    global _agent
    if _agent is None:
        print("Building & training AI system (first request only)...")
        agent = HealthcareDiagnosticAgent()
        kb, bayes = MedicalKnowledgeBase(), SimpleBayesianDiagnostics()
        ml_clf, nn_model = MLDiagnosticClassifier(), NeuralDiagnosticModel()
        fuzzy, planner = FuzzySeverityAssessor(), TreatmentPlanner()

        ml_clf.train(verbose=False)
        nn_model.train(epochs=30, verbose=0)

        agent.register_module('KnowledgeBase', kb)
        agent.register_module('BayesianNet', bayes)
        agent.register_module('MLClassifier', ml_clf)
        agent.register_module('NeuralNetwork', nn_model)
        agent.register_module('FuzzySeverity', fuzzy)
        agent.planner = planner
        _agent = agent
        print("AI system ready.")
    return _agent


def get_evaluation():
    """Computed once, cached — same methodology as app.py's run_evaluation()."""
    global _eval_cache
    if _eval_cache is not None:
        return _eval_cache

    agent = get_agent()
    df = pd.read_csv("data/patient_records.csv")
    sample = df.sample(n=min(150, len(df)), random_state=7)
    symptom_cols = [c for c in df.columns if c not in
                    ('patient_id', 'age', 'temperature', 'heart_rate', 'diagnosis')]

    accuracies = {}
    for module_name in ['MLClassifier', 'NeuralNetwork', 'BayesianNet', 'KnowledgeBase']:
        module = agent.modules[module_name]
        y_true, y_pred = [], []
        for _, row in sample.iterrows():
            symptoms = [c for c in symptom_cols if row[c] == 1]
            patient = PatientPercept(row['patient_id'], symptoms,
                                      age=int(row['age']), temperature=float(row['temperature']),
                                      heart_rate=int(row['heart_rate']))
            try:
                result = module.analyze(patient)
                pred = result.get('diagnosis', 'unknown')
            except Exception:
                pred = 'unknown'
            y_true.append(row['diagnosis'])
            y_pred.append(pred)
        metrics = compute_classification_metrics(y_true, y_pred)
        accuracies[module_name] = round(metrics['accuracy'], 4)

    _eval_cache = accuracies
    return _eval_cache


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", symptoms=SYMPTOM_FEATURES, samples=SAMPLE_PATIENTS)


@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    data = request.get_json(force=True)
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "Select at least one symptom."}), 400

    patient = PatientPercept(
        patient_id=data.get("patient_id", "WEB-001"),
        symptoms=symptoms,
        age=int(data.get("age", 30)),
        temperature=float(data.get("temperature", 37.0)),
        heart_rate=int(data.get("heart_rate", 75)),
        blood_pressure=data.get("blood_pressure", "120/80"),
    )

    agent = get_agent()
    report = agent.diagnose(patient)

    severity = agent.modules['FuzzySeverity'].assess(
        patient.temperature, patient.heart_rate, len(patient.symptoms)
    )
    report['severity'] = severity

    plan = agent.planner.create_treatment_plan(report['diagnosis'], severity['severity_label'])
    report['treatment_plan'] = plan
    report['patient'] = {
        "id": patient.patient_id, "symptoms": patient.symptoms, "age": patient.age,
        "temperature": patient.temperature, "heart_rate": patient.heart_rate,
        "blood_pressure": patient.blood_pressure,
    }

    # Strip non-JSON-serializable 'raw' dataclass objects from module opinions
    clean_opinions = {}
    for name, res in report['module_opinions'].items():
        clean_opinions[name] = {k: v for k, v in res.items() if k != 'raw'}
    report['module_opinions'] = clean_opinions

    return jsonify(report)


@app.route("/api/evaluation")
def api_evaluation():
    return jsonify(get_evaluation())


if __name__ == "__main__":
    print("Pre-training AI system before server starts...")
    get_agent()
    print("\n  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
