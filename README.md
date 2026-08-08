# 🏥 Intelligent Healthcare Diagnostic Assistant

**Introduction to AI — Capstone Project**  
**Dedan Kimathi University of Technology (DeKUT)**  

An advanced, multi-paradigm Artificial Intelligence system designed to simulate a comprehensive medical diagnostic pipeline. This project integrates 9 distinct AI methodologies into a single, cohesive agent capable of processing patient vitals, inferring risks, and recommending optimal treatment pathways.

---

## 🧠 System Architecture & AI Paradigms

This system successfully demonstrates the decoupling of logic, machine learning, and evaluation through a modular architecture:

1. **Intelligent Agent (PEAS)**: The central orchestrator that perceives data and coordinates modules.
2. **Natural Language Processing (NLP)**: Cleans and extracts actionable symptoms from raw patient text.
3. **Search Algorithms (BFS)**: Traverses medical risk pathways to identify critical escalation routes.
4. **Knowledge Base (FOL)**: A First-Order Logic inference engine using forward-chaining rules.
5. **Bayesian Network**: A probabilistic reasoning model using dynamic priors and conditional probabilities.
6. **Machine Learning Classifier**: An ensemble of Decision Trees, Random Forests, and Gradient Boosting.
7. **Deep Neural Network (DNN)**: A multi-layer architecture built with TensorFlow/Keras.
8. **Fuzzy Logic Controller**: Mathematically maps patient vitals to a severity index.
9. **Reinforcement Learning (RL)**: Uses Q-Table policies to optimize and recommend final treatment actions.

---

## 📁 Project Structure

```text
Capstone-Project/
│
├── app.py                      # Main application & interactive terminal UI
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── data/
│   └── patient_records.csv     # Persistent database for patient histories
│
├── evaluation/
│   ├── metrics.py              # Mathematical grading (Precision, Recall, F1)
│   └── visualizations.py       # Generation of Confusion Matrices and Training Curves
│
├── reports/
│   └── final_report.pdf        # Written documentation and final paper
│
└── modules/
    ├── agent.py                # Core PEAS Agent
    ├── nlp_processor.py        # Text processing module
    ├── search.py               # BFS risk pathway search
    ├── knowledge_base.py       # First-Order Logic module
    ├── bayesian_net.py         # Probabilistic reasoning
    ├── ml_classifier.py        # Supervised ML ensemble
    ├── neural_network.py       # TensorFlow/Keras DNN
    ├── fuzzy_controller.py     # Vitals severity mapping
    ├── planner.py              # STRIPS-based treatment planner
    └── rl_agent.py             # Q-Learning treatment optimizer
    ⚙️ Installation & Setup
Initialize a Virtual Environment:

Bash
python3 -m venv venv
source venv/bin/activate
Install Dependencies:

Bash
pip install -r requirements.txt
🚀 Usage Guide
1. Standard Patient Diagnostics
To run the interactive diagnostic assistant, start the main application:

Bash
python3 app.py
Follow the terminal prompts to enter the Patient ID, symptoms, age, temperature, heart rate, and blood pressure. The agent will process the percepts, output a JSON Diagnostic Report, save the record to the database, and display an internal thought process log.

2. AI Model Evaluation
To generate academic performance metrics and visual charts for the ML and DNN models, run the application and type evaluate at the ID prompt:

Bash
Patient ID (Format e.g., PT-1234) > evaluate
This will automatically evaluate the models against synthetic datasets and generate ml_evaluation.png, nn_training.png, and module_comparison.png in the evaluation/ directory.

📊 Evaluation Results
The system's modular architecture was evaluated against a synthetic dataset of 3,000 patient records. The performance metrics demonstrated the efficacy of combining different AI paradigms:

Deep Neural Network (DNN): Achieved ~90.8% validation accuracy over 50 epochs, excelling at complex, non-linear symptom overlapping.

Machine Learning Ensemble: The Gradient Boosting classifier achieved the highest standalone accuracy at ~92.3%, effectively mapping rigid feature sets.

Bayesian & Logic Nets: Provided highly interpretable, probabilistic baselines (85-89%) essential for transparent medical decision-making.

Conclusion: The RL-driven agent successfully utilizes these overlapping diagnostic inputs to recommend safe, optimized patient treatment pathways (e.g., ICU Admission vs. Home Rest).