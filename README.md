# 🧬 OncoGuard AI: Advanced Cancer Prediction System

**Privacy-Preserving • Explainable • Uncertainty-Aware**

## 📌 Overview
OncoGuard AI is a patent-grade, hackathon-ready medical AI system designed to predict cancer risk with high accuracy while addressing the critical challenges of **Trust**, **Privacy**, and **Liability** in healthcare AI.

Unlike traditional black-box models, OncoGuard AI provides:
1.  **Transparent Explanations** (Why was this prediction made?)
2.  **Privacy Protection** (Federated Learning simulation)
3.  **Confidence Scores** (When should a human doctor intervene?)

## 🚀 Key Features

### 1. 🧠 Multi-Modal Data Fusion
Combines diverse data sources to mimic real-world clinical complexity:
*   **Clinical Data**: Age, BMI, Smoking History.
*   **Biomarkers**: Simulated blood protein levels.
*   **Genomics**: Polygenic Risk Scores.

### 2. 🔍 Explainable AI (XAI)
Integrated **SHAP (SHapley Additive exPlanations)** values provide real-time, patient-specific explanations. Doctors can see exactly which factors (e.g., "High Genetic Risk") drove the AI's decision.

### 3. 🛡️ Federated Learning Simulation
Demonstrates a privacy-first architecture where patient data never leaves the local device/hospital. The model learns from decentralized data without compromising patient confidentiality.

### 4. 🔮 Uncertainty Quantification
Uses **Entropy-based Uncertainty Estimation** to detect when the model is "confused". High uncertainty triggers a "Human Review Needed" alert, ensuring patient safety.

### 5. 📝 Automated Clinical Reports
Generates natural language summaries of the patient's risk profile and allows doctors to download a standardized text report for electronic health records (EHR).

### 6. 🔒 Immutable Audit Log
Simulates a blockchain-style **SHA-256 Audit Trail** to ensure data integrity and prevent tampering with patient records.

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd cancer-prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Option 1: Interactive Dashboard (Recommended)
Launch the full-featured web application for demos and clinical use.
```bash
streamlit run dashboard.py
```
*   **Input**: Adjust patient parameters using the sidebar sliders.
*   **Analyze**: Click "Run Prediction Analysis".
*   **View**: See the Risk Gauge, SHAP Explanations, Radar Charts, and download the Clinical Report.

### Option 2: Research Notebook
Explore the underlying data science, model training, and federated learning simulation.
1.  Open `Advanced_Cancer_Prediction_System.ipynb` in VS Code or Jupyter.
2.  Run all cells to generate synthetic data and train the ensemble models.

## 📂 Project Structure
*   `dashboard.py`: The main Streamlit application code.
*   `Advanced_Cancer_Prediction_System.ipynb`: Jupyter notebook for research, data generation, and model prototyping.
*   `requirements.txt`: List of Python dependencies.
*   `README.md`: Project documentation.

## 🏗️ Tech Stack
*   **Frontend**: Streamlit, Plotly
*   **Machine Learning**: Scikit-learn (Random Forest, Gradient Boosting), Numpy, Pandas
*   **Explainability**: SHAP (SHapley Additive exPlanations)
*   **Visualization**: Seaborn, Matplotlib

## 📜 License
This project is open-source and available under the MIT License.

---
*Developed for Hackathons & Research Demonstrations*