
# Explainable AI-Based DoS Detection and Mitigation in VANETs

## 1. Project Overview

This project develops an Explainable Artificial Intelligence (XAI)-based framework for detecting and analyzing Denial-of-Service (DoS) attacks in Vehicular Ad Hoc Networks (VANETs).

It uses machine learning algorithms to identify malicious vehicle messages and provides understandable explanations, confidence-based classification, attack severity assessment, and prototype mitigation recommendations.

The project is evaluated using the VeReMi NextGen dataset, specifically its DoS scenario.

## 2. Project Objectives

- Detect malicious and abnormal vehicle messages.
- Compare Random Forest and XGBoost algorithms.
- Classify messages based on detection confidence.
- Explain model predictions using SHAP and LIME.
- Monitor message frequency using EWMA.
- Assess attack severity.
- Recommend suitable mitigation actions.

## 3. Main Features

- **Machine Learning:** Random Forest and XGBoost detection.
- **Confidence-Aware Detection:** Normal, Suspicious, and DoS classification.
- **Explainable AI:** SHAP global feature importance and LIME local explanations.
- **EWMA Monitoring:** Frequency-based anomaly detection.
- **Severity Assessment:** Low, Medium, and High.
- **Prototype Mitigation:** Allow, Monitor, and Filter recommendations.
- **Interactive Dashboard:** Eight sections for analyzing detection results.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| Scikit-learn | Random Forest |
| XGBoost | Machine learning |
| SHAP | Global model explanation |
| LIME | Local model explanation |
| Matplotlib | Data visualization |
| Streamlit | Interactive dashboard |

## 5. Dataset

**Dataset:** VeReMi NextGen  
**Scenario:** DoS Attack

Dataset source:
https://zenodo.org/records/19665762

Dataset documentation:
https://veremi-dataset.github.io/

The project uses receiver-level vehicle message observations. The original dataset is not included in this repository.

## 6. Model Performance

The following results were obtained from the receiver-level test evaluation.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 93.88% | 93.67% |
| Precision | 99.15% | 97.39% |
| Recall | 85.67% | 86.74% |
| F1-score | 91.92% | 91.76% |
| False Positive Rate | 0.50% | 1.59% |

## 7. Dashboard Sections

1. Overview
2. Model Performance
3. Confidence Detection
4. Explainable AI
5. EWMA Monitoring
6. Severity and Mitigation
7. Detection Comparison
8. Explore Test Records

## 8. Project Structure

```text
VANET-DDoS-Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── train_receiver_models.py
│   ├── receiver_confidence_detection.py
│   ├── receiver_real_ewma.py
│   ├── receiver_severity_mitigation.py
│   └── other source files
│
└── outputs/
    └── real/
        ├── receiver_model_evaluation.csv
        ├── receiver_shap_importance.csv
        ├── receiver_confidence_results.csv
        ├── receiver_ewma_test.csv
        ├── receiver_severity_mitigation.csv
        ├── receiver_detection_comparison.csv
        ├── receiver_shap_global.png
        ├── receiver_shap_beeswarm.png
        ├── receiver_lime_benign.png
        └── receiver_lime_attacker.png
```

## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/nandinijunju079-tech/VANET-DDoS-Detection.git
cd VANET-DDoS-Detection
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the dashboard

```bash
streamlit run app.py
```

## 10. Research Limitations

- The dashboard displays precomputed test results and is not connected to live VANET traffic.
- Evaluation focuses on the selected VeReMi NextGen DoS scenario.
- SHAP and LIME explain model predictions but do not establish causal relationships.
- EWMA and machine learning detection use related message-frequency information.
- Mitigation actions are prototype recommendations; actual network packet blocking is not implemented.

## 11. Future Enhancements

- Integration with live VANET traffic.
- Real-time attack detection and alert generation.
- Evaluation across additional attack scenarios.
- Deployment of practical network-level mitigation.
- Optimization of explanation and prediction overhead.

## 12. Author

**Junju Nandini**

B.Tech – Computer Science and Engineering  
Specialization: Cyber Security

## 13. Disclaimer

This project is developed for academic and educational purposes. Detection and mitigation results are based on the selected dataset and experimental evaluation.
