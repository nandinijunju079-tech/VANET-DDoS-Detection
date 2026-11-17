
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
DATA_PATH = BASE_DIR / "models" / "preprocessed_data.joblib"
SEVERITY_PATH = BASE_DIR / "outputs" / "severity_results.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "mitigation_results.csv"

# Load trained model and test data
model = joblib.load(MODEL_PATH)
data = joblib.load(DATA_PATH)

X_test = data["X_test"]

# Load severity results
severity_data = pd.read_csv(SEVERITY_PATH)

# Check that rows match
if len(severity_data) != len(X_test):
    raise ValueError(
        "Mismatch between severity results and test records."
    )

# Predict DDoS probability
probabilities = model.predict_proba(X_test)[:, 1]

results = []

for i in range(len(X_test)):

    probability = probabilities[i]
    severity = severity_data.iloc[i]["Attack_Severity"]
    ewma_status = severity_data.iloc[i]["EWMA_Anomaly"]

    # Assumption: emergency_priority = 1 means emergency vehicle
    emergency = int(X_test.iloc[i]["emergency_priority"]) == 1

    if emergency:
        if probability >= 0.70 or ewma_status == "Anomaly":
            decision = "PRIORITIZE_AND_MONITOR"
        else:
            decision = "ALLOW_WITH_PRIORITY"

    elif probability >= 0.70 and severity == "High":
        decision = "FILTER"

    elif (
        probability >= 0.30
        or severity == "Medium"
        or ewma_status == "Anomaly"
    ):
        decision = "MONITOR"

    else:
        decision = "ALLOW"

    results.append({
        "Record_ID": i + 1,
        "DDoS_Probability": round(probability * 100, 2),
        "Severity": severity,
        "EWMA_Status": ewma_status,
        "Emergency_Vehicle": emergency,
        "Mitigation_Decision": decision
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n----- EXPLAINABLE MITIGATION RESULTS -----")
print(results_df.head(15).to_string(index=False))

print("\n----- MITIGATION SUMMARY -----")
print(results_df["Mitigation_Decision"].value_counts())

print("\nResults saved to:", OUTPUT_PATH)
