
import os
import joblib
import pandas as pd
import numpy as np

# -----------------------------------------
# 1. CREATE OUTPUT FOLDER
# -----------------------------------------

os.makedirs("outputs", exist_ok=True)

# -----------------------------------------
# 2. LOAD TRAINED MODEL AND DATA
# -----------------------------------------

rf_model = joblib.load(
    "models/random_forest_model.joblib"
)

data = joblib.load(
    "models/preprocessed_data.joblib"
)

X_test = data["X_test"]

# Convert to DataFrame if needed
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(
        X_test,
        columns=[
            "vehicle_speed",
            "vehicle_density",
            "neighbor_count",
            "packet_rate",
            "packet_size",
            "flow_duration",
            "protocol",
            "emergency_priority"
        ]
    )

# -----------------------------------------
# 3. LOAD EWMA RESULTS
# -----------------------------------------

ewma_data = pd.read_csv(
    "outputs/ewma_monitoring_results.csv"
)

# -----------------------------------------
# 4. CALCULATE DDoS PROBABILITY
# -----------------------------------------

probabilities = rf_model.predict_proba(X_test)

ddos_index = list(rf_model.classes_).index(1)

ddos_probability = probabilities[:, ddos_index]
# -----------------------------------------
# 5. GET CORRECTLY ALIGNED EWMA INFORMATION
# -----------------------------------------

ewma_anomaly = (
    ewma_data.iloc[X_test.index]["Anomaly"]
    .reset_index(drop=True)
)
# 6. CALCULATE SEVERITY SCORE
# -----------------------------------------

results = X_test.reset_index(drop=True).copy()

results["DDoS_Probability"] = ddos_probability * 100

results["EWMA_Anomaly"] = ewma_anomaly

# Severity score:
# 70% model probability
# 30% EWMA anomaly indicator

results["Severity_Score"] = (
    0.70 * results["DDoS_Probability"]
    + 0.30 * (
        results["EWMA_Anomaly"] == "Anomaly"
    ).astype(int) * 100
)

# -----------------------------------------
# 7. CLASSIFY SEVERITY
# -----------------------------------------

def classify_severity(score):

    if score < 30:
        return "Low"

    elif score < 70:
        return "Medium"

    else:
        return "High"


results["Attack_Severity"] = (
    results["Severity_Score"]
    .apply(classify_severity)
)

# -----------------------------------------
# 8. DISPLAY RESULTS
# -----------------------------------------

print("\n----- ATTACK SEVERITY CLASSIFICATION -----")

print(
    results[
        [
            "DDoS_Probability",
            "EWMA_Anomaly",
            "Severity_Score",
            "Attack_Severity"
        ]
    ].head(20).to_string(index=False)
)

print("\nSeverity Distribution:")

print(
    results["Attack_Severity"].value_counts()
)

# -----------------------------------------
# 9. SAVE RESULTS
# -----------------------------------------

results.to_csv(
    "outputs/severity_results.csv",
    index=False
)

print("\nSeverity Classification Completed Successfully!")
