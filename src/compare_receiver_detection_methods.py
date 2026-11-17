
import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

INPUT = "outputs/real/receiver_severity_mitigation.csv"

df = pd.read_csv(INPUT)

actual = df["attacker"].astype(int)

# 1. Random Forest: default binary probability threshold
rf_prediction = (
    df["DoS_Probability"] >= 0.50
).astype(int)

# 2. EWMA: anomaly means positive detection
ewma_prediction = (
    df["EWMA_Anomaly"] == "Anomaly"
).astype(int)

# 3. Combined mitigation: Monitor or Filter means alert
combined_prediction = (
    df["Mitigation_Action"] != "ALLOW"
).astype(int)

# 4. Filter-only decision
filter_prediction = (
    df["Mitigation_Action"] == "FILTER"
).astype(int)


def evaluate(name, prediction):

    tn, fp, fn, tp = confusion_matrix(
        actual,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "Method": name,
        "Accuracy": accuracy_score(actual, prediction),
        "Precision": precision_score(
            actual, prediction, zero_division=0
        ),
        "Recall": recall_score(
            actual, prediction, zero_division=0
        ),
        "F1": f1_score(
            actual, prediction, zero_division=0
        ),
        "FPR": fp / (fp + tn) if (fp + tn) else 0,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


results = [
    evaluate("Random Forest", rf_prediction),
    evaluate("EWMA", ewma_prediction),
    evaluate("Combined Alert", combined_prediction),
    evaluate("Filter Only", filter_prediction)
]

comparison = pd.DataFrame(results)

print("\n===== DETECTION METHOD COMPARISON =====")
print(comparison.to_string(index=False))

output = "outputs/real/receiver_detection_comparison.csv"
comparison.to_csv(output, index=False)

print("\nSaved:", output)
print("\nDETECTION COMPARISON COMPLETED.")
