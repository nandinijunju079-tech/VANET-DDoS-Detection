
import os
import pandas as pd
import numpy as np

OUTPUT_DIR = "outputs/real"

confidence_path = os.path.join(
    OUTPUT_DIR,
    "receiver_confidence_results.csv"
)

ewma_path = os.path.join(
    OUTPUT_DIR,
    "receiver_ewma_test.csv"
)

confidence = pd.read_csv(confidence_path)
ewma = pd.read_csv(ewma_path)

# Match the same receiver-message observations
keys = [
    "receiver_id",
    "sender_id",
    "messageID",
    "sendTime"
]

for df in [confidence, ewma]:
    for col in keys:
        df[col] = df[col].astype(str)

if confidence.duplicated(keys).any():
    raise ValueError("Duplicate message identities in confidence data")

if ewma.duplicated(keys).any():
    raise ValueError("Duplicate message identities in EWMA data")

results = confidence.merge(
    ewma[keys + ["EWMA_Frequency", "EWMA_Anomaly"]],
    on=keys,
    how="inner",
    validate="one_to_one"
)

if len(results) != len(confidence) or len(results) != len(ewma):
    raise ValueError(
        "Confidence and EWMA records do not match. "
        "Check the message identity columns."
    )

# Severity score: 70% model probability + 30% EWMA signal
results["EWMA_Anomaly_Flag"] = (
    results["EWMA_Anomaly"] == "Anomaly"
).astype(int)

results["Severity_Score"] = (
    0.70 * results["DoS_Probability_Percent"]
    + 0.30 * results["EWMA_Anomaly_Flag"] * 100
)

def assign_severity(score):
    if score < 30:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

results["Severity"] = results["Severity_Score"].apply(
    assign_severity
)

# Prototype mitigation decisions
def mitigation_decision(row):

    probability = row["DoS_Probability"]
    severity = row["Severity"]

    if probability >= 0.70 and severity == "High":
        return "FILTER"

    elif (
        probability >= 0.30
        or severity == "Medium"
        or row["EWMA_Anomaly_Flag"] == 1
    ):
        return "MONITOR"

    else:
        return "ALLOW"

results["Mitigation_Action"] = results.apply(
    mitigation_decision,
    axis=1
)

# Display results
print("\n===== SEVERITY DISTRIBUTION =====")
print(results["Severity"].value_counts())

print("\n===== MITIGATION DISTRIBUTION =====")
print(results["Mitigation_Action"].value_counts())

print("\n===== ACTUAL LABEL VS SEVERITY =====")
print(
    pd.crosstab(
        results["Actual_Label"],
        results["Severity"]
    )
)

print("\n===== ACTUAL LABEL VS MITIGATION =====")
print(
    pd.crosstab(
        results["Actual_Label"],
        results["Mitigation_Action"]
    )
)

# Save complete results
output_path = os.path.join(
    OUTPUT_DIR,
    "receiver_severity_mitigation.csv"
)

results.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("Total matched records:", len(results))
print("\nSEVERITY AND MITIGATION COMPLETED.")
