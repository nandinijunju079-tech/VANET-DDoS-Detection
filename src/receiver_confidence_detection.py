
import os
import joblib
import pandas as pd

# 1. Load trained receiver-side Random Forest
model = joblib.load(
    "models/real/random_forest_receiver.joblib"
)

# 2. Load official test dataset
test = pd.read_csv(
    "dataset/real/real_receiver_test.csv"
)

features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "messages_received_last_1s",
    "distance_to_road_edge",
    "driver_profile"
]

X_test = test[features]

# 3. Calculate DoS probabilities
probabilities = model.predict_proba(X_test)

# Find the probability column for class 1
classes = list(model.named_steps["classifier"].classes_)
attacker_column = classes.index(1)

dos_probability = probabilities[:, attacker_column]

# 4. Confidence-aware classification
def classify_confidence(probability):

    if probability < 0.30:
        return "Normal"

    elif probability < 0.70:
        return "Suspicious"

    else:
        return "DoS"


confidence_labels = [
    classify_confidence(p)
    for p in dos_probability
]

# 5. Create output
results = test.copy()

results["DoS_Probability"] = dos_probability
results["DoS_Probability_Percent"] = dos_probability * 100
results["Confidence_Classification"] = confidence_labels

results["Actual_Label"] = results["attacker"].map({
    0: "Benign",
    1: "Attacker"
})

# 6. Display classification counts
print("\n===== CONFIDENCE-AWARE DETECTION =====")

print("\nClassification counts:")
print(
    results["Confidence_Classification"]
    .value_counts()
    .reindex(["Normal", "Suspicious", "DoS"], fill_value=0)
)

print("\nClassification percentages:")
print(
    results["Confidence_Classification"]
    .value_counts(normalize=True)
    .reindex(["Normal", "Suspicious", "DoS"], fill_value=0)
    .mul(100)
    .round(2)
)

print("\nActual label vs confidence classification:")
print(
    pd.crosstab(
        results["Actual_Label"],
        results["Confidence_Classification"]
    )
)

# 7. Save results
os.makedirs("outputs/real", exist_ok=True)

output_path = (
    "outputs/real/receiver_confidence_results.csv"
)

results.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nCONFIDENCE-AWARE DETECTION COMPLETED")
