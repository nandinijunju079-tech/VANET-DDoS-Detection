
import time
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load deduplicated datasets
train = pd.read_csv("dataset/real/real_train_deduplicated.csv")
validation = pd.read_csv("dataset/real/real_validation_deduplicated.csv")
test = pd.read_csv("dataset/real/real_test_deduplicated.csv")

# Combine training and validation data
train_data = pd.concat(
    [train, validation],
    ignore_index=True
)

# Remove message frequency from model inputs
numeric_features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "distance_to_road_edge"
]

categorical_features = ["driver_profile"]

X_train = train_data[numeric_features + categorical_features]
y_train = train_data["attacker"]

X_test = test[numeric_features + categorical_features]
y_test = test["attacker"]

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# Random Forest model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", rf)
])

# Train
print("\nTraining Random Forest WITHOUT messages_last_1s...")

model.fit(X_train, y_train)

# Evaluate
start_time = time.perf_counter()
predictions = model.predict(X_test)
end_time = time.perf_counter()

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
fpr = fp / (fp + tn) if (fp + tn) else 0

total_time = end_time - start_time
avg_latency = (total_time / len(X_test)) * 1000

print("\n===== RF WITHOUT MESSAGE FREQUENCY =====")
print(f"Accuracy  : {accuracy:.6f} ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1 Score  : {f1:.6f}")
print(f"FPR       : {fpr:.6f}")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Total prediction time: {total_time:.6f} seconds")
print(f"Average latency: {avg_latency:.6f} ms/record")

# Save results
results = pd.DataFrame([{
    "Model": "Random Forest without messages_last_1s",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "False_Positive_Rate": fpr,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "Total_Prediction_Time_Seconds": total_time,
    "Average_Latency_ms": avg_latency
}])

results.to_csv(
    "outputs/real/rf_without_frequency_evaluation.csv",
    index=False
)

print("\nSaved: outputs/real/rf_without_frequency_evaluation.csv")
