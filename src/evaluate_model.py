
import joblib
import numpy as np
import pandas as pd
import time
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load model and test data
model = joblib.load("models/random_forest_model.joblib")
data = joblib.load("models/preprocessed_data.joblib")

X_test = data["X_test"]
y_test = data["y_test"]

# 2. Predict test data
start_time = time.perf_counter()
y_pred = model.predict(X_test)
end_time = time.perf_counter()

prediction_time = end_time - start_time

# 3. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# 4. Calculate False Positive Rate
tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# 5. Calculate prediction latency
latency_ms = (prediction_time / len(X_test)) * 1000

# 6. Display results
print("\n----- RANDOM FOREST EVALUATION -----")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"False Positive Rate: {fpr * 100:.2f}%")
print(f"Total Prediction Time: {prediction_time:.6f} seconds")
print(f"Average Prediction Latency: {latency_ms:.6f} ms/record")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "DDoS"],
    zero_division=0
))

# 7. Save evaluation results
os.makedirs("outputs", exist_ok=True)

results = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "False_Positive_Rate": fpr,
    "Total_Prediction_Time_Seconds": prediction_time,
    "Average_Latency_ms_per_record": latency_ms
}])

results.to_csv(
    "outputs/model_evaluation.csv",
    index=False
)

print("\nEvaluation saved to outputs/model_evaluation.csv")
