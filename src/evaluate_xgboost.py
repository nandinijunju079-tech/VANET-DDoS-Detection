
import joblib
import time
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load XGBoost model and test data
model = joblib.load("models/xgboost_model.joblib")
data = joblib.load("models/preprocessed_data.joblib")

X_test = data["X_test"]
y_test = data["y_test"]

# Prediction time
start = time.perf_counter()
y_pred = model.predict(X_test)
end = time.perf_counter()

total_time = end - start

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0

latency = (total_time / len(X_test)) * 1000

print("\n----- XGBOOST EVALUATION -----")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"False Positive Rate: {fpr * 100:.2f}%")
print(f"Total Prediction Time: {total_time:.6f} seconds")
print(f"Average Latency: {latency:.6f} ms/record")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# Save results
os.makedirs("outputs", exist_ok=True)

result = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "False_Positive_Rate": fpr,
    "Total_Prediction_Time_Seconds": total_time,
    "Average_Latency_ms_per_record": latency
}])

result.to_csv(
    "outputs/xgboost_evaluation.csv",
    index=False
)

print("\nXGBoost evaluation saved successfully!")
