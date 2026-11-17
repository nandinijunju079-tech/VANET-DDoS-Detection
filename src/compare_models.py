
import joblib
import os

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load preprocessed data
data = joblib.load("models/preprocessed_data.joblib")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# Train XGBoost
print("Training XGBoost Model...")

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# Predict test data
y_pred = xgb_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n----- XGBOOST PERFORMANCE -----")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    xgb_model,
    "models/xgboost_model.joblib"
)

print("\nXGBoost Model Trained Successfully!")
print("Model saved in models/xgboost_model.joblib")

# Compare with Random Forest
rf_model = joblib.load("models/random_forest_model.joblib")
rf_pred = rf_model.predict(X_test)

print("\n----- MODEL COMPARISON -----")

print("Algorithm       Accuracy       F1 Score")
print("---------------------------------------")

print(
    "Random Forest  ",
    round(accuracy_score(y_test, rf_pred) * 100, 2),
    "%",
    round(f1_score(y_test, rf_pred) * 100, 2),
    "%"
)

print(
    "XGBoost        ",
    round(accuracy * 100, 2),
    "%",
    round(f1 * 100, 2),
    "%"
)
