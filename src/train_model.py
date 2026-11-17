
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load preprocessed data
data = joblib.load("models/preprocessed_data.joblib")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# 2. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

# 3. Train the model
print("Training Random Forest Model...")

rf_model.fit(X_train, y_train)

# 4. Predict test data
y_pred = rf_model.predict(X_test)

# 5. Calculate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n----- MODEL PERFORMANCE -----")

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

# 6. Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "DDoS"]
    )
)

# 7. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    rf_model,
    "models/random_forest_model.joblib"
)

print("\nRandom Forest Model Trained Successfully!")
print("Model saved in models/random_forest_model.joblib")
