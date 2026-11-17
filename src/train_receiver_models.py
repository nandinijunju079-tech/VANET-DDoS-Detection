
import os
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

from xgboost import XGBClassifier


# -----------------------------------------
# 1. LOAD RECEIVER-LEVEL DATASETS
# -----------------------------------------

BASE = "dataset/real"

train = pd.read_csv(f"{BASE}/real_receiver_train.csv")
validation = pd.read_csv(f"{BASE}/real_receiver_validation.csv")
test = pd.read_csv(f"{BASE}/real_receiver_test.csv")

# Exclude identifiers and audit columns.
numeric_features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "messages_received_last_1s",
    "distance_to_road_edge"
]

categorical_features = ["driver_profile"]

features = numeric_features + categorical_features
target = "attacker"

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]


# -----------------------------------------
# 2. PREPROCESSING
# -----------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# -----------------------------------------
# 3. DEFINE MODELS
# -----------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# -----------------------------------------
# 4. EVALUATION FUNCTION
# -----------------------------------------

def evaluate(model, X, y, model_name, split_name):

    start = time.perf_counter()
    predictions = model.predict(X)
    elapsed = time.perf_counter() - start

    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n{model_name} - {split_name}")
    print("Confusion Matrix:")
    print(matrix)
    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1 Score  : {f1:.6f}")
    print(f"FPR       : {fpr:.6f}")
    print(f"Total prediction time: {elapsed:.6f} seconds")
    print(f"Average latency: {elapsed / len(X) * 1000:.6f} ms/record")

    return {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Prediction_Time_Seconds": elapsed,
        "Average_Latency_ms": elapsed / len(X) * 1000
    }


# -----------------------------------------
# 5. TRAIN, VALIDATE AND TEST
# -----------------------------------------

results = []
os.makedirs("models/real", exist_ok=True)
os.makedirs("outputs/real", exist_ok=True)

for name, classifier in models.items():

    print(f"\n{'=' * 50}")
    print("TRAINING:", name)
    print(f"{'=' * 50}")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    # Validation evaluation
    results.append(
        evaluate(pipeline, X_val, y_val, name, "Validation")
    )

    # Refit using Train + Validation
    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)

    pipeline.fit(combined_X, combined_y)

    # Final test evaluation
    results.append(
        evaluate(pipeline, X_test, y_test, name, "Test")
    )

    filename = (
        "random_forest_receiver.joblib"
        if name == "Random Forest"
        else "xgboost_receiver.joblib"
    )

    joblib.dump(pipeline, f"models/real/{filename}")


# -----------------------------------------
# 6. SAVE RESULTS
# -----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/real/receiver_model_evaluation.csv",
    index=False
)

print("\n===== RECEIVER MODEL EVALUATION COMPLETE =====")
print(results_df.to_string(index=False))
print("\nResults saved to outputs/real/receiver_model_evaluation.csv")
