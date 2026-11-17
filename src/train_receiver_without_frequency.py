
import os
import time
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from xgboost import XGBClassifier

BASE = "dataset/real"

train = pd.read_csv(f"{BASE}/real_receiver_train.csv")
validation = pd.read_csv(f"{BASE}/real_receiver_validation.csv")
test = pd.read_csv(f"{BASE}/real_receiver_test.csv")

# Remove receiver-frequency feature for this experiment.
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
features = numeric_features + categorical_features

X_train, y_train = train[features], train["attacker"]
X_val, y_val = validation[features], validation["attacker"]
X_test, y_test = test[features], test["attacker"]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

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

def evaluate(model, X, y, model_name, split_name):
    start = time.perf_counter()
    pred = model.predict(X)
    elapsed = time.perf_counter() - start

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    result = {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Prediction_Time_Seconds": elapsed,
        "Average_Latency_ms": elapsed / len(X) * 1000
    }

    print(f"\n{model_name} - {split_name}")
    print("Confusion Matrix:")
    print([[tn, fp], [fn, tp]])

    for key in ["Accuracy", "Precision", "Recall", "F1", "FPR"]:
        print(f"{key}: {result[key]:.6f}")

    print(f"Prediction time: {elapsed:.6f} seconds")
    print(f"Average latency: {result['Average_Latency_ms']:.6f} ms/record")

    return result


results = []

for name, classifier in models.items():
    print(f"\n{'=' * 45}")
    print("TRAINING WITHOUT RECEIVER FREQUENCY:", name)
    print(f"{'=' * 45}")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    results.append(
        evaluate(pipeline, X_val, y_val, name, "Validation")
    )

    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)

    pipeline.fit(combined_X, combined_y)

    results.append(
        evaluate(pipeline, X_test, y_test, name, "Test")
    )


os.makedirs("outputs/real", exist_ok=True)

result_df = pd.DataFrame(results)
result_df.to_csv(
    "outputs/real/receiver_without_frequency_evaluation.csv",
    index=False
)

print("\n===== RESULTS WITHOUT RECEIVER FREQUENCY =====")
print(result_df.to_string(index=False))
print("\nSaved: outputs/real/receiver_without_frequency_evaluation.csv")
