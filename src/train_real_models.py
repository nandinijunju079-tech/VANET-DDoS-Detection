
import os
import time
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ------------------------------------------
# 1. PATHS
# ------------------------------------------

DATA = "dataset/real"
MODEL_DIR = "models/real"
OUTPUT_DIR = "outputs/real"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------
# 2. LOAD DATA
# ------------------------------------------

train = pd.read_csv(
    os.path.join(DATA, "real_train_deduplicated.csv")
)

validation = pd.read_csv(
    os.path.join(DATA, "real_validation_deduplicated.csv")
)

test = pd.read_csv(
    os.path.join(DATA, "real_test_deduplicated.csv")
)

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)


# ------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------

numeric_features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "messages_last_1s",
    "distance_to_road_edge"
]

categorical_features = [
    "driver_profile"
]

features = numeric_features + categorical_features

X_train = train[features]
y_train = train["attacker"]

X_val = validation[features]
y_val = validation["attacker"]

X_test = test[features]
y_test = test["attacker"]


# ------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ------------------------------------------
# 5. DEFINE MODELS
# ------------------------------------------

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


# ------------------------------------------
# 6. EVALUATION FUNCTION
# ------------------------------------------

def evaluate_model(name, model, X, y, stage):

    start = time.perf_counter()

    predictions = model.predict(X)

    prediction_time = time.perf_counter() - start

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    result = {
        "Model": name,
        "Stage": stage,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y, predictions, zero_division=0
        ),
        "False_Positive_Rate": fpr,
        "Prediction_Time_Seconds": prediction_time,
        "Average_Latency_ms": (
            prediction_time / len(X)
        ) * 1000
    }

    print("\n", "=" * 45)
    print(name, "-", stage)
    print("=" * 45)

    print("Confusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["Benign", "Attacker"],
            zero_division=0
        )
    )

    print("Accuracy:", round(result["Accuracy"], 4))
    print("Precision:", round(result["Precision"], 4))
    print("Recall:", round(result["Recall"], 4))
    print("F1 Score:", round(result["F1_Score"], 4))
    print("FPR:", round(result["False_Positive_Rate"], 4))
    print("Prediction time:", round(prediction_time, 4), "seconds")
    print("Average latency:", round(result["Average_Latency_ms"], 4), "ms")

    return result


# ------------------------------------------
# 7. TRAIN AND VALIDATE BOTH MODELS
# ------------------------------------------

validation_results = []

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    print("\nTraining:", name)

    pipeline.fit(X_train, y_train)

    result = evaluate_model(
        name,
        pipeline,
        X_val,
        y_val,
        "Validation"
    )

    validation_results.append(result)


# ------------------------------------------
# 8. REFIT ON TRAIN + VALIDATION
# ------------------------------------------

combined = pd.concat(
    [train, validation],
    ignore_index=True
)

X_combined = combined[features]
y_combined = combined["attacker"]

test_results = []

for name, classifier in models.items():

    final_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    print("\nFinal training:", name)

    final_pipeline.fit(X_combined, y_combined)

    # Save trained pipeline
    safe_name = name.lower().replace(" ", "_")

    model_path = os.path.join(
        MODEL_DIR,
        safe_name + "_real.joblib"
    )

    joblib.dump(final_pipeline, model_path)

    print("Saved model:", model_path)

    result = evaluate_model(
        name,
        final_pipeline,
        X_test,
        y_test,
        "Test"
    )

    test_results.append(result)


# ------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------

all_results = pd.DataFrame(
    validation_results + test_results
)

output_file = os.path.join(
    OUTPUT_DIR,
    "real_model_evaluation.csv"
)

all_results.to_csv(output_file, index=False)

print("\nFINAL MODEL EVALUATION")
print(all_results.to_string(index=False))

print("\nResults saved:", output_file)

print("\nREAL DATASET MODEL TRAINING COMPLETED")
