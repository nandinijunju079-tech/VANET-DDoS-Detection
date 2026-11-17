
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
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

df = pd.read_csv(
    "dataset/real/real_train_deduplicated.csv"
)

# Exclude messages_last_1s
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

X = df[numeric_features + categorical_features]
y = df["attacker"]
groups = df["sender_id"]

# Same group-splitting configuration as previous experiment
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_senders = set(groups.iloc[train_idx])
test_senders = set(groups.iloc[test_idx])

print("\n===== GROUP SPLIT WITHOUT FREQUENCY =====")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training senders:", len(train_senders))
print("Testing senders:", len(test_senders))
print("Overlapping senders:", len(train_senders & test_senders))

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

print("\nTraining model without message frequency...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

print("\n===== FINAL RESULTS =====")
print(f"Accuracy  : {accuracy_score(y_test, predictions):.6f}")
print(f"Precision : {precision_score(y_test, predictions, zero_division=0):.6f}")
print(f"Recall    : {recall_score(y_test, predictions, zero_division=0):.6f}")
print(f"F1 Score  : {f1_score(y_test, predictions, zero_division=0):.6f}")
print(f"FPR       : {fp / (fp + tn):.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nTN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)

print("\nExperiment completed.")
