
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATA = "dataset/real"
MODEL_DIR = "models/real"

train = pd.read_csv(
    os.path.join(DATA, "real_train_deduplicated.csv")
)

validation = pd.read_csv(
    os.path.join(DATA, "real_validation_deduplicated.csv")
)

test = pd.read_csv(
    os.path.join(DATA, "real_test_deduplicated.csv")
)

# Sender IDs already encountered during training
known_senders = (
    set(train["sender_id"].astype(str))
    | set(validation["sender_id"].astype(str))
)

test["sender_id"] = test["sender_id"].astype(str)

known_mask = test["sender_id"].isin(known_senders)
unseen_mask = ~known_mask

print("\n========== SENDER OVERLAP ==========")

print("Train unique senders:",
      train["sender_id"].nunique())

print("Validation unique senders:",
      validation["sender_id"].nunique())

print("Test unique senders:",
      test["sender_id"].nunique())

print("Test records from known senders:",
      int(known_mask.sum()))

print("Test records from unseen senders:",
      int(unseen_mask.sum()))

features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "messages_last_1s",
    "driver_profile",
    "distance_to_road_edge"
]


def evaluate_subset(model, subset, name):

    if len(subset) == 0:
        print("\n", name, ": No records available")
        return

    X = subset[features]
    y = subset["attacker"]

    predictions = model.predict(X)

    matrix = confusion_matrix(
        y, predictions, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    print("\n==========", name, "==========")
    print("Records:", len(subset))
    print("Benign:", int((y == 0).sum()))
    print("Attacker:", int((y == 1).sum()))

    print("Confusion matrix:")
    print(matrix)

    print("Accuracy:", round(
        accuracy_score(y, predictions), 4
    ))

    print("Precision:", round(
        precision_score(y, predictions, zero_division=0), 4
    ))

    print("Recall:", round(
        recall_score(y, predictions, zero_division=0), 4
    ))

    print("F1-score:", round(
        f1_score(y, predictions, zero_division=0), 4
    ))

    print("FPR:", round(fpr, 4))


models = {
    "Random Forest": "random_forest_real.joblib",
    "XGBoost": "xgboost_real.joblib"
}

for name, filename in models.items():

    model_path = os.path.join(MODEL_DIR, filename)
    model = joblib.load(model_path)

    print("\n\n################################")
    print("MODEL:", name)
    print("################################")

    evaluate_subset(
        model,
        test,
        "All Test Records"
    )

    evaluate_subset(
        model,
        test[known_mask],
        "Known Sender Test Records"
    )

    evaluate_subset(
        model,
        test[unseen_mask],
        "Unseen Sender Test Records"
    )

print("\nSENDER GENERALIZATION ANALYSIS COMPLETED")
