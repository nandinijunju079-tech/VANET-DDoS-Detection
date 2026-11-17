
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Configuration
INPUT_DIR = "dataset/real"
OUTPUT_DIR = "outputs/real"

ALPHA = 0.3
THRESHOLD_MULTIPLIER = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_ewma(df):
    df = df.copy()

    df["EWMA_Frequency"] = np.nan

    # Process each receiver-sender stream independently
    for (receiver, sender), group in df.groupby(
        ["receiver_id", "sender_id"]
    ):
        group = group.sort_values("rcvTime")

        ewma_values = group[
            "messages_received_last_1s"
        ].ewm(
            alpha=ALPHA,
            adjust=False
        ).mean()

        df.loc[group.index, "EWMA_Frequency"] = (
            ewma_values.values
        )

    return df


def load_dataset(split):
    path = os.path.join(
        INPUT_DIR,
        f"real_receiver_{split}_with_time.csv"
    )

    df = pd.read_csv(path)

    required = [
        "receiver_id",
        "sender_id",
        "rcvTime",
        "messages_received_last_1s",
        "attacker"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    return df


print("\nLoading receiver datasets...")

train = load_dataset("train")
validation = load_dataset("validation")
test = load_dataset("test")

print("Train records:", len(train))
print("Validation records:", len(validation))
print("Test records:", len(test))


print("\nCalculating EWMA values...")

train = calculate_ewma(train)
validation = calculate_ewma(validation)
test = calculate_ewma(test)


# Learn threshold from benign training observations only
benign_train = train.loc[
    train["attacker"] == 0,
    "EWMA_Frequency"
].dropna()

baseline_mean = benign_train.mean()
baseline_std = benign_train.std(ddof=0)

threshold = (
    baseline_mean +
    THRESHOLD_MULTIPLIER * baseline_std
)

print("\n========== EWMA BASELINE ==========")
print(f"Alpha                 : {ALPHA}")
print(f"Benign baseline mean  : {baseline_mean:.4f}")
print(f"Benign baseline std   : {baseline_std:.4f}")
print(f"Threshold             : {threshold:.4f}")


def evaluate_split(df, split_name):

    df = df.copy()

    df["EWMA_Anomaly"] = np.where(
        df["EWMA_Frequency"] > threshold,
        "Anomaly",
        "Normal"
    )

    df["EWMA_Prediction"] = (
        df["EWMA_Frequency"] > threshold
    ).astype(int)

    actual = df["attacker"]
    predicted = df["EWMA_Prediction"]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(
        actual, predicted, zero_division=0
    )
    recall = recall_score(
        actual, predicted, zero_division=0
    )
    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0

    print(f"\n========== {split_name.upper()} RESULTS ==========")
    print("Confusion Matrix:")
    print([[tn, fp], [fn, tp]])

    print(f"Accuracy  : {accuracy:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")
    print(f"F1 Score  : {f1:.6f}")
    print(f"FPR       : {fpr:.6f}")

    print("\nEWMA classification counts:")
    print(df["EWMA_Anomaly"].value_counts())

    print("\nActual label vs EWMA:")
    print(
        pd.crosstab(
            df["attacker"].map({
                0: "Benign",
                1: "Attacker"
            }),
            df["EWMA_Anomaly"]
        )
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"receiver_ewma_{split_name}.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)

    return {
        "Split": split_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


results = []

results.append(evaluate_split(train, "train"))
results.append(evaluate_split(validation, "validation"))
results.append(evaluate_split(test, "test"))

summary = pd.DataFrame(results)

summary_path = os.path.join(
    OUTPUT_DIR,
    "receiver_ewma_evaluation.csv"
)

summary.to_csv(summary_path, index=False)

print("\n========== FINAL EWMA COMPARISON ==========")
print(summary.to_string(index=False))

print("\nSaved:", summary_path)
print("\nREAL-DATA EWMA MONITORING COMPLETED.")
