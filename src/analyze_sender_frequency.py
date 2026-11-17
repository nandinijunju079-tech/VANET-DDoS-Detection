
import pandas as pd

files = {
    "TRAIN": "dataset/real/real_train_deduplicated.csv",
    "VALIDATION": "dataset/real/real_validation_deduplicated.csv",
    "TEST": "dataset/real/real_test_deduplicated.csv"
}

for split, path in files.items():

    df = pd.read_csv(path)

    print(f"\n========== {split} SENDER ANALYSIS ==========")

    sender_summary = df.groupby("sender_id").agg(
        messages=("attacker", "size"),
        attacker_label=("attacker", "first"),
        frequency_mean=("messages_last_1s", "mean"),
        frequency_min=("messages_last_1s", "min"),
        frequency_max=("messages_last_1s", "max"),
        label_count=("attacker", "nunique")
    ).reset_index()

    print("\nSender count by label:")

    print(
        sender_summary.groupby("attacker_label")["sender_id"]
        .count()
        .rename(index={0: "Benign", 1: "Attacker"})
    )

    print("\nFrequency summary by sender label:")

    print(
        sender_summary.groupby("attacker_label")[
            ["messages", "frequency_mean", "frequency_min", "frequency_max"]
        ].agg(["mean", "median", "min", "max"]).round(3)
    )

    print("\nSenders with mixed labels:", 
          (sender_summary["label_count"] > 1).sum())

print("\nSender frequency analysis completed.")
