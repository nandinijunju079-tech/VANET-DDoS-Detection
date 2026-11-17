
import pandas as pd

files = {
    "TRAIN": "dataset/real/real_train_deduplicated.csv",
    "VALIDATION": "dataset/real/real_validation_deduplicated.csv",
    "TEST": "dataset/real/real_test_deduplicated.csv"
}

for split, path in files.items():

    df = pd.read_csv(path)

    print(f"\n========== {split} DATASET ==========")

    print("\nMessage frequency by actual label:")

    summary = df.groupby("attacker")["messages_last_1s"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    summary.index = ["Benign (0)", "Attacker (1)"]

    print(summary.round(3))

    print("\nFrequency value counts by label:")

    frequency_table = pd.crosstab(
        df["messages_last_1s"],
        df["attacker"]
    )

    frequency_table.columns = ["Benign", "Attacker"]

    print(frequency_table.to_string())

print("\nAnalysis completed.")
