
import pandas as pd
import os

BASE = "dataset/real"

files = {
    "Train": "real_train_deduplicated.csv",
    "Validation": "real_validation_deduplicated.csv",
    "Test": "real_test_deduplicated.csv"
}

for split, filename in files.items():

    path = os.path.join(BASE, filename)
    df = pd.read_csv(path)

    print("\n==========", split, "==========")

    print("Total records:", len(df))

    print("Columns:", list(df.columns))

    print(
        "Unique message identities:",
        df[["sender_id", "messageID", "sendTime"]]
        .drop_duplicates().shape[0]
    )

    print(
        "Duplicate messages:",
        df.duplicated(
            subset=["sender_id", "messageID", "sendTime"]
        ).sum()
    )

    print("\nClass distribution:")
    print(df["attacker"].value_counts().sort_index())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMessage frequency statistics:")
    print(df["messages_last_1s"].describe().round(2))

print("\nVALIDATION COMPLETED")
