
import os
import pandas as pd

DATA = "dataset/real"

files = {
    "Train": "real_train_deduplicated.csv",
    "Validation": "real_validation_deduplicated.csv",
    "Test": "real_test_deduplicated.csv"
}

sender_labels = {}

for split, filename in files.items():
    df = pd.read_csv(os.path.join(DATA, filename))

    for _, row in df[["sender_id", "attacker"]].drop_duplicates().iterrows():
        sender = str(row["sender_id"])
        label = int(row["attacker"])

        sender_labels.setdefault(sender, {})
        sender_labels[sender][split] = label

print("\n========== CROSS-SPLIT SENDER LABEL ANALYSIS ==========")

overlapping = 0
consistent = 0
inconsistent = 0

for sender, labels in sender_labels.items():
    if len(labels) > 1:
        overlapping += 1

        if len(set(labels.values())) == 1:
            consistent += 1
        else:
            inconsistent += 1
            print(
                "Different labels:",
                sender,
                labels
            )

print("\nTotal unique senders:", len(sender_labels))
print("Senders appearing in multiple splits:", overlapping)
print("Overlapping senders with consistent labels:", consistent)
print("Overlapping senders with different labels:", inconsistent)

print("\nANALYSIS COMPLETED")
