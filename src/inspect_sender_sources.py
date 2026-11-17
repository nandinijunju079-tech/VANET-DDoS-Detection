
import json
import glob
import os
from collections import defaultdict

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"

SPLITS = {
    "Train": os.path.join(BASE, "Expanded_Train"),
    "Validation": os.path.join(BASE, "Expanded_Validation"),
    "Test": os.path.join(BASE, "Expanded_Test")
}

sender_sources = {
    split: defaultdict(set) for split in SPLITS
}

sender_labels = {
    split: defaultdict(set) for split in SPLITS
}

for split, folder in SPLITS.items():

    files = glob.glob(
        os.path.join(folder, "**", "*.json"),
        recursive=True
    )

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        for msg in messages:
            sender = str(msg["sender_id"])

            sender_sources[split][sender].add(
                os.path.relpath(file, folder)
            )

            sender_labels[split][sender].add(
                int(msg["attacker"])
            )

train_ids = set(sender_sources["Train"])
val_ids = set(sender_sources["Validation"])
test_ids = set(sender_sources["Test"])

overlaps = {
    "Train-Test": train_ids & test_ids,
    "Train-Validation": train_ids & val_ids,
    "Validation-Test": val_ids & test_ids
}

for name, ids in overlaps.items():

    print("\n", name)
    print("Overlapping senders:", len(ids))

    for sender in sorted(ids)[:10]:
        print("\nSender:", sender)

        for split in SPLITS:
            if sender in sender_sources[split]:
                print(split, "files:",
                      sorted(sender_sources[split][sender]))
                print(split, "labels:",
                      sorted(sender_labels[split][sender]))

print("\nSOURCE INSPECTION COMPLETED")
