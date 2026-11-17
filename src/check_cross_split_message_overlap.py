import json
import glob
import os
from collections import defaultdict

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"

folders = {
    "Train": os.path.join(BASE, "Expanded_Train"),
    "Validation": os.path.join(BASE, "Expanded_Validation"),
    "Test": os.path.join(BASE, "Expanded_Test")
}

split_keys = {}
split_labels = {}

for split, folder in folders.items():

    files = glob.glob(
        os.path.join(folder, "**", "*.json"),
        recursive=True
    )

    keys = set()
    labels = defaultdict(set)

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        for msg in messages:
            key = (
                str(msg.get("sender_id")),
                str(msg.get("messageID")),
                str(msg.get("sendTime"))
            )

            keys.add(key)
            labels[key].add(int(msg["attacker"]))

    split_keys[split] = keys
    split_labels[split] = labels

    print(f"\n{split}: {len(keys)} unique identities")

pairs = [
    ("Train", "Validation"),
    ("Train", "Test"),
    ("Validation", "Test")
]

print("\n========== CROSS-SPLIT OVERLAP ==========")

for first, second in pairs:

    overlap = split_keys[first] & split_keys[second]

    print(f"\n{first} - {second}")
    print("Shared message identities:", len(overlap))

    conflicts = 0

    for key in overlap:
        combined = (
            split_labels[first][key]
            | split_labels[second][key]
        )

        if len(combined) > 1:
            conflicts += 1

    print("Conflicting labels:", conflicts)

common = (
    split_keys["Train"]
    & split_keys["Validation"]
    & split_keys["Test"]
)

print("\nIdentities present in all three splits:", len(common))

print("\nCROSS-SPLIT CHECK COMPLETED")