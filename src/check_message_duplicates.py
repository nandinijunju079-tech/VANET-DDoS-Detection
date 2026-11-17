
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

for split, folder in folders.items():

    files = glob.glob(
        os.path.join(folder, "**", "*.json"),
        recursive=True
    )

    seen = defaultdict(set)
    labels = defaultdict(set)
    total = 0

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        for msg in messages:
            total += 1

            key = (
                str(msg.get("sender_id")),
                str(msg.get("messageID")),
                str(msg.get("sendTime"))
            )

            seen[key].add(os.path.basename(file))
            labels[key].add(int(msg["attacker"]))

    repeated = sum(1 for files_set in seen.values()
                   if len(files_set) > 1)

    conflicts = sum(1 for label_set in labels.values()
                    if len(label_set) > 1)

    print("\nDataset:", split)
    print("Total messages:", total)
    print("Unique message identities:", len(seen))
    print("Repeated identities across files:", repeated)
    print("Identities with conflicting labels:", conflicts)

print("\nMESSAGE DUPLICATE ANALYSIS COMPLETED")
