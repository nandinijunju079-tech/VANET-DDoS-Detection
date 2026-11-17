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

    locations = defaultdict(set)

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        for msg in messages:
            key = (
                str(msg.get("sender_id")),
                str(msg.get("messageID")),
                str(msg.get("sendTime"))
            )

            locations[key].add(os.path.basename(file))

    print(f"\n========== {split} ==========")

    shown = 0

    for key, filenames in locations.items():

        if len(filenames) > 1:
            print("\nMessage identity:", key)
            print("Found in files:", sorted(filenames))

            shown += 1

            if shown == 5:
                break

    print("\nRepeated identity examples displayed:", shown)

print("\nREPEATED MESSAGE FILE INSPECTION COMPLETED")