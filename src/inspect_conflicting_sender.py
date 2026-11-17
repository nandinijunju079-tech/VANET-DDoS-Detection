
import os
import json
from collections import defaultdict

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset"

FOLDERS = {
    "Train": os.path.join(
        BASE, "Extracted_DOS", "Expanded_Train"
    ),
    "Validation": os.path.join(
        BASE, "Extracted_DOS", "Expanded_Validation"
    ),
    "Test": os.path.join(
        BASE, "Extracted_DOS", "Expanded_Test"
    )
}

TARGET = "veh_2404"

print("\n========== CONFLICTING SENDER INSPECTION ==========")
print("Target sender:", TARGET)

for split, folder in FOLDERS.items():

    print(f"\n========== {split} ==========")

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    found = 0

    for root, _, files in os.walk(folder):
        for filename in files:

            if not filename.endswith(".json"):
                continue

            path = os.path.join(root, filename)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                for msg in data:

                    if not isinstance(msg, dict):
                        continue

                    sender_id = str(msg.get("sender_id", ""))

                    if sender_id != TARGET:
                        continue

                    found += 1

                    print("\nReceiver file:", filename)
                    print("Sender:", sender_id)
                    print("Message ID:", msg.get("messageID"))
                    print("Send time:", msg.get("sendTime"))
                    print("Attacker label:", msg.get("attacker"))
                    print("Sender data:", msg.get("sender"))

            except Exception as e:
                print("Error reading:", filename, e)

    print("\nTotal matching messages:", found)

print("\nINSPECTION COMPLETED")
