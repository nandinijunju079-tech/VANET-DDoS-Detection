import json
import glob
import os

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"
FOLDER = os.path.join(BASE, "Expanded_Train")

target = (
    "veh_1451",
    "329",
    "3765125163170"
)

files = glob.glob(
    os.path.join(FOLDER, "**", "*.json"),
    recursive=True
)

print("TARGET MESSAGE:", target)

for file in files:

    with open(file, "r", encoding="utf-8") as f:
        messages = json.load(f)

    for msg in messages:

        key = (
            str(msg.get("sender_id")),
            str(msg.get("messageID")),
            str(msg.get("sendTime"))
        )

        if key == target:

            print("\nFile:", os.path.basename(file))
            print("Sender ID:", msg.get("sender_id"))
            print("Sender alias:", msg.get("sender_alias"))
            print("Receiver data:", msg.get("receiver"))
            print("Sender data:", msg.get("sender"))
            print("Attack label:", msg.get("attacker"))