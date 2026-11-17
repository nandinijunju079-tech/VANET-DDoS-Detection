
import json
import glob
import os

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

    print("\n==========", split, "==========")

    if not files:
        print("No JSON files found")
        continue

    file = files[0]

    with open(file, "r", encoding="utf-8") as f:
        messages = json.load(f)

    print("Sample file:", os.path.basename(file))
    print("Messages in file:", len(messages))

    if messages:
        msg = messages[0]

        print("Message fields:", list(msg.keys()))
        print("Sender ID:", msg.get("sender_id"))
        print("Message ID:", msg.get("messageID"))
        print("Receiver:", msg.get("receiver"))
        print("Send time:", msg.get("sendTime"))
        print("Attack label:", msg.get("attacker"))

print("\nMESSAGE STRUCTURE INSPECTION COMPLETED")
