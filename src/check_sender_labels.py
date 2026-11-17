
import json
import glob
from collections import defaultdict

base = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"

splits = {
    "Train": base + r"\Expanded_Train",
    "Validation": base + r"\Expanded_Validation",
    "Test": base + r"\Expanded_Test"
}

for split, folder in splits.items():

    files = glob.glob(folder + r"\**\*.json", recursive=True)

    sender_labels = defaultdict(set)

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            messages = json.load(f)

        for msg in messages:
            sender = msg["sender_id"]
            label = int(msg["attacker"])

            sender_labels[sender].add(label)

    mixed = {
        sender: labels
        for sender, labels in sender_labels.items()
        if len(labels) > 1
    }

    print("\n", split)
    print("Unique senders:", len(sender_labels))
    print("Consistent-label senders:", len(sender_labels) - len(mixed))
    print("Mixed-label senders:", len(mixed))

    if mixed:
        print("Examples:", list(mixed.items())[:10])

print("\nSENDER LABEL CHECK COMPLETED")
