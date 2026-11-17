import json
import glob
from collections import defaultdict

folder = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS\Expanded_Train"

files = glob.glob(folder + r"\**\*.json", recursive=True)

sender_data = defaultdict(lambda: {
    "times": [],
    "label": None
})

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        messages = json.load(f)

    for msg in messages:
        sender = msg["sender_id"]

        sender_data[sender]["times"].append(msg["sendTime"])
        sender_data[sender]["label"] = str(msg["attacker"])

rates = {"0": [], "1": []}

for sender, data in sender_data.items():
    times = data["times"]

    duration = (max(times) - min(times)) / 1_000_000_000

    if duration > 0:
        rate = len(times) / duration
        rates[data["label"]].append(rate)

for label, name in [("0", "Benign"), ("1", "Attacker")]:
    values = rates[label]

    print("\n", name)
    print("Unique senders:", len(values))
    print("Average messages per second:", sum(values) / len(values))
    print("Minimum rate:", min(values))
    print("Maximum rate:", max(values))