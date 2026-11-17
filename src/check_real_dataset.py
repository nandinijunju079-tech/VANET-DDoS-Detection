
import json
import glob
from collections import Counter

folder = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS\Expanded_test"
files = glob.glob(folder + r"\**\*.json", recursive=True)

counts = Counter()
total_messages = 0
missing_labels = 0

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for message in data:
        total_messages += 1

        if "attacker" in message:
            counts[str(message["attacker"])] += 1
        else:
            missing_labels += 1

print("JSON files:", len(files))
print("Total messages:", total_messages)
print("Label counts:", counts)
print("Messages without attacker label:", missing_labels)
