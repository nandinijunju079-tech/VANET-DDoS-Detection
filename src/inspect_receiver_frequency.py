
import json
import glob
import os
from collections import defaultdict
import statistics

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS\Expanded_Test\InTAS_highway_2_dosAttack"

files = glob.glob(os.path.join(BASE, "*.json"))

if not files:
    raise FileNotFoundError("No JSON files found.")

file_path = files[0]
receiver_id = os.path.splitext(os.path.basename(file_path))[0]

with open(file_path, "r", encoding="utf-8") as f:
    messages = json.load(f)

print("Receiver:", receiver_id)
print("Total received records:", len(messages))

# Group messages by sender
sender_messages = defaultdict(dict)

for msg in messages:
    sender_id = msg["sender_id"]

    # Remove repeated copies of the same message
    identity = (
        sender_id,
        str(msg["messageID"]),
        str(msg["sendTime"])
    )

    sender_messages[sender_id][identity] = msg

# Select the sender with the most unique messages
sender_id = max(sender_messages, key=lambda s: len(sender_messages[s]))

records = list(sender_messages[sender_id].values())
records.sort(key=lambda m: int(m["rcvTime"]))

print("\nSelected sender:", sender_id)
print("Unique messages received:", len(records))

if len(records) > 1:
    times = [int(m["rcvTime"]) for m in records]
    gaps = [
        (times[i] - times[i - 1]) / 1_000_000
        for i in range(1, len(times))
    ]

    print("\nReception time statistics (milliseconds):")
    print("Minimum gap:", min(gaps))
    print("Median gap:", statistics.median(gaps))
    print("Maximum gap:", max(gaps))

    print("\nFirst 10 reception gaps:")
    for gap in gaps[:10]:
        print(round(gap, 3), "ms")

print("\nTimestamp sample:")
for msg in records[:5]:
    print(
        "sendTime:", msg["sendTime"],
        "| rcvTime:", msg["rcvTime"],
        "| messageID:", msg["messageID"],
        "| attacker:", msg["attacker"]
    )
