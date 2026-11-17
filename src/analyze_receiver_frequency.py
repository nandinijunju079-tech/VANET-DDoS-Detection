
import json
import glob
import os
from collections import defaultdict, deque

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"

SPLITS = {
    "TRAIN": os.path.join(BASE, "Expanded_Train", "highway_2_dosAttack"),
    "VALIDATION": os.path.join(BASE, "Expanded_Validation", "highway_2_dosAttack"),
    "TEST": os.path.join(BASE, "Expanded_Test", "InTAS_highway_2_dosAttack")
}

WINDOW_NS = 1_000_000_000


def analyze_split(name, folder):

    files = glob.glob(os.path.join(folder, "*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files found in: {folder}")

    benign_counts = []
    attacker_counts = []

    total_records = 0
    unique_records = 0

    for file_path in files:

        with open(file_path, "r", encoding="utf-8") as f:
            messages = json.load(f)

        total_records += len(messages)

        unique_messages = {}

        for msg in messages:

            identity = (
                msg["sender_id"],
                str(msg["messageID"]),
                str(msg["sendTime"])
            )

            unique_messages[identity] = msg

        unique_records += len(unique_messages)

        by_sender = defaultdict(list)

        for msg in unique_messages.values():
            by_sender[msg["sender_id"]].append(msg)

        for sender_id, records in by_sender.items():

            records.sort(key=lambda m: int(m["rcvTime"]))

            recent_times = deque()

            for msg in records:

                receive_time = int(msg["rcvTime"])

                while recent_times and (
                    receive_time - recent_times[0] >= WINDOW_NS
                ):
                    recent_times.popleft()

                recent_times.append(receive_time)

                frequency = len(recent_times)
                label = int(msg["attacker"])

                if label == 0:
                    benign_counts.append(frequency)
                elif label == 1:
                    attacker_counts.append(frequency)

    print("\n" + "=" * 55)
    print(name, "RECEIVER-SIDE FREQUENCY")
    print("=" * 55)

    print("Receiver files:", len(files))
    print("Original records:", total_records)
    print("Unique receiver-message records:", unique_records)

    for label, values in [
        ("BENIGN", benign_counts),
        ("ATTACKER", attacker_counts)
    ]:

        print(f"\n{label}")
        print("Messages:", len(values))

        if not values:
            continue

        print("Mean:", round(sum(values) / len(values), 3))
        print("Median:", sorted(values)[len(values) // 2])
        print("Minimum:", min(values))
        print("Maximum:", max(values))

        for threshold in [2, 3, 4, 5]:

            count = sum(v >= threshold for v in values)

            print(
                f"Frequency >= {threshold}: "
                f"{count} ({count / len(values) * 100:.2f}%)"
            )


for split_name, folder in SPLITS.items():
    analyze_split(split_name, folder)
