
import json
import glob
import csv
import os
import math
from collections import defaultdict, deque

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"

SPLITS = {
    "Train": os.path.join(BASE, "Expanded_Train"),
    "Validation": os.path.join(BASE, "Expanded_Validation"),
    "Test": os.path.join(BASE, "Expanded_Test")
}

OUTPUT = r"C:\Users\ADMIN\Downloads\VANET_DDoS_project\dataset\real"
os.makedirs(OUTPUT, exist_ok=True)

WINDOW_NS = 1_000_000_000


def extract_split(folder, split_name):

    print("\nProcessing:", split_name)

    files = glob.glob(
        os.path.join(folder, "**", "*.json"),
        recursive=True
    )

    # Store all receiver copies under one message identity
    unique_messages = defaultdict(list)

    total_records = 0

    for file in files:

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for msg in data:

            total_records += 1

            key = (
                str(msg["sender_id"]),
                str(msg["messageID"]),
                str(msg["sendTime"])
            )

            unique_messages[key].append(msg)

    print("Original records:", total_records)
    print("Unique messages:", len(unique_messages))
    print("Duplicate copies removed:",
          total_records - len(unique_messages))

    # Check consistency between receiver copies
    inconsistent_sender = 0
    inconsistent_labels = 0

    messages_by_sender = defaultdict(list)

    for key, copies in unique_messages.items():

        reference = copies[0]

        reference_sender = reference["sender"]
        reference_label = int(reference["attacker"])

        for copy in copies[1:]:

            if copy["sender"] != reference_sender:
                inconsistent_sender += 1

            if int(copy["attacker"]) != reference_label:
                inconsistent_labels += 1

        # Average delay across receiver copies
        delays = []

        for copy in copies:

            delay = (
                int(copy["rcvTime"]) - int(copy["sendTime"])
            ) / 1_000_000

            delays.append(delay)

        average_delay = sum(delays) / len(delays)

        record = {
            "sender_id": key[0],
            "messageID": key[1],
            "sendTime": int(reference["sendTime"]),
            "sender": reference["sender"],
            "attacker": reference_label,
            "message_delay_ms": average_delay
        }

        messages_by_sender[key[0]].append(record)

    print("Inconsistent sender records:", inconsistent_sender)
    print("Inconsistent labels:", inconsistent_labels)

    if inconsistent_sender > 0 or inconsistent_labels > 0:
        raise ValueError(
            "Duplicate copies contain inconsistent data. "
            "Inspect before building the dataset."
        )

    rows = []

    # Calculate message frequency after deduplication
    for sender, sender_messages in messages_by_sender.items():

        sender_messages.sort(key=lambda m: m["sendTime"])

        recent_times = deque()

        for msg in sender_messages:

            s = msg["sender"]
            send_time = msg["sendTime"]

            while recent_times and (
                send_time - recent_times[0] > WINDOW_NS
            ):
                recent_times.popleft()

            recent_times.append(send_time)

            noise = s["pos_noise"]
            x, y, z = map(float, noise.split(","))

            position_noise = math.sqrt(
                x*x + y*y + z*z
            )

            heading = float(s["hed"])
            heading_radians = math.radians(heading)

            row = {
                "sender_id": sender,
                "messageID": msg["messageID"],
                "sendTime": send_time,
                "vehicle_speed": float(s["spd"]),
                "acceleration": float(s["acl"]),
                "heading_sin": math.sin(heading_radians),
                "heading_cos": math.cos(heading_radians),
                "heading_noise": float(s["hed_noise"]),
                "position_noise": position_noise,
                "message_delay_ms": msg["message_delay_ms"],
                "messages_last_1s": len(recent_times),
                "driver_profile": s.get("driversProfile", ""),
                "distance_to_road_edge": s.get(
                    "distance_to_road_edge", ""
                ),
                "attacker": msg["attacker"]
            }

            rows.append(row)

    columns = [
        "sender_id",
        "messageID",
        "sendTime",
        "vehicle_speed",
        "acceleration",
        "heading_sin",
        "heading_cos",
        "heading_noise",
        "position_noise",
        "message_delay_ms",
        "messages_last_1s",
        "driver_profile",
        "distance_to_road_edge",
        "attacker"
    ]

    # Save separately; preserve original CSV files
    output_file = os.path.join(
        OUTPUT,
        f"real_{split_name.lower()}_deduplicated.csv"
    )

    with open(output_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    benign = sum(1 for r in rows if r["attacker"] == 0)
    attacker = sum(1 for r in rows if r["attacker"] == 1)

    print("Final records:", len(rows))
    print("Benign:", benign)
    print("Attacker:", attacker)
    print("Saved:", output_file)


for split_name, folder in SPLITS.items():
    extract_split(folder, split_name)

print("\nDEDUPLICATED REAL DATASET CREATION COMPLETED")
