
import os
import json
import glob
import math
import pandas as pd
from collections import defaultdict, deque

BASE = r"C:\Users\ADMIN\Desktop\VANET_Real_Dataset\Extracted_DOS"

OUTPUT = "dataset/real"

SPLITS = {
    "train": os.path.join(BASE, "Expanded_Train", "highway_2_dosAttack"),
    "validation": os.path.join(BASE, "Expanded_Validation", "highway_2_dosAttack"),
    "test": os.path.join(BASE, "Expanded_Test", "InTAS_highway_2_dosAttack")
}

WINDOW_NS = 1_000_000_000


def vector_magnitude(value):
    if not value:
        return float("nan")
    parts = [float(x) for x in value.split(",")]
    return math.sqrt(sum(x * x for x in parts))


def build_split(split_name, folder):
    files = glob.glob(os.path.join(folder, "*.json"))

    if not files:
        raise FileNotFoundError(folder)

    rows = []
    total_original = 0
    total_unique = 0

    for file_path in files:
        receiver_id = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, "r", encoding="utf-8") as f:
            messages = json.load(f)

        total_original += len(messages)
        unique_messages = {}

        for msg in messages:
            identity = (
                msg["sender_id"],
                str(msg["messageID"]),
                str(msg["sendTime"])
            )

            if identity in unique_messages:
                old = unique_messages[identity]
                if int(old["attacker"]) != int(msg["attacker"]):
                    raise ValueError(f"Conflicting labels: {identity}")
            else:
                unique_messages[identity] = msg

        total_unique += len(unique_messages)

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
                sender = msg["sender"]
                heading = math.radians(float(sender["hed"]))

                row = {
                    "receiver_id": receiver_id,
                    "sender_id": sender_id,
                    "messageID": msg["messageID"],
                    "sendTime": int(msg["sendTime"]),
                    "rcvTime": receive_time,
                    "vehicle_speed": float(sender["spd"]),
                    "acceleration": float(sender["acl"]),
                    "heading_sin": math.sin(heading),
                    "heading_cos": math.cos(heading),
                    "heading_noise": float(sender["hed_noise"]),
                    "position_noise": vector_magnitude(sender["pos_noise"]),
                    "message_delay_ms": (
                        receive_time - int(msg["sendTime"])
                    ) / 1_000_000,
                    "messages_received_last_1s": len(recent_times),
                    "driver_profile": sender["driversProfile"],
                    "distance_to_road_edge": float(
                        sender["distance_to_road_edge"]
                    ),
                    "attacker": int(msg["attacker"])
                }

                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT, exist_ok=True)

    output_path = os.path.join(
        OUTPUT,
        f"real_receiver_{split_name}_with_time.csv"
    )

    df.to_csv(output_path, index=False)

    print("\n", "=" * 45)
    print(split_name.upper())
    print("=" * 45)
    print("Receiver files:", len(files))
    print("Original records:", total_original)
    print("Unique receiver-message records:", total_unique)
    print("Output rows:", len(df))
    print("rcvTime column included:", "rcvTime" in df.columns)
    print("Saved:", output_path)


for split_name, folder in SPLITS.items():
    build_split(split_name, folder)

print("\nAll receiver-level datasets created successfully.")
