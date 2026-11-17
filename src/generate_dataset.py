
import pandas as pd
import numpy as np
import os

np.random.seed(42)

records = []

for i in range(1000):

    vehicle_speed = np.random.randint(20, 120)
    vehicle_density = np.random.randint(5, 100)
    neighbor_count = np.random.randint(1, 30)
    packet_size = np.random.randint(100, 1500)
    flow_duration = np.random.uniform(1, 60)
    emergency_priority = np.random.choice(
    [0, 1],
    p=[0.85, 0.15]
)
    protocol = np.random.choice([0, 1, 2])

    if np.random.random() < 0.75:
        packet_rate = np.random.randint(10, 150)
        label = "Normal"
    else:
        packet_rate = np.random.randint(300, 1500)
        label = "DDoS"

    records.append([
        vehicle_speed,
        vehicle_density,
        neighbor_count,
        packet_rate,
        packet_size,
        flow_duration,
        protocol,
        emergency_priority,
        label
    ])

columns = [
    "vehicle_speed",
    "vehicle_density",
    "neighbor_count",
    "packet_rate",
    "packet_size",
    "flow_duration",
    "protocol",
    "emergency_priority",
    "label"
]

df = pd.DataFrame(records, columns=columns)

os.makedirs("dataset", exist_ok=True)

df.to_csv("dataset/vanet_dataset.csv", index=False)

print("VANET Dataset Generated Successfully!")
print("\nFirst 10 Records:")
print(df.head(10))

print("\nClass Distribution:")
print(df["label"].value_counts())
