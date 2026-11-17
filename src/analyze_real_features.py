
import csv
import os
import math
from collections import Counter, defaultdict
from statistics import mean, median

folder = r"C:\Users\ADMIN\Downloads\VANET_DDoS_project\dataset\real"

files = [
    "real_train.csv",
    "real_validation.csv",
    "real_test.csv"
]

numeric_features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "messages_last_1s",
    "distance_to_road_edge"
]

for filename in files:

    path = os.path.join(folder, filename)

    print("\n" + "=" * 55)
    print("FILE:", filename)
    print("=" * 55)

    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = defaultdict(lambda: defaultdict(list))
    profiles = defaultdict(Counter)

    invalid_values = 0

    for row in rows:
        label = row["attacker"]

        profiles[label][row["driver_profile"]] += 1

        for feature in numeric_features:
            try:
                value = float(row[feature])

                if not math.isfinite(value):
                    invalid_values += 1
                    continue

                grouped[label][feature].append(value)

            except (ValueError, TypeError):
                invalid_values += 1

    for label, name in [("0", "BENIGN"), ("1", "ATTACKER")]:

        print("\nCLASS:", name)
        print("Driver profiles:", dict(profiles[label]))

        for feature in numeric_features:
            values = grouped[label][feature]

            if values:
                print("\n", feature)
                print("Count:", len(values))
                print("Mean:", round(mean(values), 4))
                print("Median:", round(median(values), 4))
                print("Minimum:", round(min(values), 4))
                print("Maximum:", round(max(values), 4))

    print("\nInvalid numeric values:", invalid_values)

print("\nFEATURE ANALYSIS COMPLETED")
