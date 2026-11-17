
import csv
import os

folder = r"C:\Users\ADMIN\Downloads\VANET_DDoS_project\dataset\real"

files = [
    "real_train.csv",
    "real_validation.csv",
    "real_test.csv"
]

for filename in files:
    path = os.path.join(folder, filename)

    print("\nChecking:", filename)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("Total records:", len(rows))
    print("Columns:", reader.fieldnames)

    missing = 0

    for row in rows:
        for value in row.values():
            if value is None or value.strip() == "":
                missing += 1

    print("Empty cells:", missing)

    labels = {}
    for row in rows:
        label = row["attacker"]
        labels[label] = labels.get(label, 0) + 1

    print("Class distribution:", labels)

print("\nDATASET VALIDATION COMPLETED")
