
import pandas as pd

BASE = "dataset/real"

FILES = {
    "TRAIN": f"{BASE}/real_receiver_train.csv",
    "VALIDATION": f"{BASE}/real_receiver_validation.csv",
    "TEST": f"{BASE}/real_receiver_test.csv"
}

identity_columns = [
    "receiver_id",
    "sender_id",
    "messageID",
    "sendTime"
]

datasets = {
    name: pd.read_csv(path, dtype={
        "receiver_id": str,
        "sender_id": str,
        "messageID": str
    })
    for name, path in FILES.items()
}

identities = {
    name: set(map(
        tuple,
        df[identity_columns].itertuples(index=False, name=None)
    ))
    for name, df in datasets.items()
}

for name, values in identities.items():
    print(f"{name} unique reception identities:", len(values))

comparisons = [
    ("TRAIN", "VALIDATION"),
    ("TRAIN", "TEST"),
    ("VALIDATION", "TEST")
]

print("\n===== CROSS-SPLIT RECEPTION IDENTITY OVERLAP =====")

for first, second in comparisons:
    shared = identities[first] & identities[second]

    print(f"\n{first} vs {second}")
    print("Exact shared reception identities:", len(shared))

    if shared:
        print("Examples:", list(shared)[:5])
