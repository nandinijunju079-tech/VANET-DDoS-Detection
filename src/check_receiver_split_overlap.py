
import pandas as pd

BASE = "dataset/real"

train = pd.read_csv(f"{BASE}/real_receiver_train.csv")
validation = pd.read_csv(f"{BASE}/real_receiver_validation.csv")
test = pd.read_csv(f"{BASE}/real_receiver_test.csv")

datasets = {
    "TRAIN": train,
    "VALIDATION": validation,
    "TEST": test
}

for name, df in datasets.items():
    print(f"\n{name}")
    print("Unique receivers:", df["receiver_id"].nunique())
    print("Unique senders:", df["sender_id"].nunique())
    print(
        "Unique receiver-sender pairs:",
        df[["receiver_id", "sender_id"]].drop_duplicates().shape[0]
    )

def compare(name1, df1, name2, df2):

    receivers1 = set(df1["receiver_id"])
    receivers2 = set(df2["receiver_id"])

    senders1 = set(df1["sender_id"])
    senders2 = set(df2["sender_id"])

    pairs1 = set(
        zip(df1["receiver_id"], df1["sender_id"])
    )

    pairs2 = set(
        zip(df2["receiver_id"], df2["sender_id"])
    )

    print(f"\n{name1} vs {name2}")
    print("Overlapping receiver IDs:", len(receivers1 & receivers2))
    print("Overlapping sender IDs:", len(senders1 & senders2))
    print("Overlapping receiver-sender pairs:", len(pairs1 & pairs2))

compare("TRAIN", train, "VALIDATION", validation)
compare("TRAIN", train, "TEST", test)
compare("VALIDATION", validation, "TEST", test)
