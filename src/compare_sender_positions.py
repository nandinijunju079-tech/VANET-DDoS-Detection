import pandas as pd

BASE = "dataset/real"

train = pd.read_csv(
    f"{BASE}/real_train_deduplicated.csv"
)

test = pd.read_csv(
    f"{BASE}/real_test_deduplicated.csv"
)

target = "veh_2404"

for name, df in [("TRAIN", train), ("TEST", test)]:
    rows = df[df["sender_id"] == target]

    print(f"\n========== {name} ==========")
    print("Messages:", len(rows))

    if len(rows) == 0:
        continue

    print("\nVehicle speed:")
    print(rows["vehicle_speed"].describe())

    print("\nHeading:")
    print(rows["heading_sin"].describe())
    print(rows["heading_cos"].describe())

    print("\nPosition-related features:")
    print("Position noise:")
    print(rows["position_noise"].describe())

    print("\nDriver profiles:")
    print(rows["driver_profile"].value_counts())

print("\nPOSITION COMPARISON COMPLETED")