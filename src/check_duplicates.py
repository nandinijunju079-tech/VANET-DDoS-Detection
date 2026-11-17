
import pandas as pd
import os

folder = r"C:\Users\ADMIN\Downloads\VANET_DDoS_project\dataset\real"

files = [
    "real_train.csv",
    "real_validation.csv",
    "real_test.csv"
]

for filename in files:
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)

    features = [col for col in df.columns if col != "attacker"]

    grouped = df.groupby(features, dropna=False)["attacker"].nunique()

    conflicting = (grouped > 1).sum()
    repeated = (grouped > 1).sum()

    print("\nDataset:", filename)
    print("Total records:", len(df))
    print("Feature combinations with conflicting labels:", conflicting)

print("\nDUPLICATE LABEL CHECK COMPLETED")
