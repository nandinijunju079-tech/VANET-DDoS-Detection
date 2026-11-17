

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Create output folder
os.makedirs("outputs", exist_ok=True)

# 2. Load dataset
data = pd.read_csv("dataset/vanet_dataset.csv")

packet_rate = data["packet_rate"]

# 3. Calculate EWMA
alpha = 0.3

data["EWMA"] = packet_rate.ewm(
    alpha=alpha,
    adjust=False
).mean()

# 4. Calculate threshold using baseline traffic
baseline_size = 100

baseline = data["EWMA"].iloc[:baseline_size]

baseline_mean = baseline.mean()
baseline_std = baseline.std()

threshold = baseline_mean + (2 * baseline_std)

# -----------------------------------------
# 5. GET CORRECTLY ALIGNED EWMA INFORMATION
# -----------------------------------------

if len(ewma_data) <= X_test.index.max():
    raise ValueError("EWMA data does not contain all required test indexes.")

ewma_anomaly = (
    ewma_data.iloc[X_test.index]["Anomaly"]
    .reset_index(drop=True)
)

# 6. Display results
print("\n----- IMPROVED EWMA MONITORING -----")

print("Baseline Mean:", round(baseline_mean, 2))
print("Baseline Standard Deviation:", round(baseline_std, 2))
print("EWMA Threshold:", round(threshold, 2))

print("\nTraffic Monitoring Results:")

print(
    data[["packet_rate", "EWMA", "Anomaly"]]
    .head(20)
    .to_string(index=False)
)

print("\nAnomaly Distribution:")
print(data["Anomaly"].value_counts())

# 7. Save results
data.to_csv(
    "outputs/ewma_monitoring_results.csv",
    index=False
)

# 8. Generate graph
plt.figure(figsize=(12, 5))

plt.plot(
    data["packet_rate"].values,
    label="Packet Rate",
    alpha=0.5
)

plt.plot(
    data["EWMA"].values,
    label="EWMA"
)

plt.axhline(
    y=threshold,
    linestyle="--",
    label="Anomaly Threshold"
)

plt.xlabel("Traffic Record")
plt.ylabel("Packet Rate")
plt.title("Improved EWMA VANET Anomaly Monitoring")

plt.legend()
plt.tight_layout()

plt.savefig("outputs/ewma_monitoring.png")
plt.close()

print("\nImproved EWMA Monitoring Completed Successfully!")
