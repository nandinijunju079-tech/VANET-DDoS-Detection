
import joblib
import pandas as pd

# Load trained Random Forest model
model = joblib.load("models/random_forest_model.joblib")

# Define the traffic records to test
traffic_records = [
    {
        "vehicle_speed": 60,
        "vehicle_density": 30,
        "neighbor_count": 10,
        "packet_rate": 80,
        "packet_size": 500,
        "flow_duration": 20,
        "protocol": 0,
        "emergency_priority": 1
    },
    {
        "vehicle_speed": 70,
        "vehicle_density": 40,
        "neighbor_count": 15,
        "packet_rate": 220,
        "packet_size": 800,
        "flow_duration": 15,
        "protocol": 1,
        "emergency_priority": 0
    },
    {
        "vehicle_speed": 90,
        "vehicle_density": 80,
        "neighbor_count": 25,
        "packet_rate": 900,
        "packet_size": 1200,
        "flow_duration": 10,
        "protocol": 2,
        "emergency_priority": 0
    }
]

# Convert records into a DataFrame
traffic_df = pd.DataFrame(traffic_records)

# Get prediction probabilities
probabilities = model.predict_proba(traffic_df)

# Find the DDoS class index
ddos_index = list(model.classes_).index(1)

print("\n----- VANET ATTACK DETECTION -----")

for i, probability in enumerate(probabilities):

    ddos_probability = probability[ddos_index]

    confidence = max(probability) * 100

    if ddos_probability < 0.30:
        result = "Normal"

    elif ddos_probability <= 0.70:
        result = "Suspicious"

    else:
        result = "DDoS"

    print(f"\nTraffic Record {i + 1}")
    print("DDoS Probability:", round(ddos_probability * 100, 2), "%")
    print("Prediction Confidence:", round(confidence, 2), "%")
    print("Detection Result:", result)

print("\nDetection Completed Successfully!")
