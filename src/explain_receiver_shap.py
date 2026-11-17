
import os
import time
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. LOAD REAL RECEIVER MODEL
# -----------------------------------------

MODEL_PATH = "models/real/random_forest_receiver.joblib"
DATA_PATH = "dataset/real/real_receiver_test.csv"

model = joblib.load(MODEL_PATH)
test = pd.read_csv(DATA_PATH)

numeric_features = [
    "vehicle_speed",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "heading_noise",
    "position_noise",
    "message_delay_ms",
    "messages_received_last_1s",
    "distance_to_road_edge"
]

categorical_features = ["driver_profile"]
features = numeric_features + categorical_features

X_test = test[features]

# -----------------------------------------
# 2. TRANSFORM TEST DATA
# -----------------------------------------

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# Use a reproducible sample for SHAP
sample_size = min(500, X_test.shape[0])

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

X_sample_transformed = preprocessor.transform(X_sample)

print("Test records:", len(X_test))
print("SHAP sample records:", sample_size)
print("Transformed features:", len(feature_names))

# -----------------------------------------
# 3. CALCULATE SHAP VALUES
# -----------------------------------------

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(classifier)

start_time = time.perf_counter()

shap_values = explainer.shap_values(
    X_sample_transformed
)

shap_time = time.perf_counter() - start_time

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_matrix = shap_values[1]
else:
    shap_matrix = shap_values

if len(shap_matrix.shape) == 3:
    shap_matrix = shap_matrix[:, :, 1]

print("SHAP calculation completed.")
print("SHAP time:", round(shap_time, 6), "seconds")
print(
    "Average explanation time:",
    round(shap_time / sample_size * 1000, 6),
    "ms/record"
)

# -----------------------------------------
# 4. GLOBAL FEATURE IMPORTANCE
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": abs(shap_matrix).mean(axis=0)
})

importance = importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

os.makedirs("outputs/real", exist_ok=True)

importance.to_csv(
    "outputs/real/receiver_shap_importance.csv",
    index=False
)

print("\n===== SHAP FEATURE IMPORTANCE =====")
print(importance.to_string(index=False))

# -----------------------------------------
# 5. SHAP BAR GRAPH
# -----------------------------------------

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_matrix,
    X_sample_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title("Receiver-Side SHAP Global Feature Importance")
plt.tight_layout()

plt.savefig(
    "outputs/real/receiver_shap_global.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------
# 6. SHAP BEESWARM GRAPH
# -----------------------------------------

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_matrix,
    X_sample_transformed,
    feature_names=feature_names,
    show=False
)

plt.title("Receiver-Side SHAP Summary Plot")
plt.tight_layout()

plt.savefig(
    "outputs/real/receiver_shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n===== SHAP ANALYSIS COMPLETED =====")
print("Saved:")
print("outputs/real/receiver_shap_importance.csv")
print("outputs/real/receiver_shap_global.png")
print("outputs/real/receiver_shap_beeswarm.png")
