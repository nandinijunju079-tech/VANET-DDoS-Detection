

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer

# ------------------------------------------------
# 1. CREATE OUTPUT FOLDER
# ------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------
# 2. LOAD TRAINED MODEL AND DATA
# ------------------------------------------------

rf_model = joblib.load("models/random_forest_model.joblib")
data = joblib.load("models/preprocessed_data.joblib")

X_train = np.asarray(data["X_train"])
X_test = np.asarray(data["X_test"])

feature_names = [
    "vehicle_speed",
    "vehicle_density",
    "neighbor_count",
    "packet_rate",
    "packet_size",
    "flow_duration",
    "protocol",
    "emergency_priority"
]

class_names = ["Normal", "DDoS"]
ddos_index = list(rf_model.classes_).index(1)

# ------------------------------------------------
# 3. TRAFFIC PREDICTION
# ------------------------------------------------

sample = X_test[0].reshape(1, -1)

prediction = rf_model.predict(sample)[0]
probabilities = rf_model.predict_proba(sample)[0]

ddos_probability = probabilities[ddos_index] * 100
predicted_class = "DDoS" if prediction == 1 else "Normal"

print("\n----- TRAFFIC PREDICTION -----")
print("Predicted Class:", predicted_class)
print("DDoS Probability:", round(ddos_probability, 2), "%")

# ------------------------------------------------
# 4. SHAP EXPLANATION AND TIMING
# ------------------------------------------------

print("\n----- SHAP EXPLANATION -----")

shap_setup_start = time.perf_counter()

explainer = shap.TreeExplainer(rf_model)

shap_setup_time = time.perf_counter() - shap_setup_start

shap_start = time.perf_counter()

shap_values = explainer.shap_values(X_test[:50])

shap_time = time.perf_counter() - shap_start

# Handle different SHAP output formats

if isinstance(shap_values, list):
    class_shap = shap_values[ddos_index]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    class_shap = shap_values[:, :, ddos_index]

else:
    class_shap = shap_values

# Global feature importance

importance = np.mean(np.abs(class_shap), axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Importance": importance
})

shap_importance = shap_importance.sort_values(
    by="SHAP_Importance",
    ascending=False
)

print("\nGlobal Feature Importance:")
print(shap_importance.to_string(index=False))

shap_importance.to_csv(
    "outputs/shap_feature_importance.csv",
    index=False
)

# SHAP graph

plt.figure(figsize=(9, 5))

plt.barh(
    shap_importance["Feature"],
    shap_importance["SHAP_Importance"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Features")
plt.title("SHAP Global Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("outputs/shap_global.png")
plt.close()

print("\nSHAP explanation generated successfully!")

# ------------------------------------------------
# 5. LIME EXPLANATION AND TIMING
# ------------------------------------------------

print("\n----- LIME EXPLANATION -----")

lime_setup_start = time.perf_counter()

lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    discretize_continuous=True
)

lime_setup_time = time.perf_counter() - lime_setup_start

lime_start = time.perf_counter()

lime_exp = lime_explainer.explain_instance(
    data_row=X_test[0],
    predict_fn=rf_model.predict_proba,
    num_features=len(feature_names),
    labels=[0, 1]
)

lime_time = time.perf_counter() - lime_start

print("\nLIME Feature Contributions:")

for feature, contribution in lime_exp.as_list(label=ddos_index):
    print(feature, ":", round(contribution, 5))

lime_exp.save_to_file(
    "outputs/lime_explanation.html"
)

fig = lime_exp.as_pyplot_figure(label=ddos_index)

fig.savefig(
    "outputs/lime_explanation.png",
    bbox_inches="tight"
)

plt.close(fig)

print("\nLIME explanation generated successfully!")

# ------------------------------------------------
# 6. CALCULATE XAI OVERHEAD
# ------------------------------------------------

shap_records = len(X_test[:50])

shap_average_time = shap_time / shap_records

total_explanation_time = shap_time + lime_time

total_setup_time = shap_setup_time + lime_setup_time

print("\n----- XAI COMPUTATIONAL OVERHEAD -----")

print("SHAP records explained:", shap_records)
print("SHAP calculation time:", round(shap_time, 6), "seconds")
print("Average SHAP time per record:",
      round(shap_average_time * 1000, 6), "ms")

print("LIME explanation time:",
      round(lime_time, 6), "seconds")

print("Total explanation time:",
      round(total_explanation_time, 6), "seconds")

print("Explainer setup time:",
      round(total_setup_time, 6), "seconds")

# Save measurements

overhead_results = pd.DataFrame([{
    "SHAP_Records": shap_records,
    "SHAP_Calculation_Time_sec": shap_time,
    "Average_SHAP_Time_per_Record_ms": shap_average_time * 1000,
    "LIME_Records": 1,
    "LIME_Explanation_Time_sec": lime_time,
    "Total_Explanation_Time_sec": total_explanation_time,
    "SHAP_Setup_Time_sec": shap_setup_time,
    "LIME_Setup_Time_sec": lime_setup_time,
    "Total_Setup_Time_sec": total_setup_time
}])

overhead_results.to_csv(
    "outputs/xai_overhead.csv",
    index=False
)

# ------------------------------------------------
# 7. FINAL STATUS
# ------------------------------------------------

print("\n--------------------------------------")
print("SHAP AND LIME COMPLETED SUCCESSFULLY!")
print("--------------------------------------")

print("Generated files are available in outputs folder.")
print("XAI overhead saved to outputs/xai_overhead.csv")
