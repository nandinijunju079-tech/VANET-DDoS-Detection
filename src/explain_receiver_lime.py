
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# 1. Load model and datasets
model = joblib.load(
    "models/real/random_forest_receiver.joblib"
)

train = pd.read_csv("dataset/real/real_receiver_train.csv")
test = pd.read_csv("dataset/real/real_receiver_test.csv")

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

# 2. Access preprocessing and classifier
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

X_train = train[features]
X_test = test[features]

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

if hasattr(X_train_transformed, "toarray"):
    X_train_transformed = X_train_transformed.toarray()

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

# 3. Create a background sample
rng = np.random.default_rng(42)
sample_indices = rng.choice(
    len(X_train_transformed),
    size=min(1000, len(X_train_transformed)),
    replace=False
)

background = X_train_transformed[sample_indices]

# 4. Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=background,
    feature_names=list(feature_names),
    class_names=["Benign", "Attacker"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

# 5. Find correctly classified examples
predictions = classifier.predict(X_test_transformed)
actual = test["attacker"].to_numpy()

benign_candidates = np.where(
    (actual == 0) & (predictions == 0)
)[0]

attacker_candidates = np.where(
    (actual == 1) & (predictions == 1)
)[0]

if len(benign_candidates) == 0 or len(attacker_candidates) == 0:
    raise RuntimeError("Could not find both correctly classified classes.")

selected = {
    "Benign": int(benign_candidates[0]),
    "Attacker": int(attacker_candidates[0])
}

os.makedirs("outputs/real", exist_ok=True)

# 6. Explain individual predictions
def predict_function(data):
    return classifier.predict_proba(data)

for label, index in selected.items():

    print(f"\n===== LIME EXPLANATION: {label} =====")
    print("Test record index:", index)
    print("Actual label:", actual[index])
    print("Predicted label:", predictions[index])

    start = time.perf_counter()

    explanation = explainer.explain_instance(
        X_test_transformed[index],
        predict_function,
        num_features=12,
        num_samples=3000,
        labels=(0, 1)
    )

    elapsed = time.perf_counter() - start

    print("Explanation time:", round(elapsed, 6), "seconds")
    print("Explanation details:")

    for feature, weight in explanation.as_list(
        label=int(predictions[index])
    ):
        print(f"{feature}: {weight:.6f}")

    safe_label = label.lower()

    explanation.save_to_file(
        f"outputs/real/receiver_lime_{safe_label}.html"
    )

    fig = explanation.as_pyplot_figure(
        label=int(predictions[index])
    )

    fig.tight_layout()
    fig.savefig(
        f"outputs/real/receiver_lime_{safe_label}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

print("\nLIME EXPLANATIONS COMPLETED")
