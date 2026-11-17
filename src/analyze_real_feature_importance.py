
import joblib
import pandas as pd

# Load trained Random Forest pipeline
model = joblib.load("models/real/random_forest_real.joblib")

# Extract the correct pipeline steps
rf = model.named_steps["classifier"]
preprocessor = model.named_steps["preprocessing"]

# Get transformed feature names and importance scores
feature_names = preprocessor.get_feature_names_out()
importance = rf.feature_importances_

results = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

results = results.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== REAL DATA FEATURE IMPORTANCE =====")
print(results.to_string(index=False))

# Save results
results.to_csv(
    "outputs/real/real_feature_importance.csv",
    index=False
)

print("\nSaved: outputs/real/real_feature_importance.csv")
