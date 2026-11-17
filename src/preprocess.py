
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

# 1. Load dataset
df = pd.read_csv("dataset/vanet_dataset.csv")

print("Original Dataset Shape:", df.shape)

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Remove duplicate records
df = df.drop_duplicates()

# 4. Separate features and target
X = df.drop("label", axis=1)
y = df["label"].map({"Normal": 0, "DDoS": 1})

# 5. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Handle missing values using training data
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)


# 7. Feature selection / ranking
mi_scores = mutual_info_classif(
    X_train,
    y_train,
    random_state=42
)

feature_ranking = pd.DataFrame({
    "Feature": X.columns,
    "MI_Score": mi_scores
}).sort_values(by="MI_Score", ascending=False)

print("\nFeature Importance Ranking:")
print(feature_ranking)

# 8. Create output folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 9. Save preprocessed dataset
processed_df = X.copy()
processed_df["label"] = y
processed_df.to_csv(
    "dataset/processed_dataset.csv",
    index=False
)

# 10. Save training and testing data
joblib.dump({
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": list(X.columns),
    "imputer": imputer
}, "models/preprocessed_data.joblib")

feature_ranking.to_csv(
    "outputs/feature_ranking.csv",
    index=False
)

print("\nPreprocessing Completed Successfully!")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Processed dataset saved successfully!")
