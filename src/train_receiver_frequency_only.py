
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

BASE = "dataset/real"

train = pd.read_csv(f"{BASE}/real_receiver_train.csv")
validation = pd.read_csv(f"{BASE}/real_receiver_validation.csv")
test = pd.read_csv(f"{BASE}/real_receiver_test.csv")

feature = ["messages_received_last_1s"]

X_train = train[feature]
y_train = train["attacker"]

X_val = validation[feature]
y_val = validation["attacker"]

X_test = test[feature]
y_test = test["attacker"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

def evaluate(name, X, y):
    predictions = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("Confusion Matrix:")
    print([[tn, fp], [fn, tp]])
    print(f"Accuracy  : {accuracy_score(y, predictions):.6f}")
    print(f"Precision : {precision_score(y, predictions, zero_division=0):.6f}")
    print(f"Recall    : {recall_score(y, predictions, zero_division=0):.6f}")
    print(f"F1        : {f1_score(y, predictions, zero_division=0):.6f}")
    print(f"FPR       : {fp / (fp + tn):.6f}")

print("===== RECEIVER FREQUENCY-ONLY MODEL =====")

evaluate("Validation", X_val, y_val)
evaluate("Test", X_test, y_test)
