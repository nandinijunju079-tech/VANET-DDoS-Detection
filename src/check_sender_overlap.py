
import pandas as pd
import os

folder = r"C:\Users\ADMIN\Downloads\VANET_DDoS_project\dataset\real"

train = pd.read_csv(os.path.join(folder, "real_train.csv"))
validation = pd.read_csv(os.path.join(folder, "real_validation.csv"))
test = pd.read_csv(os.path.join(folder, "real_test.csv"))

train_ids = set(train["sender_id"].dropna())
validation_ids = set(validation["sender_id"].dropna())
test_ids = set(test["sender_id"].dropna())

print("Unique Train senders:", len(train_ids))
print("Unique Validation senders:", len(validation_ids))
print("Unique Test senders:", len(test_ids))

print("\nTrain-Validation overlap:", len(train_ids & validation_ids))
print("Train-Test overlap:", len(train_ids & test_ids))
print("Validation-Test overlap:", len(validation_ids & test_ids))

all_overlap = train_ids & validation_ids & test_ids
print("Present in all three datasets:", len(all_overlap))
print("\nOverlapping sender IDs:")

print("\nTrain - Validation:")
print(sorted(train_ids & validation_ids))

print("\nTrain - Test:")
print(sorted(train_ids & test_ids))

print("\nValidation - Test:")
print(sorted(validation_ids & test_ids))
print("\nSENDER OVERLAP CHECK COMPLETED")
