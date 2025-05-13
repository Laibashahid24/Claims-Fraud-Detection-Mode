import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 1000

# Generate fake data
data = {
    "claim_id": range(1, n + 1),
    "customer_age": np.random.randint(18, 75, n),
    "policy_type": np.random.choice(["Auto", "Home", "Health", "Life"], n),
    "incident_type": np.random.choice(["Collision", "Theft", "Fire", "Injury"], n),
    "claim_amount": np.round(np.random.uniform(500, 20000, n), 2),
    "num_previous_claims": np.random.poisson(1.5, n),
    "region": np.random.choice(["Ontario", "Quebec", "Alberta", "BC"], n),
    "fraudulent": np.random.choice([0, 1], size=n, p=[0.9, 0.1])  # 10% fraud
}

df = pd.DataFrame(data)
print(df.head())

# Save it
df.to_csv("claims_data.csv", index=False)

# ---------------------
# STEP 5: Load & Prepare Data
# ---------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the data again
df = pd.read_csv("claims_data.csv")

# Encode categorical columns
label_cols = ["policy_type", "incident_type", "region"]
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop(["claim_id", "fraudulent"], axis=1)
y = df["fraudulent"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------
# STEP 6: Train a Model
# ---------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ---------------------
# STEP 7: Evaluate the Model
# ---------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
