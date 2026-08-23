import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Read CSV
data = pd.read_csv("treatment_data.csv")

# Encode Gender
gender_encoder = LabelEncoder()

data["Gender"] = gender_encoder.fit_transform(
    data["Gender"]
)

# Encode Outcome
outcome_encoder = LabelEncoder()

data["Outcome"] = outcome_encoder.fit_transform(
    data["Outcome"]
)

# Features
X = data[[
    "Age",
    "Gender",
    "BloodPressure",
    "Cholesterol"
]]

# Target
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

# Display metrics
print("KNN Treatment Outcome Prediction")
print("---------------------------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

# Actual vs predicted
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results["Actual"] = outcome_encoder.inverse_transform(
    results["Actual"]
)

results["Predicted"] = outcome_encoder.inverse_transform(
    results["Predicted"]
)

print("\nTest Set Predictions")
print("--------------------")
print(results)