import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------
# Read CSV file
# --------------------------------

data = pd.read_csv("model_data.csv")

print("Available Columns:")
print(data.columns.tolist())

# --------------------------------
# User input
# --------------------------------

features_input = input(
    "\nEnter feature names separated by comma: "
)

target = input(
    "Enter target variable name: "
)

# Convert input into list
features = [
    feature.strip()
    for feature in features_input.split(",")
]

# --------------------------------
# Features and target
# --------------------------------

X = data[features]
y = data[target]

# --------------------------------
# Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------
# Decision Tree Model
# --------------------------------

model = DecisionTreeClassifier(
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# --------------------------------
# Evaluation metrics
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# --------------------------------
# Display results
# --------------------------------

print("\nModel Performance")
print("-----------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

# --------------------------------
# Actual vs Predicted
# --------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted")
print("-------------------")
print(results)