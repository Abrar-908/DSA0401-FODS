# ============================================================
# QUESTION 2
# CUSTOMER CHURN PREDICTION USING LOGISTIC REGRESSION
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 1. CREATE THE DATASET
# ------------------------------------------------------------

data = {
    "Customer": ["C1", "C2", "C3", "C4",
                 "C5", "C6", "C7", "C8"],

    "Monthly_Bill": [500, 900, 750, 400,
                     1000, 550, 850, 450],

    "Complaints": [0, 3, 2, 0,
                   4, 1, 3, 0],

    "Data_Usage_GB": [25, 10, 15, 30,
                      8, 22, 12, 28],

    "Tenure_Months": [36, 8, 12, 48,
                      6, 30, 10, 40],

    "Churn": [0, 1, 1, 0,
              1, 0, 1, 0]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 60)
print("CUSTOMER CHURN DATASET")
print("=" * 60)

print(df)


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. IDENTIFY INPUT FEATURES AND TARGET
# ------------------------------------------------------------

X = df[
    [
        "Monthly_Bill",
        "Complaints",
        "Data_Usage_GB",
        "Tenure_Months"
    ]
]

y = df["Churn"]


print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Variable:")
print("Churn")


# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAINING DATA")
print("=" * 60)

print(X_train)

print("\nTraining Target:")
print(y_train)


print("\n" + "=" * 60)
print("TESTING DATA")
print("=" * 60)

print(X_test)

print("\nTesting Target:")
print(y_test)


# ------------------------------------------------------------
# 6. CREATE LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)


print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)


# ------------------------------------------------------------
# 8. DISPLAY MODEL COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nLogistic Regression Coefficients:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_[0])


# ------------------------------------------------------------
# 9. PREDICT TEST CUSTOMERS
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 10. DISPLAY TEST PREDICTIONS
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Churn"] = y_test.values

results["Predicted Churn"] = y_pred

results["Churn Probability"] = y_probability


print("\n" + "=" * 60)
print("CUSTOMER CHURN PREDICTIONS")
print("=" * 60)

print(results)


# ------------------------------------------------------------
# 11. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 13. PRECISION
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 14. RECALL
# ------------------------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 15. F1 SCORE
# ------------------------------------------------------------

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# 16. DISPLAY EVALUATION METRICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 17. CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Churn"]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix - Customer Churn Prediction")

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. ACTUAL VS PREDICTED CHURN
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_positions = np.arange(len(y_test))

plt.plot(
    x_positions,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Churn"
)

plt.plot(
    x_positions,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Churn"
)

plt.xlabel("Test Customer")
plt.ylabel("Churn Status")
plt.title("Actual vs Predicted Customer Churn")

plt.yticks(
    [0, 1],
    ["Stay", "Churn"]
)

plt.xticks(
    x_positions,
    X_test.index
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. FEATURE COEFFICIENT VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Features")
plt.ylabel("Logistic Regression Coefficient")

plt.title(
    "Feature Coefficients - Customer Churn Prediction"
)

plt.xticks(rotation=20)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. CHURN DISTRIBUTION
# ------------------------------------------------------------

churn_counts = df["Churn"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Stay", "Churn"],
    [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ]
)

plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.title("Customer Churn Distribution")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. PREDICT CHURN FOR A NEW CUSTOMER
# ------------------------------------------------------------

new_customer = pd.DataFrame({
    "Monthly_Bill": [800],
    "Complaints": [2],
    "Data_Usage_GB": [14],
    "Tenure_Months": [10]
})


new_prediction = model.predict(
    new_customer
)

new_probability = model.predict_proba(
    new_customer
)[0][1]


print("\n" + "=" * 60)
print("NEW CUSTOMER CHURN PREDICTION")
print("=" * 60)

print("\nNew Customer Details:")
print(new_customer)

print(
    "\nChurn Probability:",
    round(new_probability * 100, 2),
    "%"
)

if new_prediction[0] == 1:
    print("Predicted Status: CHURN")
else:
    print("Predicted Status: STAY")


# ------------------------------------------------------------
# 22. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
Logistic Regression is appropriate for customer churn
prediction because the target variable has two classes:

0 = Customer stays
1 = Customer churns

The model learns the relationship between monthly bill,
complaints, data usage and customer tenure.

Accuracy shows the overall percentage of correctly
classified customers.

Precision shows how many customers predicted as churn
actually churn.

Recall shows how many actual churn customers were
correctly identified.

F1-score provides a balance between precision and recall.

The model can help the telecom company identify customers
who are likely to leave and take appropriate retention
actions such as offers, discounts or improved customer
support.
""")
