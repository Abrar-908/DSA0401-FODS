# ============================================================
# QUESTION 10
# EMPLOYEE ATTRITION PREDICTION
# USING LOGISTIC REGRESSION
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
    "Employee": [
        "E1", "E2", "E3", "E4",
        "E5", "E6", "E7", "E8"
    ],

    "Experience": [
        2, 3, 1, 5,
        6, 8, 4, 7
    ],

    "Satisfaction_Score": [
        8, 7, 4, 6,
        3, 2, 7, 4
    ],

    "Overtime_Hours": [
        2, 4, 10, 6,
        12, 14, 3, 11
    ],

    "Salary_Increment": [
        12, 10, 5, 8,
        4, 3, 11, 5
    ],

    "Leave": [
        0, 0, 1, 0,
        1, 1, 0, 1
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 70)
print("EMPLOYEE ATTRITION DATASET")
print("=" * 70)

print(df)


# ------------------------------------------------------------
# 3. DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 4. IDENTIFY FEATURES AND TARGET
# ------------------------------------------------------------

X = df[
    [
        "Experience",
        "Satisfaction_Score",
        "Overtime_Hours",
        "Salary_Increment"
    ]
]

y = df["Leave"]


print("\n" + "=" * 70)
print("INPUT AND OUTPUT VARIABLES")
print("=" * 70)

print("Independent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Leave")


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


print("\n" + "=" * 70)
print("TRAINING DATA")
print("=" * 70)

print(X_train)

print("\nTraining Target:")
print(y_train)


print("\n" + "=" * 70)
print("TESTING DATA")
print("=" * 70)

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

model.fit(
    X_train,
    y_train
)


print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL")
print("=" * 70)

print("Model training completed.")


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
# 9. PREDICT ATTRITION FOR TEST EMPLOYEES
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 10. CREATE PREDICTION TABLE
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Leave"] = y_test.values

results["Predicted Leave"] = y_pred

results["Leave Probability"] = y_probability


print("\n" + "=" * 70)
print("EMPLOYEE ATTRITION PREDICTIONS")
print("=" * 70)

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
# 16. DISPLAY EVALUATION RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    "Accuracy  :",
    round(accuracy, 4)
)

print(
    "Precision :",
    round(precision, 4)
)

print(
    "Recall    :",
    round(recall, 4)
)

print(
    "F1 Score  :",
    round(f1, 4)
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# 17. CONFUSION MATRIX VISUALIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Leave"]
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Confusion Matrix - Employee Attrition Prediction"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. ACTUAL VS PREDICTED ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted"
)

plt.xlabel("Test Employee")

plt.ylabel("Leave Status")

plt.title(
    "Actual vs Predicted Employee Attrition"
)

plt.yticks(
    [0, 1],
    ["Stay", "Leave"]
)

plt.xticks(
    x_values,
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

plt.xlabel("Employee Feature")

plt.ylabel("Logistic Regression Coefficient")

plt.title(
    "Feature Coefficients - Employee Attrition"
)

plt.xticks(
    rotation=20
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. SATISFACTION SCORE VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Satisfaction_Score"],
    df["Leave"],
    s=100
)

plt.xlabel("Satisfaction Score")

plt.ylabel("Leave Status")

plt.title(
    "Satisfaction Score vs Employee Attrition"
)

plt.yticks(
    [0, 1],
    ["Stay", "Leave"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. OVERTIME HOURS VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Overtime_Hours"],
    df["Leave"],
    s=100
)

plt.xlabel("Overtime Hours")

plt.ylabel("Leave Status")

plt.title(
    "Overtime Hours vs Employee Attrition"
)

plt.yticks(
    [0, 1],
    ["Stay", "Leave"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. SALARY INCREMENT VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Salary_Increment"],
    df["Leave"],
    s=100
)

plt.xlabel("Salary Increment (%)")

plt.ylabel("Leave Status")

plt.title(
    "Salary Increment vs Employee Attrition"
)

plt.yticks(
    [0, 1],
    ["Stay", "Leave"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. EXPERIENCE VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Experience"],
    df["Leave"],
    s=100
)

plt.xlabel("Experience (Years)")

plt.ylabel("Leave Status")

plt.title(
    "Experience vs Employee Attrition"
)

plt.yticks(
    [0, 1],
    ["Stay", "Leave"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. ATTRITION DISTRIBUTION
# ------------------------------------------------------------

leave_counts = df["Leave"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Stay", "Leave"],
    [
        leave_counts.get(0, 0),
        leave_counts.get(1, 0)
    ]
)

plt.xlabel("Employee Status")

plt.ylabel("Number of Employees")

plt.title(
    "Employee Attrition Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 25. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "Experience",
        "Satisfaction_Score",
        "Overtime_Hours",
        "Salary_Increment",
        "Leave"
    ]
].corr()


plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    cmap="coolwarm"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title(
    "Correlation Matrix - Employee Attrition Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 26. PREDICT ATTRITION FOR A NEW EMPLOYEE
# ------------------------------------------------------------

new_employee = pd.DataFrame({
    "Experience": [5],
    "Satisfaction_Score": [4],
    "Overtime_Hours": [9],
    "Salary_Increment": [6]
})


new_prediction = model.predict(
    new_employee
)

new_probability = model.predict_proba(
    new_employee
)[0][1]


print("\n" + "=" * 70)
print("NEW EMPLOYEE ATTRITION PREDICTION")
print("=" * 70)

print("\nNew Employee Details:")
print(new_employee)

print(
    "\nLeave Probability:",
    round(new_probability * 100, 2),
    "%"
)

if new_prediction[0] == 1:

    print(
        "Predicted Status: EMPLOYEE MAY LEAVE"
    )

else:

    print(
        "Predicted Status: EMPLOYEE MAY STAY"
    )


# ------------------------------------------------------------
# 27. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Logistic Regression is used to predict employee attrition
because the target variable is binary.

Leave = 1 means the employee may leave.
Leave = 0 means the employee may stay.

The model uses Experience, Satisfaction Score,
Overtime Hours and Salary Increment Percentage.

Accuracy measures the overall percentage of correct
predictions.

Precision measures how many employees predicted to leave
actually belong to the leave class.

Recall measures how many actual employees who may leave
are correctly identified.

F1-score provides a balance between precision and recall.

The HR department can use the model to identify employees
who may have a higher probability of leaving and take
appropriate retention actions.
""")
