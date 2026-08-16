# ============================================================
# QUESTION 6
# LOAN APPROVAL PREDICTION USING LOGISTIC REGRESSION
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
    "Applicant": [
        "A1", "A2", "A3", "A4",
        "A5", "A6", "A7", "A8"
    ],

    "Income": [
        25000, 30000, 45000, 60000,
        70000, 28000, 55000, 35000
    ],

    "Credit_Score": [
        580, 600, 680, 750,
        800, 590, 720, 620
    ],

    "Existing_Loan": [
        200000, 180000, 120000, 80000,
        50000, 220000, 100000, 160000
    ],

    "Employment_Years": [
        1, 2, 3, 5,
        6, 1, 4, 2
    ],

    "Approved": [
        0, 0, 1, 1,
        1, 0, 1, 0
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 65)
print("LOAN APPROVAL DATASET")
print("=" * 65)

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
        "Income",
        "Credit_Score",
        "Existing_Loan",
        "Employment_Years"
    ]
]

y = df["Approved"]


print("\n" + "=" * 65)
print("INPUT AND OUTPUT VARIABLES")
print("=" * 65)

print("Independent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Approved")


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


print("\n" + "=" * 65)
print("TRAINING DATA")
print("=" * 65)

print(X_train)

print("\nTraining Target:")
print(y_train)


print("\n" + "=" * 65)
print("TESTING DATA")
print("=" * 65)

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


print("\n" + "=" * 65)
print("LOGISTIC REGRESSION MODEL")
print("=" * 65)

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
# 9. PREDICT TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 10. DISPLAY PREDICTIONS
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Approval"] = y_test.values

results["Predicted Approval"] = y_pred

results["Approval Probability"] = y_probability


print("\n" + "=" * 65)
print("LOAN APPROVAL PREDICTIONS")
print("=" * 65)

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

print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

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
    display_labels=["Rejected", "Approved"]
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Confusion Matrix - Loan Approval Prediction"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. ACTUAL VS PREDICTED APPROVAL
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

plt.xlabel("Test Applicant")
plt.ylabel("Approval Status")

plt.title(
    "Actual vs Predicted Loan Approval"
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
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

plt.xlabel("Features")

plt.ylabel("Logistic Regression Coefficient")

plt.title(
    "Feature Coefficients - Loan Approval"
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
# 20. CREDIT SCORE VS APPROVAL
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Credit_Score"],
    df["Approved"],
    s=100
)

plt.xlabel("Credit Score")

plt.ylabel("Approval Status")

plt.title(
    "Credit Score vs Loan Approval"
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. INCOME VS APPROVAL
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Income"],
    df["Approved"],
    s=100
)

plt.xlabel("Applicant Income")

plt.ylabel("Approval Status")

plt.title(
    "Applicant Income vs Loan Approval"
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. EXISTING LOAN VS APPROVAL
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Existing_Loan"],
    df["Approved"],
    s=100
)

plt.xlabel("Existing Loan")

plt.ylabel("Approval Status")

plt.title(
    "Existing Loan vs Loan Approval"
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. EMPLOYMENT YEARS VS APPROVAL
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Employment_Years"],
    df["Approved"],
    s=100
)

plt.xlabel("Employment Years")

plt.ylabel("Approval Status")

plt.title(
    "Employment Years vs Loan Approval"
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. APPROVAL DISTRIBUTION
# ------------------------------------------------------------

approval_counts = df["Approved"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Rejected", "Approved"],
    [
        approval_counts.get(0, 0),
        approval_counts.get(1, 0)
    ]
)

plt.xlabel("Loan Decision")

plt.ylabel("Number of Applicants")

plt.title(
    "Loan Approval Distribution"
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
        "Income",
        "Credit_Score",
        "Existing_Loan",
        "Employment_Years",
        "Approved"
    ]
].corr()


plt.figure(figsize=(8, 6))

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
    "Correlation Matrix - Loan Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 26. PREDICT NEW LOAN APPLICATION
# ------------------------------------------------------------

new_applicant = pd.DataFrame({
    "Income": [50000],
    "Credit_Score": [700],
    "Existing_Loan": [100000],
    "Employment_Years": [4]
})


new_prediction = model.predict(
    new_applicant
)

new_probability = model.predict_proba(
    new_applicant
)[0][1]


print("\n" + "=" * 65)
print("NEW LOAN APPLICATION PREDICTION")
print("=" * 65)

print("\nApplicant Details:")
print(new_applicant)

print(
    "\nApproval Probability:",
    round(new_probability * 100, 2),
    "%"
)

if new_prediction[0] == 1:

    print(
        "Predicted Decision: LOAN APPROVED"
    )

else:

    print(
        "Predicted Decision: LOAN REJECTED"
    )


# ------------------------------------------------------------
# 27. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)

print("""
Logistic Regression is suitable for loan approval
prediction because the target variable is binary.

Approved = 1 means the loan is approved.
Approved = 0 means the loan is rejected.

The model uses applicant income, credit score,
existing loan amount and employment years to estimate
the probability of loan approval.

Accuracy measures the overall percentage of correct
predictions.

Precision measures how many predicted approvals are
actually approved.

Recall measures how many actual approved applications
are correctly identified.

F1-score provides a balance between precision and recall.

The model can assist a bank in evaluating loan
applications and supporting faster and more consistent
loan decisions.
""")
