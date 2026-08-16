# ============================================================
# QUESTION 9
# STUDENT SALARY PACKAGE PREDICTION
# USING LINEAR REGRESSION
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. CREATE THE DATASET
# ------------------------------------------------------------

data = {
    "Student": [
        "S1", "S2", "S3", "S4",
        "S5", "S6", "S7", "S8"
    ],

    "CGPA": [
        6.5, 7.0, 7.5, 8.0,
        8.5, 9.0, 6.8, 8.2
    ],

    "Aptitude_Score": [
        60, 65, 70, 78,
        85, 90, 62, 80
    ],

    "Coding_Score": [
        55, 60, 68, 80,
        88, 92, 58, 82
    ],

    "Communication_Score": [
        65, 70, 72, 75,
        80, 85, 68, 78
    ],

    "Package_LPA": [
        3.2, 4.0, 5.0, 6.5,
        8.0, 10.0, 3.8, 7.0
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 70)
print("STUDENT SALARY PACKAGE DATASET")
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
# 4. IDENTIFY INPUT AND OUTPUT VARIABLES
# ------------------------------------------------------------

X = df[
    [
        "CGPA",
        "Aptitude_Score",
        "Coding_Score",
        "Communication_Score"
    ]
]

y = df["Package_LPA"]


print("\n" + "=" * 70)
print("INPUT AND OUTPUT VARIABLES")
print("=" * 70)

print("Input Variables:")
print(X.columns.tolist())

print("\nOutput Variable:")
print("Package_LPA")


# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
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
# 6. CREATE LINEAR REGRESSION MODEL
# ------------------------------------------------------------

model = LinearRegression()


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)


print("\n" + "=" * 70)
print("LINEAR REGRESSION MODEL")
print("=" * 70)

print("Model training completed.")


# ------------------------------------------------------------
# 8. DISPLAY MODEL COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})


print("\nRegression Coefficients:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_)


# ------------------------------------------------------------
# 9. DISPLAY REGRESSION EQUATION
# ------------------------------------------------------------

print("\nRegression Equation:")

print(
    "Package =",
    round(model.intercept_, 4),
    "+",
    round(model.coef_[0], 4),
    "* CGPA",
    "+",
    round(model.coef_[1], 4),
    "* Aptitude",
    "+",
    round(model.coef_[2], 4),
    "* Coding",
    "+",
    round(model.coef_[3], 4),
    "* Communication"
)


# ------------------------------------------------------------
# 10. PREDICT PACKAGE FOR TEST STUDENTS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 11. CREATE PREDICTION TABLE
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Package"] = y_test.values

results["Predicted Package"] = y_pred

results["Error"] = (
    results["Actual Package"]
    - results["Predicted Package"]
)


print("\n" + "=" * 70)
print("TEST STUDENT PREDICTIONS")
print("=" * 70)

print(results)


# ------------------------------------------------------------
# 12. MODEL EVALUATION
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    "Mean Absolute Error (MAE):",
    round(mae, 4)
)

print(
    "Mean Squared Error (MSE):",
    round(mse, 4)
)

print(
    "Root Mean Squared Error (RMSE):",
    round(rmse, 4)
)

print(
    "R² Score:",
    round(r2, 4)
)


# ------------------------------------------------------------
# 13. PREDICT PACKAGE FOR A NEW STUDENT
# ------------------------------------------------------------

new_student = pd.DataFrame({
    "CGPA": [8.1],
    "Aptitude_Score": [82],
    "Coding_Score": [85],
    "Communication_Score": [78]
})


new_prediction = model.predict(
    new_student
)


print("\n" + "=" * 70)
print("NEW STUDENT PACKAGE PREDICTION")
print("=" * 70)

print("\nNew Student Details:")
print(new_student)

print(
    "\nPredicted Salary Package:",
    round(new_prediction[0], 2),
    "LPA"
)


# ------------------------------------------------------------
# 14. ACTUAL VS PREDICTED PACKAGE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Package"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Package"
)

plt.xlabel("Test Student")

plt.ylabel("Package (LPA)")

plt.title(
    "Actual vs Predicted Student Salary Package"
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
# 15. ACTUAL VS PREDICTED SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    s=100
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2
)

plt.xlabel("Actual Package (LPA)")

plt.ylabel("Predicted Package (LPA)")

plt.title(
    "Actual vs Predicted Salary Package"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. FEATURE COEFFICIENT VISUALIZATION
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

plt.xlabel("Student Feature")

plt.ylabel("Regression Coefficient")

plt.title(
    "Linear Regression Feature Coefficients"
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
# 17. CGPA VS PACKAGE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["CGPA"],
    df["Package_LPA"],
    s=100
)

plt.xlabel("CGPA")

plt.ylabel("Package (LPA)")

plt.title(
    "CGPA vs Salary Package"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. CODING SCORE VS PACKAGE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Coding_Score"],
    df["Package_LPA"],
    s=100
)

plt.xlabel("Coding Score")

plt.ylabel("Package (LPA)")

plt.title(
    "Coding Score vs Salary Package"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. APTITUDE SCORE VS PACKAGE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Aptitude_Score"],
    df["Package_LPA"],
    s=100
)

plt.xlabel("Aptitude Score")

plt.ylabel("Package (LPA)")

plt.title(
    "Aptitude Score vs Salary Package"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. COMMUNICATION SCORE VS PACKAGE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Communication_Score"],
    df["Package_LPA"],
    s=100
)

plt.xlabel("Communication Score")

plt.ylabel("Package (LPA)")

plt.title(
    "Communication Score vs Salary Package"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "CGPA",
        "Aptitude_Score",
        "Coding_Score",
        "Communication_Score",
        "Package_LPA"
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
    "Correlation Matrix - Student Placement Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. PREDICTION ERROR PLOT
# ------------------------------------------------------------

errors = (
    y_test.values
    - y_pred
)

plt.figure(figsize=(8, 5))

plt.bar(
    range(len(errors)),
    errors
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Test Student")

plt.ylabel("Prediction Error (LPA)")

plt.title(
    "Salary Package Prediction Errors"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Linear Regression is used to predict the expected
salary package of students based on academic and
skill-related attributes.

The model uses CGPA, Aptitude Score, Coding Score
and Communication Score as input variables.

MAE represents the average absolute difference between
actual and predicted salary packages.

MSE represents the average squared prediction error.

RMSE represents prediction error in the same unit
as salary package.

R² indicates how well the student features explain
the variation in salary package.

The placement cell can use the model to estimate the
expected salary package of students and identify the
skills associated with higher placement packages.
""")
