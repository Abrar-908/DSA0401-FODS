# ============================================================
# QUESTION 7
# HEALTHCARE TREATMENT COST PREDICTION
# USING LASSO REGRESSION
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. CREATE THE DATASET
# ------------------------------------------------------------

data = {
    "Patient": [
        "P1", "P2", "P3", "P4",
        "P5", "P6", "P7", "P8"
    ],

    "Age": [
        25, 35, 45, 55,
        60, 30, 50, 40
    ],

    "BP": [
        115, 125, 140, 155,
        165, 120, 150, 135
    ],

    "Sugar_Level": [
        90, 110, 150, 180,
        200, 100, 170, 130
    ],

    "BMI": [
        22, 25, 29, 32,
        35, 24, 31, 27
    ],

    "Previous_Visits": [
        1, 2, 3, 5,
        6, 1, 4, 2
    ],

    "Treatment_Cost": [
        5000, 8000, 15000, 25000,
        32000, 7000, 22000, 12000
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 70)
print("HEALTHCARE TREATMENT COST DATASET")
print("=" * 70)

print(df)


# ------------------------------------------------------------
# 3. BASIC DATASET INFORMATION
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
        "Age",
        "BP",
        "Sugar_Level",
        "BMI",
        "Previous_Visits"
    ]
]

y = df["Treatment_Cost"]


print("\n" + "=" * 70)
print("INPUT AND OUTPUT VARIABLES")
print("=" * 70)

print("Independent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Treatment_Cost")


# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
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
# 6. LASSO REGRESSION USING ALPHA = 0.1
# ------------------------------------------------------------

alpha_value = 0.1

model = Lasso(
    alpha=alpha_value,
    max_iter=10000
)


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)


print("\n" + "=" * 70)
print("LASSO REGRESSION MODEL")
print("=" * 70)

print("Alpha:", alpha_value)

print("Model training completed.")


# ------------------------------------------------------------
# 8. DISPLAY COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})


print("\nLasso Regression Coefficients:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_)


# ------------------------------------------------------------
# 9. IDENTIFY IMPORTANT FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

for feature, coefficient in zip(
    X.columns,
    model.coef_
):

    if abs(coefficient) < 1e-6:

        print(
            feature,
            "-> Removed / coefficient approximately zero"
        )

    else:

        print(
            feature,
            "-> Important, coefficient =",
            round(coefficient, 4)
        )


# ------------------------------------------------------------
# 10. PREDICT TEST PATIENTS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 11. CREATE PREDICTION TABLE
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Cost"] = y_test.values

results["Predicted Cost"] = y_pred

results["Error"] = (
    results["Actual Cost"]
    - results["Predicted Cost"]
)


print("\n" + "=" * 70)
print("TEST PATIENT PREDICTIONS")
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
# 13. PREDICT TREATMENT COST FOR NEW PATIENT
# ------------------------------------------------------------

new_patient = pd.DataFrame({
    "Age": [48],
    "BP": [145],
    "Sugar_Level": [160],
    "BMI": [30],
    "Previous_Visits": [4]
})


new_prediction = model.predict(
    new_patient
)


print("\n" + "=" * 70)
print("NEW PATIENT TREATMENT COST PREDICTION")
print("=" * 70)

print("\nNew Patient Details:")
print(new_patient)

print(
    "\nPredicted Treatment Cost:",
    round(new_prediction[0], 2)
)


# ------------------------------------------------------------
# 14. ACTUAL VS PREDICTED COST
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Cost"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Cost"
)

plt.xlabel("Test Patient")

plt.ylabel("Treatment Cost")

plt.title(
    "Actual vs Predicted Treatment Cost"
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

plt.xlabel("Actual Treatment Cost")

plt.ylabel("Predicted Treatment Cost")

plt.title(
    "Actual vs Predicted Treatment Cost"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. LASSO COEFFICIENT VISUALIZATION
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

plt.xlabel("Medical Feature")

plt.ylabel("Lasso Coefficient")

plt.title(
    "Lasso Regression Coefficients"
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
# 17. AGE VS TREATMENT COST
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Age"],
    df["Treatment_Cost"],
    s=100
)

plt.xlabel("Patient Age")

plt.ylabel("Treatment Cost")

plt.title(
    "Age vs Treatment Cost"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. SUGAR LEVEL VS TREATMENT COST
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Sugar_Level"],
    df["Treatment_Cost"],
    s=100
)

plt.xlabel("Sugar Level")

plt.ylabel("Treatment Cost")

plt.title(
    "Sugar Level vs Treatment Cost"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. BMI VS TREATMENT COST
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["BMI"],
    df["Treatment_Cost"],
    s=100
)

plt.xlabel("BMI")

plt.ylabel("Treatment Cost")

plt.title(
    "BMI vs Treatment Cost"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. PREVIOUS VISITS VS TREATMENT COST
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Previous_Visits"],
    df["Treatment_Cost"],
    s=100
)

plt.xlabel("Previous Visits")

plt.ylabel("Treatment Cost")

plt.title(
    "Previous Visits vs Treatment Cost"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "Age",
        "BP",
        "Sugar_Level",
        "BMI",
        "Previous_Visits",
        "Treatment_Cost"
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
    "Correlation Matrix - Healthcare Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. COMPARE DIFFERENT ALPHA VALUES
# ------------------------------------------------------------

alpha_values = [
    0.1,
    1.0,
    10.0
]

alpha_results = []

for alpha in alpha_values:

    temp_model = Lasso(
        alpha=alpha,
        max_iter=10000
    )

    temp_model.fit(
        X_train,
        y_train
    )

    temp_prediction = temp_model.predict(
        X_test
    )

    temp_mae = mean_absolute_error(
        y_test,
        temp_prediction
    )

    temp_mse = mean_squared_error(
        y_test,
        temp_prediction
    )

    temp_rmse = np.sqrt(
        temp_mse
    )

    temp_r2 = r2_score(
        y_test,
        temp_prediction
    )

    alpha_results.append([
        alpha,
        temp_mae,
        temp_mse,
        temp_rmse,
        temp_r2
    ])


alpha_df = pd.DataFrame(
    alpha_results,
    columns=[
        "Alpha",
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]
)


print("\n" + "=" * 70)
print("ALPHA COMPARISON")
print("=" * 70)

print(alpha_df)


# ------------------------------------------------------------
# 23. ALPHA VS R2 SCORE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    alpha_df["Alpha"],
    alpha_df["R2"],
    marker="o",
    linewidth=2
)

plt.xlabel("Alpha")

plt.ylabel("R² Score")

plt.title(
    "Effect of Alpha on Lasso R² Score"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. ALPHA VS RMSE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    alpha_df["Alpha"],
    alpha_df["RMSE"],
    marker="o",
    linewidth=2
)

plt.xlabel("Alpha")

plt.ylabel("RMSE")

plt.title(
    "Effect of Alpha on Lasso RMSE"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 25. COMPARE COEFFICIENTS FOR DIFFERENT ALPHAS
# ------------------------------------------------------------

coefficient_comparison = []

for alpha in alpha_values:

    temp_model = Lasso(
        alpha=alpha,
        max_iter=10000
    )

    temp_model.fit(
        X_train,
        y_train
    )

    coefficient_comparison.append(
        temp_model.coef_
    )


coefficient_df = pd.DataFrame(
    coefficient_comparison,
    columns=X.columns,
    index=alpha_values
)

print("\n" + "=" * 70)
print("COEFFICIENT COMPARISON")
print("=" * 70)

print(coefficient_df)


# ------------------------------------------------------------
# 26. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Lasso Regression is useful for healthcare treatment cost
prediction because it performs regression while also
providing feature selection.

The model uses Age, BP, Sugar Level, BMI and Previous
Visits to predict treatment cost.

Lasso applies L1 regularization to the regression
coefficients.

When a coefficient becomes zero or approaches zero,
the corresponding feature has a reduced contribution
to the model.

A lower MAE and RMSE indicate smaller prediction errors.

A higher R² score indicates better model performance.

The alpha value controls the strength of L1
regularization.

The hospital can use the model to estimate treatment
costs and identify which available medical parameters
are most useful for prediction.
""")
