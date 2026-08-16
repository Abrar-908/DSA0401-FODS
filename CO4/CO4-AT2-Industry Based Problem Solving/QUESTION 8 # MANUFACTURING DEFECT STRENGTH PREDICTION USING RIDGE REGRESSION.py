# ============================================================
# QUESTION 8
# MANUFACTURING DEFECT STRENGTH PREDICTION
# USING RIDGE REGRESSION
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ------------------------------------------------------------
# 1. CREATE THE DATASET
# ------------------------------------------------------------

data = {
    "Product": [
        "P1", "P2", "P3", "P4",
        "P5", "P6", "P7", "P8"
    ],

    "Temperature": [
        70, 75, 80, 85,
        90, 95, 78, 88
    ],

    "Pressure": [
        30, 35, 38, 42,
        45, 50, 36, 44
    ],

    "Machine_Speed": [
        100, 110, 120, 125,
        130, 140, 115, 128
    ],

    "Material_Quality": [
        8, 7, 6, 6,
        5, 4, 7, 5
    ],

    "Strength_Score": [
        75, 78, 80, 82,
        79, 76, 81, 80
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 70)
print("MANUFACTURING STRENGTH DATASET")
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
        "Temperature",
        "Pressure",
        "Machine_Speed",
        "Material_Quality"
    ]
]

y = df["Strength_Score"]


print("\n" + "=" * 70)
print("INPUT AND OUTPUT VARIABLES")
print("=" * 70)

print("Independent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Strength_Score")


# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING
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
# 6. CREATE RIDGE REGRESSION MODEL
# ------------------------------------------------------------

# Alpha required by the question
alpha_value = 1.0

model = Ridge(
    alpha=alpha_value
)


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(
    X_train,
    y_train
)


print("\n" + "=" * 70)
print("RIDGE REGRESSION MODEL")
print("=" * 70)

print("Alpha:", alpha_value)

print("Model training completed.")


# ------------------------------------------------------------
# 8. DISPLAY RIDGE COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})


print("\nRidge Regression Coefficients:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_)


# ------------------------------------------------------------
# 9. PREDICT STRENGTH SCORE FOR TEST PRODUCTS
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 10. CREATE PREDICTION TABLE
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Strength"] = y_test.values

results["Predicted Strength"] = y_pred

results["Error"] = (
    results["Actual Strength"]
    - results["Predicted Strength"]
)


print("\n" + "=" * 70)
print("TEST PRODUCT PREDICTIONS")
print("=" * 70)

print(results)


# ------------------------------------------------------------
# 11. MODEL EVALUATION
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
# 12. PREDICT STRENGTH FOR NEW PRODUCT
# ------------------------------------------------------------

new_product = pd.DataFrame({
    "Temperature": [82],
    "Pressure": [40],
    "Machine_Speed": [122],
    "Material_Quality": [6]
})


new_prediction = model.predict(
    new_product
)


print("\n" + "=" * 70)
print("NEW PRODUCT STRENGTH PREDICTION")
print("=" * 70)

print("\nNew Product Details:")
print(new_product)

print(
    "\nPredicted Strength Score:",
    round(new_prediction[0], 2)
)


# ------------------------------------------------------------
# 13. ACTUAL VS PREDICTED STRENGTH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Strength"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Strength"
)

plt.xlabel("Test Product")

plt.ylabel("Strength Score")

plt.title(
    "Actual vs Predicted Strength Score"
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
# 14. ACTUAL VS PREDICTED SCATTER PLOT
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

plt.xlabel("Actual Strength Score")

plt.ylabel("Predicted Strength Score")

plt.title(
    "Actual vs Predicted Strength"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. RIDGE COEFFICIENT VISUALIZATION
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

plt.xlabel("Manufacturing Feature")

plt.ylabel("Ridge Coefficient")

plt.title(
    "Ridge Regression Feature Coefficients"
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
# 16. TEMPERATURE VS STRENGTH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Temperature"],
    df["Strength_Score"],
    s=100
)

plt.xlabel("Temperature")

plt.ylabel("Strength Score")

plt.title(
    "Temperature vs Strength Score"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. PRESSURE VS STRENGTH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Pressure"],
    df["Strength_Score"],
    s=100
)

plt.xlabel("Pressure")

plt.ylabel("Strength Score")

plt.title(
    "Pressure vs Strength Score"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. MACHINE SPEED VS STRENGTH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Machine_Speed"],
    df["Strength_Score"],
    s=100
)

plt.xlabel("Machine Speed")

plt.ylabel("Strength Score")

plt.title(
    "Machine Speed vs Strength Score"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. MATERIAL QUALITY VS STRENGTH
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Material_Quality"],
    df["Strength_Score"],
    s=100
)

plt.xlabel("Material Quality")

plt.ylabel("Strength Score")

plt.title(
    "Material Quality vs Strength Score"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "Temperature",
        "Pressure",
        "Machine_Speed",
        "Material_Quality",
        "Strength_Score"
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
    "Correlation Matrix - Manufacturing Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. COMPARE DIFFERENT ALPHA VALUES
# ------------------------------------------------------------

alpha_values = [
    0.1,
    1.0,
    10.0
]

alpha_results = []

for alpha in alpha_values:

    temp_model = Ridge(
        alpha=alpha
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
# 22. ALPHA VS R2 SCORE
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
    "Effect of Alpha on Ridge Regression R²"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 23. ALPHA VS RMSE
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
    "Effect of Alpha on Ridge Regression RMSE"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 24. COMPARE RIDGE COEFFICIENTS
# ------------------------------------------------------------

coefficient_comparison = []

for alpha in alpha_values:

    temp_model = Ridge(
        alpha=alpha
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
print("RIDGE COEFFICIENT COMPARISON")
print("=" * 70)

print(coefficient_df)


# ------------------------------------------------------------
# 25. COEFFICIENT COMPARISON PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for feature in X.columns:

    plt.plot(
        alpha_values,
        coefficient_df[feature],
        marker="o",
        linewidth=2,
        label=feature
    )

plt.xlabel("Alpha")

plt.ylabel("Coefficient Value")

plt.title(
    "Effect of Alpha on Ridge Coefficients"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 26. PREDICTION ERROR PLOT
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

plt.xlabel("Test Product")

plt.ylabel("Prediction Error")

plt.title(
    "Ridge Regression Prediction Errors"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 27. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print("""
Ridge Regression is used to predict the strength score
of manufactured products.

The model uses Temperature, Pressure, Machine Speed
and Material Quality as input variables.

Temperature, Pressure and Machine Speed may be correlated.
This can create multicollinearity in ordinary linear
regression.

Ridge Regression addresses multicollinearity by adding
an L2 regularization penalty to the model.

The penalty shrinks the coefficients toward zero but
normally does not make them exactly zero.

MAE measures the average absolute prediction error.

MSE measures the average squared prediction error.

RMSE expresses the prediction error in the same unit
as the strength score.

R² indicates how well the model explains the variation
in product strength.

The manufacturing company can use the model to estimate
the strength score of a new product and understand the
effect of machine parameters on product quality.
""")
