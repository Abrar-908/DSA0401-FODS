# ============================================================
# QUESTION 4
# ELECTRICITY CONSUMPTION PREDICTION USING RIDGE REGRESSION
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
    "House": ["H1", "H2", "H3", "H4",
              "H5", "H6", "H7", "H8"],

    "Appliances": [5, 7, 8, 10, 4, 6, 9, 11],

    "Usage_Hours": [4, 6, 7, 8, 3, 5, 7, 9],

    "House_Size": [800, 1000, 1200, 1500,
                   700, 900, 1300, 1600],

    "Occupants": [3, 4, 4, 5, 2, 3, 5, 6],

    "Consumption": [120, 190, 240, 320,
                    90, 160, 280, 350]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 60)
print("ELECTRICITY CONSUMPTION DATASET")
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
# 4. IDENTIFY FEATURES AND TARGET
# ------------------------------------------------------------

X = df[
    [
        "Appliances",
        "Usage_Hours",
        "House_Size",
        "Occupants"
    ]
]

y = df["Consumption"]


print("\nIndependent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Consumption")


# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
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
# 6. CREATE RIDGE REGRESSION MODEL
# ------------------------------------------------------------

# Alpha value required by the question
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


print("\n" + "=" * 60)
print("RIDGE REGRESSION MODEL")
print("=" * 60)

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
# 9. PREDICT TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# Create prediction table
results = X_test.copy()

results["Actual Consumption"] = y_test.values

results["Predicted Consumption"] = y_pred

results["Error"] = (
    results["Actual Consumption"]
    - results["Predicted Consumption"]
)


print("\n" + "=" * 60)
print("TEST DATA PREDICTIONS")
print("=" * 60)

print(results)


# ------------------------------------------------------------
# 10. MODEL EVALUATION
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


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Mean Absolute Error (MAE):",
      round(mae, 4))

print("Mean Squared Error (MSE):",
      round(mse, 4))

print("Root Mean Squared Error (RMSE):",
      round(rmse, 4))

print("R² Score:",
      round(r2, 4))


# ------------------------------------------------------------
# 11. PREDICT CONSUMPTION FOR NEW HOUSE
# ------------------------------------------------------------

new_house = pd.DataFrame({
    "Appliances": [8],
    "Usage_Hours": [6],
    "House_Size": [1100],
    "Occupants": [4]
})


new_prediction = model.predict(
    new_house
)


print("\n" + "=" * 60)
print("NEW HOUSE CONSUMPTION PREDICTION")
print("=" * 60)

print("\nNew House Details:")
print(new_house)

print(
    "\nPredicted Electricity Consumption:",
    round(new_prediction[0], 2),
    "Units"
)


# ------------------------------------------------------------
# 12. ACTUAL VS PREDICTED PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Consumption"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Consumption"
)

plt.xlabel("Test House")
plt.ylabel("Consumption (Units)")

plt.title(
    "Actual vs Predicted Electricity Consumption"
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
# 13. COEFFICIENT VISUALIZATION
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
# 14. CONSUMPTION VS USAGE HOURS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Usage_Hours"],
    df["Consumption"],
    s=100
)

plt.xlabel("Daily Usage Hours")

plt.ylabel("Electricity Consumption (Units)")

plt.title(
    "Usage Hours vs Electricity Consumption"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 15. CONSUMPTION VS NUMBER OF APPLIANCES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Appliances"],
    df["Consumption"],
    s=100
)

plt.xlabel("Number of Appliances")

plt.ylabel("Electricity Consumption (Units)")

plt.title(
    "Appliances vs Electricity Consumption"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "Appliances",
        "Usage_Hours",
        "House_Size",
        "Occupants",
        "Consumption"
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
    "Correlation Matrix - Electricity Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. COMPARE DIFFERENT ALPHA VALUES
# ------------------------------------------------------------

alpha_values = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
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

    temp_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            temp_prediction
        )
    )

    temp_r2 = r2_score(
        y_test,
        temp_prediction
    )

    alpha_results.append([
        alpha,
        temp_mae,
        temp_rmse,
        temp_r2
    ])


alpha_df = pd.DataFrame(
    alpha_results,
    columns=[
        "Alpha",
        "MAE",
        "RMSE",
        "R2"
    ]
)


print("\n" + "=" * 60)
print("RIDGE ALPHA COMPARISON")
print("=" * 60)

print(alpha_df)


# ------------------------------------------------------------
# 18. ALPHA VS R2 SCORE
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
    "Effect of Alpha on Ridge Regression Performance"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. ALPHA VS RMSE
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
# 20. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
Ridge Regression is a regularized form of linear
regression.

It adds an L2 penalty to the model coefficients.

This is useful when independent variables are highly
correlated, such as the number of appliances and usage
hours in this electricity consumption problem.

Ridge reduces the magnitude of the coefficients instead
of completely removing features.

The alpha parameter controls the strength of regularization.

A higher alpha produces stronger regularization.

MAE measures the average absolute prediction error.

MSE measures the average squared prediction error.

RMSE represents prediction error in the same unit as
electricity consumption.

R² measures how well the model explains the variation
in electricity consumption.
""")
