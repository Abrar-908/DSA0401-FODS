# ============================================================
# QUESTION 5
# DELIVERY TIME PREDICTION USING LINEAR REGRESSION
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
    "Delivery": [
        "D1", "D2", "D3", "D4",
        "D5", "D6", "D7", "D8"
    ],

    "Distance_km": [
        10, 20, 30, 40,
        50, 15, 35, 25
    ],

    "Traffic_Level": [
        2, 4, 6, 8,
        9, 3, 7, 5
    ],

    "Packages": [
        20, 30, 40, 50,
        60, 25, 45, 35
    ],

    "Weather_Score": [
        8, 7, 6, 5,
        4, 8, 5, 6
    ],

    "Time_Hours": [
        1.2, 2.4, 3.5, 4.8,
        6.0, 1.8, 4.2, 3.0
    ]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 65)
print("DELIVERY TIME DATASET")
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
# 4. IDENTIFY INPUT AND OUTPUT VARIABLES
# ------------------------------------------------------------

X = df[
    [
        "Distance_km",
        "Traffic_Level",
        "Packages",
        "Weather_Score"
    ]
]

y = df["Time_Hours"]


print("\n" + "=" * 65)
print("INPUT AND OUTPUT VARIABLES")
print("=" * 65)

print("Input Variables:")
print(X.columns.tolist())

print("\nOutput Variable:")
print("Time_Hours")


# ------------------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
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


print("\n" + "=" * 65)
print("LINEAR REGRESSION MODEL")
print("=" * 65)

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
    "Delivery Time =",
    round(model.intercept_, 4),
    "+",
    round(model.coef_[0], 4),
    "* Distance",
    "+",
    round(model.coef_[1], 4),
    "* Traffic",
    "+",
    round(model.coef_[2], 4),
    "* Packages",
    "+",
    round(model.coef_[3], 4),
    "* Weather"
)


# ------------------------------------------------------------
# 10. PREDICT DELIVERY TIME FOR TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# 11. CREATE PREDICTION TABLE
# ------------------------------------------------------------

results = X_test.copy()

results["Actual Time"] = y_test.values

results["Predicted Time"] = y_pred

results["Error"] = (
    results["Actual Time"]
    - results["Predicted Time"]
)


print("\n" + "=" * 65)
print("DELIVERY TIME PREDICTIONS")
print("=" * 65)

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


print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

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
# 13. PREDICT DELIVERY TIME FOR NEW DELIVERY
# ------------------------------------------------------------

new_delivery = pd.DataFrame({
    "Distance_km": [28],
    "Traffic_Level": [6],
    "Packages": [38],
    "Weather_Score": [6]
})


new_prediction = model.predict(
    new_delivery
)


print("\n" + "=" * 65)
print("NEW DELIVERY TIME PREDICTION")
print("=" * 65)

print("\nNew Delivery Details:")
print(new_delivery)

print(
    "\nPredicted Delivery Time:",
    round(new_prediction[0], 2),
    "Hours"
)


# ------------------------------------------------------------
# 14. ACTUAL VS PREDICTED PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Time"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Time"
)

plt.xlabel("Test Delivery")
plt.ylabel("Delivery Time (Hours)")

plt.title(
    "Actual vs Predicted Delivery Time"
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

# Perfect prediction line
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

plt.xlabel("Actual Delivery Time (Hours)")

plt.ylabel("Predicted Delivery Time (Hours)")

plt.title(
    "Actual vs Predicted Delivery Time"
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

plt.xlabel("Features")

plt.ylabel("Regression Coefficient")

plt.title(
    "Feature Coefficients - Delivery Time Prediction"
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
# 17. DISTANCE VS DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Distance_km"],
    df["Time_Hours"],
    s=100
)

plt.xlabel("Distance (km)")

plt.ylabel("Delivery Time (Hours)")

plt.title(
    "Distance vs Delivery Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. TRAFFIC VS DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Traffic_Level"],
    df["Time_Hours"],
    s=100
)

plt.xlabel("Traffic Level")

plt.ylabel("Delivery Time (Hours)")

plt.title(
    "Traffic Level vs Delivery Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. PACKAGES VS DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Packages"],
    df["Time_Hours"],
    s=100
)

plt.xlabel("Number of Packages")

plt.ylabel("Delivery Time (Hours)")

plt.title(
    "Number of Packages vs Delivery Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. WEATHER SCORE VS DELIVERY TIME
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Weather_Score"],
    df["Time_Hours"],
    s=100
)

plt.xlabel("Weather Score")

plt.ylabel("Delivery Time (Hours)")

plt.title(
    "Weather Score vs Delivery Time"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 21. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "Distance_km",
        "Traffic_Level",
        "Packages",
        "Weather_Score",
        "Time_Hours"
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
    "Correlation Matrix - Delivery Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 22. PREDICTION ERROR PLOT
# ------------------------------------------------------------

errors = y_test.values - y_pred

plt.figure(figsize=(8, 5))

plt.bar(
    range(len(errors)),
    errors
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Test Delivery")

plt.ylabel("Prediction Error (Hours)")

plt.title(
    "Prediction Errors"
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

print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)

print("""
Linear Regression is used to predict delivery time
based on distance, traffic level, number of packages
and weather score.

Distance indicates how far the delivery has to travel.

Traffic Level represents the traffic conditions.

Packages represents the number of packages being
delivered.

Weather Score represents the weather condition.

MAE measures the average absolute difference between
actual and predicted delivery time.

MSE measures the average squared prediction error.

RMSE expresses the prediction error in hours.

R² indicates how well the input variables explain
the variation in delivery time.

The model can help a logistics company estimate
delivery times and improve delivery scheduling.
""")
