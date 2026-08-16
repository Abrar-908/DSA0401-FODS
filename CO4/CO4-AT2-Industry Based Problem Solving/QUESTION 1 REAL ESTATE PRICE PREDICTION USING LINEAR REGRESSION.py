# ============================================================
# QUESTION 1
# REAL ESTATE PRICE PREDICTION USING LINEAR REGRESSION
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
    "House": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
    "Area": [900, 1100, 1300, 1500, 1800, 2000, 1000, 1600],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 2, 3],
    "Age": [12, 10, 8, 5, 3, 2, 15, 6],
    "Distance": [10, 8, 6, 5, 3, 2, 12, 4],
    "Price": [45, 55, 68, 82, 115, 135, 40, 90]
}

df = pd.DataFrame(data)

print("================================================")
print("REAL ESTATE DATASET")
print("================================================")
print(df)


# ------------------------------------------------------------
# 2. DISPLAY BASIC INFORMATION
# ------------------------------------------------------------

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ------------------------------------------------------------
# 3. IDENTIFY FEATURES AND TARGET
# ------------------------------------------------------------

X = df[["Area", "Bedrooms", "Age", "Distance"]]
y = df["Price"]

print("\nIndependent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Price")


# ------------------------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

print("\nTraining Target:")
print(y_train)

print("\nTesting Target:")
print(y_test)


# ------------------------------------------------------------
# 5. CREATE LINEAR REGRESSION MODEL
# ------------------------------------------------------------

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# ------------------------------------------------------------
# 6. DISPLAY MODEL PARAMETERS
# ------------------------------------------------------------

print("\n================================================")
print("LINEAR REGRESSION MODEL")
print("================================================")

print("Intercept:", model.intercept_)

print("\nCoefficients:")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)


# ------------------------------------------------------------
# 7. PREDICT HOUSE PRICES FOR TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(X_test)

prediction_results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n================================================")
print("TEST DATA PREDICTIONS")
print("================================================")

print(prediction_results)


# ------------------------------------------------------------
# 8. EVALUATE THE MODEL
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n================================================")
print("MODEL EVALUATION")
print("================================================")

print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R² Score:", round(r2, 4))


# ------------------------------------------------------------
# 9. PREDICT PRICE FOR A NEW HOUSE
# ------------------------------------------------------------

new_house = pd.DataFrame({
    "Area": [1700],
    "Bedrooms": [3],
    "Age": [4],
    "Distance": [4]
})

new_price = model.predict(new_house)

print("\n================================================")
print("NEW HOUSE PRICE PREDICTION")
print("================================================")

print("New House Details:")
print(new_house)

print(
    "\nPredicted House Price:",
    round(new_price[0], 2),
    "Lakhs"
)


# ------------------------------------------------------------
# 10. ACTUAL VS PREDICTED PRICE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    range(len(y_test)),
    y_test,
    marker="o",
    s=100,
    label="Actual Price"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    marker="x",
    s=100,
    label="Predicted Price"
)

plt.xlabel("Test House")
plt.ylabel("Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 11. ACTUAL VS PREDICTED LINE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted"
)

plt.xlabel("Test House")
plt.ylabel("Price (Lakhs)")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 12. FEATURE COEFFICIENT VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    coefficients["Feature"],
    coefficients["Coefficient"]
)

plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Linear Regression Feature Coefficients")

plt.xticks(rotation=20)
plt.grid(axis="y")

plt.show()


# ------------------------------------------------------------
# 13. CORRELATION HEATMAP USING MATPLOTLIB
# ------------------------------------------------------------

correlation = df[
    ["Area", "Bedrooms", "Age", "Distance", "Price"]
].corr()

plt.figure(figsize=(8, 6))

plt.imshow(correlation, cmap="coolwarm")

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

plt.title("Correlation Matrix - Real Estate Dataset")

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 14. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n================================================")
print("INTERPRETATION")
print("================================================")

print("""
Linear Regression is used to predict house prices from
Area, Bedrooms, Age and Distance from City.

MAE represents the average absolute prediction error.

MSE represents the average squared prediction error.

RMSE represents the square root of MSE and expresses
the error in the same unit as house price.

R² indicates how well the independent variables explain
the variation in house prices.

A higher R² score and lower MAE/RMSE indicate better
prediction performance.

The trained model can be used by the real estate company
to estimate the selling price of a new house.
""")
