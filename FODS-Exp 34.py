import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Read CSV
data = pd.read_csv("car_prices.csv")

# Features
X = data[[
    "Engine_Size",
    "Horsepower",
    "Fuel_Efficiency"
]]

# Target
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("Car Price Prediction")
print("--------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# --------------------------------
# Feature influence
# --------------------------------

print("\nFeature Coefficients")
print("--------------------")

for feature, coefficient in zip(
    X.columns,
    model.coef_
):
    print(feature, ":", coefficient)

# --------------------------------
# User input
# --------------------------------

engine = float(
    input("\nEnter Engine Size: ")
)

horsepower = float(
    input("Enter Horsepower: ")
)

fuel_efficiency = float(
    input("Enter Fuel Efficiency: ")
)

new_car = [[
    engine,
    horsepower,
    fuel_efficiency
]]

predicted_price = model.predict(new_car)

print("\nPredicted Car Price:",
      predicted_price[0])