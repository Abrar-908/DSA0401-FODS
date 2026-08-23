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
data = pd.read_csv("housing.csv")

# Features
X = data[[
    "Area",
    "Bedrooms",
    "Location"
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

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("Housing Price Prediction")
print("------------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# User input
area = float(input("\nEnter Area: "))
bedrooms = int(input("Enter Number of Bedrooms: "))
location = int(input("Enter Location Code: "))

new_house = [[
    area,
    bedrooms,
    location
]]

# Predict price
predicted_price = model.predict(new_house)

print("\nPredicted House Price:",
      predicted_price[0])