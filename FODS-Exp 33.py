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

# Read CSV
data = pd.read_csv("house_prices.csv")

# --------------------------------
# Bivariate Analysis
# --------------------------------

correlation = data["House_Size"].corr(
    data["Price"]
)

print("Bivariate Analysis")
print("------------------")
print("Correlation:", correlation)

# Scatter plot
plt.scatter(
    data["House_Size"],
    data["Price"]
)

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Size vs House Price")

plt.show()

# --------------------------------
# Linear Regression
# --------------------------------

X = data[["House_Size"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# --------------------------------
# Evaluation
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nCoefficient:",
      model.coef_[0])

print("Intercept:",
      model.intercept_)

# Regression line
plt.scatter(
    data["House_Size"],
    data["Price"]
)

plt.plot(
    data["House_Size"],
    model.predict(data[["House_Size"]])
)

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Linear Regression")

plt.show()