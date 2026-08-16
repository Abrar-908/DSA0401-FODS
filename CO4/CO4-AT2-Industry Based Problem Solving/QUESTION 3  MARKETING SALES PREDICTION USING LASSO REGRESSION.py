# ============================================================
# QUESTION 3
# MARKETING SALES PREDICTION USING LASSO REGRESSION
# ============================================================

# Import libraries
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
    "Month": ["M1", "M2", "M3", "M4",
              "M5", "M6", "M7", "M8"],

    "TV_Ads": [50, 60, 80, 100,
               120, 40, 90, 110],

    "Social_Media_Ads": [20, 25, 40, 60,
                         70, 15, 50, 65],

    "Newspaper_Ads": [10, 12, 15, 20,
                      22, 8, 18, 21],

    "Email_Campaigns": [5, 6, 8, 10,
                        12, 4, 9, 11],

    "Sales": [15, 18, 25, 32,
              38, 12, 28, 35]
}

df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY DATASET
# ------------------------------------------------------------

print("=" * 60)
print("MARKETING SALES DATASET")
print("=" * 60)

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
        "TV_Ads",
        "Social_Media_Ads",
        "Newspaper_Ads",
        "Email_Campaigns"
    ]
]

y = df["Sales"]


print("\nIndependent Variables:")
print(X.columns.tolist())

print("\nDependent Variable:")
print("Sales")


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

print("\nTraining Sales:")
print(y_train)


print("\n" + "=" * 60)
print("TESTING DATA")
print("=" * 60)

print(X_test)

print("\nTesting Sales:")
print(y_test)


# ------------------------------------------------------------
# 6. CREATE LASSO MODEL
# ------------------------------------------------------------

# Alpha = 0.1 as suggested in the question
alpha_value = 0.1

model = Lasso(
    alpha=alpha_value,
    max_iter=10000
)


# ------------------------------------------------------------
# 7. TRAIN THE MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("\n" + "=" * 60)
print("LASSO MODEL TRAINED")
print("=" * 60)

print("Alpha:", alpha_value)


# ------------------------------------------------------------
# 8. DISPLAY COEFFICIENTS
# ------------------------------------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nLasso Coefficients:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_)


# ------------------------------------------------------------
# 9. IDENTIFY SELECTED AND REDUCED FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

for feature, coefficient in zip(
    X.columns,
    model.coef_
):

    if abs(coefficient) < 1e-6:

        print(
            feature,
            "-> Removed by Lasso"
        )

    else:

        print(
            feature,
            "-> Selected, coefficient =",
            round(coefficient, 4)
        )


# ------------------------------------------------------------
# 10. PREDICT SALES FOR TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(X_test)


results = X_test.copy()

results["Actual Sales"] = y_test.values

results["Predicted Sales"] = y_pred

results["Error"] = (
    results["Actual Sales"]
    - results["Predicted Sales"]
)


print("\n" + "=" * 60)
print("TEST DATA PREDICTIONS")
print("=" * 60)

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
# 12. PREDICT SALES FOR NEW CAMPAIGN
# ------------------------------------------------------------

new_campaign = pd.DataFrame({
    "TV_Ads": [95],
    "Social_Media_Ads": [55],
    "Newspaper_Ads": [18],
    "Email_Campaigns": [9]
})


new_sales = model.predict(
    new_campaign
)


print("\n" + "=" * 60)
print("NEW MARKETING CAMPAIGN PREDICTION")
print("=" * 60)

print("\nNew Campaign:")
print(new_campaign)

print(
    "\nPredicted Sales:",
    round(new_sales[0], 2),
    "Lakhs"
)


# ------------------------------------------------------------
# 13. ACTUAL VS PREDICTED SALES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

x_values = np.arange(len(y_test))

plt.plot(
    x_values,
    y_test.values,
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

plt.plot(
    x_values,
    y_pred,
    marker="x",
    linewidth=2,
    label="Predicted Sales"
)

plt.xlabel("Test Month")
plt.ylabel("Sales (Lakhs)")

plt.title(
    "Actual vs Predicted Sales - Lasso Regression"
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
# 14. COEFFICIENT VISUALIZATION
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

plt.xlabel("Marketing Channel")

plt.ylabel("Lasso Coefficient")

plt.title(
    "Lasso Regression Feature Coefficients"
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
# 15. MARKETING SPENDING VS SALES
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["TV_Ads"],
    df["Sales"],
    s=100
)

plt.xlabel("TV Advertisement Spending")

plt.ylabel("Sales (Lakhs)")

plt.title(
    "TV Advertisement Spending vs Sales"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 16. CORRELATION MATRIX
# ------------------------------------------------------------

correlation = df[
    [
        "TV_Ads",
        "Social_Media_Ads",
        "Newspaper_Ads",
        "Email_Campaigns",
        "Sales"
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
    "Correlation Matrix - Marketing Dataset"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. COMPARE DIFFERENT ALPHA VALUES
# ------------------------------------------------------------

alpha_values = [0.01, 0.1, 1.0, 10.0]

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
print("ALPHA COMPARISON")
print("=" * 60)

print(alpha_df)


# ------------------------------------------------------------
# 18. ALPHA VS R2 PLOT
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
    "Effect of Alpha on Lasso Regression Performance"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. FINAL INTERPRETATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("""
Lasso Regression is a regularized linear regression
algorithm that adds an L1 penalty to the model.

The L1 penalty reduces the magnitude of coefficients
and can make some coefficients exactly zero.

A coefficient close to zero indicates that the feature
has little contribution to the prediction.

Therefore, Lasso can perform both prediction and
feature selection.

In this marketing problem, the coefficients help the
company understand the relative contribution of each
advertising channel to sales.

A lower MAE and RMSE indicate smaller prediction errors,
while a higher R² indicates better model performance.
""")
