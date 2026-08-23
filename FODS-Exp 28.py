import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\Downloads\Exp-28.csv")

print("\nCAR DATASET")
print(df.head())

# Encode categorical column
fuel_encoder = LabelEncoder()
df["Fuel_Type"] = fuel_encoder.fit_transform(df["Fuel_Type"])

# Features
X = df[["Year", "Mileage", "Fuel_Type"]]

# Target
y = df["Selling_Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CART Regression Tree
model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# User input
print("\nENTER NEW CAR DETAILS")

year = int(input("Enter car year: "))
mileage = float(input("Enter mileage (km): "))
fuel = input("Enter fuel type (Petrol/Diesel/CNG/LPG): ")

# Encode fuel
fuel_values = list(fuel_encoder.classes_)

if fuel not in fuel_values:
    print("Unknown fuel type.")
    print("Available:", fuel_values)
    exit()

fuel_encoded = fuel_encoder.transform([fuel])[0]

# Create input dataframe
new_car = pd.DataFrame({
    "Year": [year],
    "Mileage": [mileage],
    "Fuel_Type": [fuel_encoded]
})

# Predict price
predicted_price = model.predict(new_car)[0]

print("\nPREDICTION RESULT")
print("Predicted Car Price: ₹", round(predicted_price, 2))

# Decision path
leaf_id = model.apply(new_car)[0]
node_indicator = model.decision_path(new_car)

print("\nDECISION PATH")

node_index = node_indicator.indices[
    node_indicator.indptr[0]:node_indicator.indptr[1]
]

for node_id in node_index:

    if leaf_id == node_id:
        print("Reached leaf node:", node_id)
        break

    feature_id = model.tree_.feature[node_id]
    threshold = model.tree_.threshold[node_id]

    feature_name = X.columns[feature_id]
    value = new_car.iloc[0, feature_id]

    if value <= threshold:
        condition = "<="
    else:
        condition = ">"

    print(
        f"{feature_name} ({value}) "
        f"{condition} {threshold:.2f}"
    )

# Display decision tree
plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("CART Decision Tree for Car Price Prediction")
plt.show()