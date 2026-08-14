import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD CSV FILE
# =========================================================

df = pd.read_csv("online_customer_purchase_knn.csv")

print("\nONLINE CUSTOMER PURCHASE DATA")
print(df)

# =========================================================
# 2. NEW CUSTOMER
# =========================================================

# Age = 30
# Previous Purchases = 4
# Mobile User = 1
# Added to Cart = 1
# Used Coupon = 0

new_customer = np.array([30, 4, 1, 1, 0])

features = [
    "Age",
    "Previous_Purchases",
    "Mobile_User",
    "Added_to_Cart",
    "Used_Coupon"
]

X = df[features].values

# =========================================================
# 3. EUCLIDEAN DISTANCE
# =========================================================

euclidean = np.sqrt(
    np.sum((X - new_customer) ** 2, axis=1)
)

# =========================================================
# 4. MANHATTAN DISTANCE
# =========================================================

manhattan = np.sum(
    np.abs(X - new_customer), axis=1
)

# =========================================================
# 5. MINKOWSKI DISTANCE (p = 3)
# =========================================================

p = 3

minkowski = (
    np.sum(np.abs(X - new_customer) ** p, axis=1)
) ** (1 / p)

# =========================================================
# 6. HAMMING DISTANCE
# =========================================================

binary_features = [
    "Mobile_User",
    "Added_to_Cart",
    "Used_Coupon"
]

binary_data = df[binary_features].values

new_binary = new_customer[2:]

hamming = np.sum(
    binary_data != new_binary,
    axis=1
)

# =========================================================
# 7. DISTANCE TABLE
# =========================================================

result = pd.DataFrame({
    "Customer": df["Customer"],
    "Class": df["Class"],
    "Euclidean": euclidean,
    "Manhattan": manhattan,
    "Minkowski": minkowski,
    "Hamming": hamming
})

print("\n==============================================")
print("DISTANCE CALCULATIONS")
print("==============================================")

print(
    result.round(4).to_string(index=False)
)

# =========================================================
# 8. KNN FUNCTION
# =========================================================

def get_neighbours(distance, method):

    temp = pd.DataFrame({
        "Customer": df["Customer"],
        "Class": df["Class"],
        "Distance": distance
    })

    # K = 3
    nearest = temp.sort_values(
        by="Distance"
    ).head(3)

    print("\n==============================================")
    print(method)
    print("==============================================")

    print("\n3 Nearest Neighbours:")
    print(
        nearest.to_string(index=False)
    )

    print("\nMajority Voting:")
    print(
        nearest["Class"].value_counts()
    )

    prediction = (
        nearest["Class"]
        .value_counts()
        .idxmax()
    )

    print("\nPrediction:", prediction)

    return nearest, prediction


# =========================================================
# 9. KNN PREDICTIONS
# =========================================================

nearest_euclidean, pred_euclidean = get_neighbours(
    euclidean,
    "EUCLIDEAN DISTANCE"
)

nearest_manhattan, pred_manhattan = get_neighbours(
    manhattan,
    "MANHATTAN DISTANCE"
)

nearest_minkowski, pred_minkowski = get_neighbours(
    minkowski,
    "MINKOWSKI DISTANCE (p = 3)"
)

nearest_hamming, pred_hamming = get_neighbours(
    hamming,
    "HAMMING DISTANCE"
)

# =========================================================
# 10. FINAL PREDICTIONS
# =========================================================

predictions = {
    "Euclidean": pred_euclidean,
    "Manhattan": pred_manhattan,
    "Minkowski": pred_minkowski,
    "Hamming": pred_hamming
}

print("\n\n==============================================")
print("FINAL PREDICTIONS")
print("==============================================")

for method, prediction in predictions.items():

    print(
        f"{method:12} : {prediction}"
    )


# =========================================================
# GRAPH 1: DISTANCE COMPARISON
# =========================================================

plt.figure(figsize=(12, 6))

x = np.arange(len(df["Customer"]))
width = 0.2

plt.bar(
    x - 1.5 * width,
    euclidean,
    width,
    label="Euclidean"
)

plt.bar(
    x - 0.5 * width,
    manhattan,
    width,
    label="Manhattan"
)

plt.bar(
    x + 0.5 * width,
    minkowski,
    width,
    label="Minkowski (p=3)"
)

plt.bar(
    x + 1.5 * width,
    hamming,
    width,
    label="Hamming"
)

plt.xlabel("Customer")
plt.ylabel("Distance")
plt.title("Comparison of KNN Distance Measures")

plt.xticks(
    x,
    df["Customer"]
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 2: 3 NEAREST CUSTOMERS
# =========================================================

nearest = nearest_euclidean

plt.figure(figsize=(8, 5))

bars = plt.bar(
    nearest["Customer"],
    nearest["Distance"]
)

plt.xlabel("Nearest Customers")
plt.ylabel("Euclidean Distance")

plt.title(
    "3 Nearest Customers Using Euclidean Distance"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    nearest["Distance"]
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 3: BOX PLOT - AGE
# =========================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Age"]
)

plt.ylabel("Age")

plt.title(
    "Distribution of Customer Age"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 4: BOX PLOT - AGE BY PURCHASE CLASS
# =========================================================

no_purchase_age = df[
    df["Class"] == "No Purchase"
]["Age"]

purchase_age = df[
    df["Class"] == "Purchase"
]["Age"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [no_purchase_age, purchase_age],
    labels=["No Purchase", "Purchase"]
)

plt.ylabel("Age")

plt.title(
    "Customer Age Distribution by Purchase Class"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 5: SCATTER PLOT
# AGE VS PREVIOUS PURCHASES
# =========================================================

plt.figure(figsize=(8, 5))

for customer_class in df["Class"].unique():

    subset = df[
        df["Class"] == customer_class
    ]

    plt.scatter(
        subset["Age"],
        subset["Previous_Purchases"],
        label=customer_class,
        s=80
    )

# Plot new customer
plt.scatter(
    new_customer[0],
    new_customer[1],
    marker="*",
    s=250,
    label="New Customer"
)

plt.xlabel("Age")
plt.ylabel("Previous Purchases")

plt.title(
    "Age vs Previous Purchases"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 6: LINE PLOT - PREVIOUS PURCHASES
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df["Customer"],
    df["Previous_Purchases"],
    marker="o",
    linewidth=2
)

plt.xlabel("Customer")
plt.ylabel("Previous Purchases")

plt.title(
    "Previous Purchases by Customer"
)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 7: FINAL PREDICTION COMPARISON
# =========================================================

prediction_names = list(
    predictions.keys()
)

prediction_values = []

for prediction in predictions.values():

    if prediction == "No Purchase":
        prediction_values.append(0)
    else:
        prediction_values.append(1)

plt.figure(figsize=(9, 5))

bars = plt.bar(
    prediction_names,
    prediction_values
)

plt.xlabel("Distance Measure")
plt.ylabel("Prediction")

plt.title(
    "Final KNN Purchase Prediction Comparison"
)

plt.yticks(
    [0, 1],
    ["No Purchase", "Purchase"]
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, prediction in zip(
    bars,
    predictions.values()
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        prediction,
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 8: CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(7, 5))

class_counts = (
    df["Class"]
    .value_counts()
)

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.xlabel("Purchase Class")
plt.ylabel("Number of Customers")

plt.title(
    "Online Customer Purchase Class Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()
