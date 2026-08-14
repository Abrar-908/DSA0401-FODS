import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD CSV FILE
# =========================================================

df = pd.read_csv("fruit_knn.csv")

print("\nFRUIT DATA")
print(df)

# =========================================================
# 2. NEW FRUIT
# =========================================================

# Weight = 170
# Diameter = 8.5
# Red Color = 1
# Citrus Smell = 0
# Smooth Skin = 1

new_fruit = np.array([170, 8.5, 1, 0, 1])

features = [
    "Weight",
    "Diameter",
    "Red_Color",
    "Citrus_Smell",
    "Smooth_Skin"
]

X = df[features].values

# =========================================================
# 3. EUCLIDEAN DISTANCE
# =========================================================

euclidean = np.sqrt(
    np.sum((X - new_fruit) ** 2, axis=1)
)

# =========================================================
# 4. MANHATTAN DISTANCE
# =========================================================

manhattan = np.sum(
    np.abs(X - new_fruit), axis=1
)

# =========================================================
# 5. MINKOWSKI DISTANCE (p = 3)
# =========================================================

p = 3

minkowski = (
    np.sum(np.abs(X - new_fruit) ** p, axis=1)
) ** (1 / p)

# =========================================================
# 6. HAMMING DISTANCE
# =========================================================

binary_features = [
    "Red_Color",
    "Citrus_Smell",
    "Smooth_Skin"
]

binary_data = df[binary_features].values

new_binary = new_fruit[2:]

hamming = np.sum(
    binary_data != new_binary,
    axis=1
)

# =========================================================
# 7. DISTANCE TABLE
# =========================================================

result = pd.DataFrame({
    "Fruit": df["Fruit"],
    "Class": df["Class"],
    "Euclidean": euclidean,
    "Manhattan": manhattan,
    "Minkowski": minkowski,
    "Hamming": hamming
})

print("\n==============================================")
print("DISTANCE CALCULATIONS")
print("==============================================")

print(result.round(4).to_string(index=False))

# =========================================================
# 8. KNN FUNCTION
# =========================================================

def get_neighbours(distance, method):

    temp = pd.DataFrame({
        "Fruit": df["Fruit"],
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
    print(nearest.to_string(index=False))

    print("\nMajority Voting:")
    print(nearest["Class"].value_counts())

    prediction = nearest["Class"].value_counts().idxmax()

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
    print(f"{method:12} : {prediction}")


# =========================================================
# GRAPH 1: DISTANCE COMPARISON
# =========================================================

plt.figure(figsize=(12, 6))

x = np.arange(len(df["Fruit"]))
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

plt.xlabel("Fruit")
plt.ylabel("Distance")
plt.title("Comparison of KNN Distance Measures")
plt.xticks(x, df["Fruit"])
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 2: 3 NEAREST FRUITS
# =========================================================

nearest = nearest_euclidean

plt.figure(figsize=(8, 5))

bars = plt.bar(
    nearest["Fruit"],
    nearest["Distance"]
)

plt.xlabel("Nearest Fruits")
plt.ylabel("Euclidean Distance")
plt.title("3 Nearest Fruits Using Euclidean Distance")

plt.grid(axis="y", alpha=0.3)

for bar, value in zip(
    bars,
    nearest["Distance"]
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 3: BOX PLOT - FRUIT WEIGHT
# =========================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Weight"],
    vert=True
)

plt.ylabel("Weight (grams)")
plt.title("Distribution of Fruit Weight")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 4: BOX PLOT - WEIGHT BY FRUIT CLASS
# =========================================================

apple_weight = df[
    df["Class"] == "Apple"
]["Weight"]

orange_weight = df[
    df["Class"] == "Orange"
]["Weight"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [apple_weight, orange_weight],
    labels=["Apple", "Orange"]
)

plt.ylabel("Weight (grams)")
plt.title("Fruit Weight Distribution by Class")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 5: SCATTER PLOT - WEIGHT VS DIAMETER
# =========================================================

plt.figure(figsize=(8, 5))

for fruit_class in df["Class"].unique():

    subset = df[
        df["Class"] == fruit_class
    ]

    plt.scatter(
        subset["Weight"],
        subset["Diameter"],
        label=fruit_class,
        s=80
    )

# Plot new fruit
plt.scatter(
    new_fruit[0],
    new_fruit[1],
    marker="*",
    s=200,
    label="New Fruit"
)

plt.xlabel("Weight (grams)")
plt.ylabel("Diameter")
plt.title("Fruit Weight vs Diameter")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 6: FINAL PREDICTION COMPARISON
# =========================================================

prediction_names = list(predictions.keys())

prediction_values = []

for prediction in predictions.values():

    if prediction == "Apple":
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
plt.title("Final KNN Prediction Comparison")

plt.yticks(
    [0, 1],
    ["Apple", "Orange"]
)

plt.grid(axis="y", alpha=0.3)

for bar, prediction in zip(
    bars,
    predictions.values()
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        prediction,
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 7: CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(7, 5))

class_counts = df["Class"].value_counts()

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.xlabel("Fruit Class")
plt.ylabel("Number of Fruits")
plt.title("Fruit Class Distribution")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
