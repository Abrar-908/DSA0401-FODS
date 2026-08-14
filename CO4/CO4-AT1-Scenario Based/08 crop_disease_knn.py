import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD CSV FILE
# =========================================================

df = pd.read_csv("crop_disease_knn.csv")

print("\nCROP DISEASE DATA")
print(df)

# =========================================================
# 2. NEW PLANT
# =========================================================

# Leaf Spot Count = 10
# Leaf Color Index = 50
# Yellow Leaves = 1
# White Fungus = 1
# Wilting = 0

new_plant = np.array([10, 50, 1, 1, 0])

features = [
    "Leaf_Spot_Count",
    "Leaf_Color_Index",
    "Yellow_Leaves",
    "White_Fungus",
    "Wilting"
]

X = df[features].values

# =========================================================
# 3. EUCLIDEAN DISTANCE
# =========================================================

euclidean = np.sqrt(
    np.sum((X - new_plant) ** 2, axis=1)
)

# =========================================================
# 4. MANHATTAN DISTANCE
# =========================================================

manhattan = np.sum(
    np.abs(X - new_plant), axis=1
)

# =========================================================
# 5. MINKOWSKI DISTANCE (p = 3)
# =========================================================

p = 3

minkowski = (
    np.sum(np.abs(X - new_plant) ** p, axis=1)
) ** (1 / p)

# =========================================================
# 6. HAMMING DISTANCE
# =========================================================

binary_features = [
    "Yellow_Leaves",
    "White_Fungus",
    "Wilting"
]

binary_data = df[binary_features].values

new_binary = new_plant[2:]

hamming = np.sum(
    binary_data != new_binary,
    axis=1
)

# =========================================================
# 7. CREATE DISTANCE TABLE
# =========================================================

result = pd.DataFrame({
    "Plant": df["Plant"],
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
        "Plant": df["Plant"],
        "Class": df["Class"],
        "Distance": distance
    })

    # Sort by distance and select K = 3
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
# 11. GRAPH 1 - DISTANCE COMPARISON
# =========================================================

plt.figure(figsize=(12, 6))

x = np.arange(len(df["Plant"]))
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

plt.xlabel("Plant")
plt.ylabel("Distance")
plt.title("Comparison of KNN Distance Measures")
plt.xticks(x, df["Plant"])
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# =========================================================
# 12. GRAPH 2 - EUCLIDEAN NEAREST NEIGHBOURS
# =========================================================

nearest = nearest_euclidean

plt.figure(figsize=(8, 5))

bars = plt.bar(
    nearest["Plant"],
    nearest["Distance"]
)

plt.xlabel("Nearest Plants")
plt.ylabel("Euclidean Distance")
plt.title("3 Nearest Neighbours - Euclidean Distance")

plt.grid(axis="y", alpha=0.3)

# Display values above bars
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
# 13. GRAPH 3 - FINAL PREDICTION COMPARISON
# =========================================================

prediction_names = list(predictions.keys())

prediction_values = []

for prediction in predictions.values():

    if prediction == "Healthy":
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
    ["Healthy", "Diseased"]
)

plt.grid(axis="y", alpha=0.3)

# Display prediction labels
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
# 14. GRAPH 4 - CLASS DISTRIBUTION
# =========================================================

plt.figure(figsize=(7, 5))

class_counts = df["Class"].value_counts()

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.xlabel("Plant Class")
plt.ylabel("Number of Plants")
plt.title("Crop Disease Class Distribution")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
