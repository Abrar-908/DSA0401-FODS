import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD CSV FILE
# =========================================================

df = pd.read_csv("house_price_knn.csv")

print("\nHOUSE PRICE DATA")
print(df)

# =========================================================
# 2. NEW HOUSE
# =========================================================

# Area = 1400
# Bedrooms = 3
# Parking Available = 1
# Near Main Road = 1
# Garden Available = 0

new_house = np.array([1400, 3, 1, 1, 0])

features = [
    "Area",
    "Bedrooms",
    "Parking_Available",
    "Near_Main_Road",
    "Garden_Available"
]

X = df[features].values

# =========================================================
# 3. EUCLIDEAN DISTANCE
# =========================================================

euclidean = np.sqrt(
    np.sum((X - new_house) ** 2, axis=1)
)

# =========================================================
# 4. MANHATTAN DISTANCE
# =========================================================

manhattan = np.sum(
    np.abs(X - new_house), axis=1
)

# =========================================================
# 5. MINKOWSKI DISTANCE (p = 3)
# =========================================================

p = 3

minkowski = (
    np.sum(np.abs(X - new_house) ** p, axis=1)
) ** (1 / p)

# =========================================================
# 6. HAMMING DISTANCE
# =========================================================

binary_features = [
    "Parking_Available",
    "Near_Main_Road",
    "Garden_Available"
]

binary_data = df[binary_features].values

new_binary = new_house[2:]

hamming = np.sum(
    binary_data != new_binary,
    axis=1
)

# =========================================================
# 7. DISTANCE TABLE
# =========================================================

result = pd.DataFrame({
    "House": df["House"],
    "Category": df["Category"],
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
        "House": df["House"],
        "Category": df["Category"],
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
    print(nearest["Category"].value_counts())

    prediction = nearest["Category"].value_counts().idxmax()

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

x = np.arange(len(df["House"]))
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

plt.xlabel("House")
plt.ylabel("Distance")
plt.title("Comparison of KNN Distance Measures")
plt.xticks(x, df["House"])
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 2: 3 NEAREST HOUSES
# =========================================================

nearest = nearest_euclidean

plt.figure(figsize=(8, 5))

bars = plt.bar(
    nearest["House"],
    nearest["Distance"]
)

plt.xlabel("Nearest Houses")
plt.ylabel("Euclidean Distance")
plt.title("3 Nearest Houses Using Euclidean Distance")

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
# GRAPH 3: BOX PLOT - HOUSE AREA
# =========================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Area"],
    vert=True
)

plt.ylabel("Area (sq ft)")
plt.title("Distribution of House Area")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 4: BOX PLOT - AREA BY CATEGORY
# =========================================================

low_price_area = df[
    df["Category"] == "Low Price"
]["Area"]

high_price_area = df[
    df["Category"] == "High Price"
]["Area"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [low_price_area, high_price_area],
    labels=["Low Price", "High Price"]
)

plt.ylabel("Area (sq ft)")
plt.title("House Area Distribution by Price Category")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 5: LINE PLOT - AREA VS BEDROOMS
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df["House"],
    df["Area"],
    marker="o",
    label="Area"
)

plt.plot(
    df["House"],
    df["Bedrooms"] * 400,
    marker="s",
    label="Bedrooms × 400"
)

plt.xlabel("House")
plt.ylabel("Value")
plt.title("House Area and Bedroom Comparison")

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

    if prediction == "Low Price":
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
    ["Low Price", "High Price"]
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

class_counts = df["Category"].value_counts()

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.xlabel("Price Category")
plt.ylabel("Number of Houses")
plt.title("House Price Category Distribution")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
