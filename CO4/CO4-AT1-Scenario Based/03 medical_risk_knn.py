import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD CSV FILE
# =========================================================

df = pd.read_csv("medical_risk_knn.csv")

print("\nMEDICAL RISK DATA")
print(df)

# =========================================================
# 2. NEW PATIENT
# =========================================================

# Age = 45
# Blood Sugar Level = 150
# Fever = 1
# Cough = 1
# Fatigue = 0

new_patient = np.array([45, 150, 1, 1, 0])

features = [
    "Age",
    "Blood_Sugar_Level",
    "Fever",
    "Cough",
    "Fatigue"
]

X = df[features].values

# =========================================================
# 3. EUCLIDEAN DISTANCE
# =========================================================

euclidean = np.sqrt(
    np.sum((X - new_patient) ** 2, axis=1)
)

# =========================================================
# 4. MANHATTAN DISTANCE
# =========================================================

manhattan = np.sum(
    np.abs(X - new_patient), axis=1
)

# =========================================================
# 5. MINKOWSKI DISTANCE (p = 3)
# =========================================================

p = 3

minkowski = (
    np.sum(np.abs(X - new_patient) ** p, axis=1)
) ** (1 / p)

# =========================================================
# 6. HAMMING DISTANCE
# =========================================================

binary_features = [
    "Fever",
    "Cough",
    "Fatigue"
]

binary_data = df[binary_features].values

new_binary = new_patient[2:]

hamming = np.sum(
    binary_data != new_binary,
    axis=1
)

# =========================================================
# 7. DISTANCE TABLE
# =========================================================

result = pd.DataFrame({
    "Patient": df["Patient"],
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
        "Patient": df["Patient"],
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

x = np.arange(len(df["Patient"]))
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

plt.xlabel("Patient")
plt.ylabel("Distance")
plt.title("Comparison of KNN Distance Measures")

plt.xticks(
    x,
    df["Patient"]
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 2: 3 NEAREST PATIENTS
# =========================================================

nearest = nearest_euclidean

plt.figure(figsize=(8, 5))

bars = plt.bar(
    nearest["Patient"],
    nearest["Distance"]
)

plt.xlabel("Nearest Patients")
plt.ylabel("Euclidean Distance")

plt.title(
    "3 Nearest Patients Using Euclidean Distance"
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
plt.title("Distribution of Patient Age")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 4: BLOOD SUGAR BY RISK CLASS
# =========================================================

low_risk = df[
    df["Class"] == "Low Risk"
]["Blood_Sugar_Level"]

high_risk = df[
    df["Class"] == "High Risk"
]["Blood_Sugar_Level"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [low_risk, high_risk],
    labels=["Low Risk", "High Risk"]
)

plt.ylabel("Blood Sugar Level")

plt.title(
    "Blood Sugar Distribution by Risk Class"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 5: SCATTER PLOT
# AGE VS BLOOD SUGAR
# =========================================================

plt.figure(figsize=(8, 5))

for patient_class in df["Class"].unique():

    subset = df[
        df["Class"] == patient_class
    ]

    plt.scatter(
        subset["Age"],
        subset["Blood_Sugar_Level"],
        label=patient_class,
        s=80
    )

# Plot new patient
plt.scatter(
    new_patient[0],
    new_patient[1],
    marker="*",
    s=250,
    label="New Patient"
)

plt.xlabel("Age")
plt.ylabel("Blood Sugar Level")

plt.title(
    "Age vs Blood Sugar Level"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 6: FINAL PREDICTION COMPARISON
# =========================================================

prediction_names = list(
    predictions.keys()
)

prediction_values = []

for prediction in predictions.values():

    if prediction == "Low Risk":
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
    "Final KNN Prediction Comparison"
)

plt.yticks(
    [0, 1],
    ["Low Risk", "High Risk"]
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
# GRAPH 7: CLASS DISTRIBUTION
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

plt.xlabel("Risk Class")
plt.ylabel("Number of Patients")

plt.title(
    "Medical Risk Class Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()
