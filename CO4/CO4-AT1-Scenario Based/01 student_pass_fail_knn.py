import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. LOAD CSV FILE
# =========================================================

df = pd.read_csv("student_pass_fail_knn.csv")

print("\nSTUDENT DATA")
print(df)

# =========================================================
# 2. NEW STUDENT
# =========================================================

# Study Hours = 5
# Attendance = 75
# Assignment Submitted = 1
# Lab Participation = 1
# Revision Done = 0

new_student = np.array([5, 75, 1, 1, 0])

features = [
    "Study_Hours",
    "Attendance",
    "Assignment_Submitted",
    "Lab_Participation",
    "Revision_Done"
]

X = df[features].values

# =========================================================
# 3. EUCLIDEAN DISTANCE
# =========================================================

euclidean = np.sqrt(
    np.sum((X - new_student) ** 2, axis=1)
)

# =========================================================
# 4. MANHATTAN DISTANCE
# =========================================================

manhattan = np.sum(
    np.abs(X - new_student), axis=1
)

# =========================================================
# 5. MINKOWSKI DISTANCE (p = 3)
# =========================================================

p = 3

minkowski = (
    np.sum(np.abs(X - new_student) ** p, axis=1)
) ** (1 / p)

# =========================================================
# 6. HAMMING DISTANCE
# =========================================================

binary_features = [
    "Assignment_Submitted",
    "Lab_Participation",
    "Revision_Done"
]

binary_data = df[binary_features].values

new_binary = new_student[2:]

hamming = np.sum(
    binary_data != new_binary,
    axis=1
)

# =========================================================
# 7. DISTANCE TABLE
# =========================================================

result = pd.DataFrame({
    "Student": df["Student"],
    "Result": df["Result"],
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
        "Student": df["Student"],
        "Result": df["Result"],
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
        nearest["Result"].value_counts()
    )

    prediction = (
        nearest["Result"]
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

x = np.arange(len(df["Student"]))
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

plt.xlabel("Student")
plt.ylabel("Distance")
plt.title("Comparison of KNN Distance Measures")

plt.xticks(
    x,
    df["Student"]
)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 2: 3 NEAREST STUDENTS
# =========================================================

nearest = nearest_euclidean

plt.figure(figsize=(8, 5))

bars = plt.bar(
    nearest["Student"],
    nearest["Distance"]
)

plt.xlabel("Nearest Students")
plt.ylabel("Euclidean Distance")

plt.title(
    "3 Nearest Students Using Euclidean Distance"
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
# GRAPH 3: BOX PLOT - STUDY HOURS
# =========================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    df["Study_Hours"]
)

plt.ylabel("Study Hours")

plt.title(
    "Distribution of Student Study Hours"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 4: BOX PLOT - ATTENDANCE BY RESULT
# =========================================================

fail_attendance = df[
    df["Result"] == "Fail"
]["Attendance"]

pass_attendance = df[
    df["Result"] == "Pass"
]["Attendance"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [fail_attendance, pass_attendance],
    labels=["Fail", "Pass"]
)

plt.ylabel("Attendance (%)")

plt.title(
    "Attendance Distribution by Result"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# =========================================================
# GRAPH 5: SCATTER PLOT
# STUDY HOURS VS ATTENDANCE
# =========================================================

plt.figure(figsize=(8, 5))

for student_result in df["Result"].unique():

    subset = df[
        df["Result"] == student_result
    ]

    plt.scatter(
        subset["Study_Hours"],
        subset["Attendance"],
        label=student_result,
        s=80
    )

# Plot new student
plt.scatter(
    new_student[0],
    new_student[1],
    marker="*",
    s=250,
    label="New Student"
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance (%)")

plt.title(
    "Study Hours vs Attendance"
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

    if prediction == "Fail":
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
    ["Fail", "Pass"]
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
    df["Result"]
    .value_counts()
)

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.xlabel("Student Result")
plt.ylabel("Number of Students")

plt.title(
    "Pass/Fail Class Distribution"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()
