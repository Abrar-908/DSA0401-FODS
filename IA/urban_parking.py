# ============================================================
# URBAN PARKING AVAILABILITY PREDICTION
# USING STATISTICAL AND MACHINE LEARNING TECHNIQUES
#
# Dataset:
# HDBCarparkInformation.csv
#
# IMPORTANT:
# The supplied dataset does not contain actual occupancy/
# availability values. Therefore, this implementation uses
# parking-system classification and parking characteristics
# while demonstrating the required data science techniques.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("URBAN PARKING DATA SCIENCE PROJECT")
print("=" * 70)

FILE_NAME = "HDBCarparkInformation.csv"

try:

    df = pd.read_csv(FILE_NAME)

except FileNotFoundError:

    print("\nERROR:")
    print(
        "HDBCarparkInformation.csv was not found."
    )

    print(
        "Place the CSV file in the same folder "
        "as this Python program."
    )

    raise


print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "car_park_no",
    "address",
    "x_coord",
    "y_coord",
    "car_park_type",
    "type_of_parking_system",
    "short_term_parking",
    "free_parking",
    "night_parking",
    "car_park_decks",
    "gantry_height",
    "car_park_basement"

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    print("\nMissing columns:")

    print(missing_columns)

    raise ValueError(
        "The dataset does not have the expected columns."
    )


print(
    "\nAll required columns are available."
)


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)


# ------------------------------------------------------------
# Remove duplicate car parks
# ------------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates(
    subset="car_park_no"
).copy()

after_duplicates = len(df)


print(
    "\nDuplicate records removed:",
    before_duplicates - after_duplicates
)


# ------------------------------------------------------------
# Convert numerical columns
# ------------------------------------------------------------

numeric_columns = [

    "x_coord",

    "y_coord",

    "car_park_decks",

    "gantry_height"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# ------------------------------------------------------------
# Display missing values
# ------------------------------------------------------------

print(
    "\nMissing values before cleaning:"
)

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# Fill numerical missing values
# ------------------------------------------------------------

for column in numeric_columns:

    df[column] = df[column].fillna(

        df[column].median()

    )


# ------------------------------------------------------------
# Fill categorical missing values
# ------------------------------------------------------------

categorical_columns = [

    "car_park_type",

    "type_of_parking_system",

    "short_term_parking",

    "free_parking",

    "night_parking",

    "car_park_basement"

]


for column in categorical_columns:

    df[column] = (

        df[column]

        .fillna("UNKNOWN")

        .astype(str)

        .str.strip()

    )


# ------------------------------------------------------------
# Remove invalid numerical records
# ------------------------------------------------------------

df = df[
    df["car_park_decks"] >= 0
].copy()


df = df[
    df["gantry_height"] >= 0
].copy()


# ------------------------------------------------------------
# Check missing values after cleaning
# ------------------------------------------------------------

print(
    "\nMissing values after cleaning:"
)

print(
    df.isnull().sum()
)


print(
    "\nFinal dataset shape:"
)

print(
    df.shape
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# Night parking
# ------------------------------------------------------------

df["night_parking_binary"] = (

    df["night_parking"]

    .str.upper()

    .eq("YES")

).astype(int)


# ------------------------------------------------------------
# Basement
# ------------------------------------------------------------

df["basement_binary"] = (

    df["car_park_basement"]

    .str.upper()

    .eq("Y")

).astype(int)


# ------------------------------------------------------------
# Free parking
# ------------------------------------------------------------

df["free_parking_binary"] = (

    df["free_parking"]

    .str.upper()

    .eq("YES")

).astype(int)


# ------------------------------------------------------------
# Multi-storey parking
# ------------------------------------------------------------

df["multi_storey_binary"] = (

    df["car_park_type"]

    .str.upper()

    .str.contains(
        "MULTI-STOREY",
        na=False
    )

).astype(int)


print(
    "\nNew features created:"
)

print(
    [
        "night_parking_binary",
        "basement_binary",
        "free_parking_binary",
        "multi_storey_binary"
    ]
)


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Numerical statistics
# ------------------------------------------------------------

print(
    "\nNumerical Statistics:"
)


print(
    df[
        [
            "x_coord",
            "y_coord",
            "car_park_decks",
            "gantry_height"
        ]
    ].describe()
)


# ------------------------------------------------------------
# Car park type
# ------------------------------------------------------------

print(
    "\nCar Park Type Distribution:"
)


print(
    df[
        "car_park_type"
    ].value_counts()
)


plt.figure(
    figsize=(10, 6)
)


sns.countplot(

    data=df,

    y="car_park_type"

)


plt.title(
    "Distribution of Car Park Types"
)


plt.xlabel(
    "Number of Car Parks"
)


plt.ylabel(
    "Car Park Type"
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# Parking system
# ------------------------------------------------------------

print(
    "\nParking System Distribution:"
)


print(
    df[
        "type_of_parking_system"
    ].value_counts()
)


plt.figure(
    figsize=(9, 5)
)


sns.countplot(

    data=df,

    x="type_of_parking_system"

)


plt.title(
    "Distribution of Parking Systems"
)


plt.xlabel(
    "Parking System"
)


plt.ylabel(
    "Number of Car Parks"
)


plt.xticks(
    rotation=30
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# Parking decks
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


sns.histplot(

    df["car_park_decks"],

    bins=20,

    kde=True

)


plt.title(
    "Distribution of Parking Decks"
)


plt.xlabel(
    "Number of Parking Decks"
)


plt.ylabel(
    "Number of Car Parks"
)


plt.tight_layout()


plt.show()


# ------------------------------------------------------------
# Gantry height
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)


sns.histplot(

    df["gantry_height"],

    bins=20,

    kde=True

)


plt.title(
    "Distribution of Gantry Height"
)


plt.xlabel(
    "Gantry Height"
)


plt.ylabel(
    "Number of Car Parks"
)


plt.tight_layout()


plt.show()


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)


correlation_features = [

    "x_coord",

    "y_coord",

    "car_park_decks",

    "gantry_height",

    "night_parking_binary",

    "basement_binary",

    "free_parking_binary",

    "multi_storey_binary"

]


correlation_matrix = df[
    correlation_features
].corr()


print(
    "\nCorrelation Matrix:"
)


print(
    correlation_matrix
)


plt.figure(
    figsize=(10, 7)
)


sns.heatmap(

    correlation_matrix,

    annot=True,

    fmt=".2f",

    cmap="coolwarm"

)


plt.title(
    "Correlation Matrix"
)


plt.tight_layout()


plt.show()


# ============================================================
# 8. STATISTICAL INFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL INFERENCE")
print("=" * 70)


# ------------------------------------------------------------
# POINT ESTIMATE
# ------------------------------------------------------------

decks = df[
    "car_park_decks"
].dropna()


sample_mean = decks.mean()


print(
    "\nPoint Estimate:"
)


print(
    "Mean number of parking decks =",
    round(
        sample_mean,
        4
    )
)


# ============================================================
# 9. CONFIDENCE INTERVAL
# ============================================================

sample_size = len(decks)


sample_std = decks.std(
    ddof=1
)


standard_error = (

    sample_std /

    np.sqrt(sample_size)

)


t_critical = stats.t.ppf(

    0.975,

    sample_size - 1

)


margin_error = (

    t_critical *

    standard_error

)


lower_limit = (

    sample_mean -

    margin_error

)


upper_limit = (

    sample_mean +

    margin_error

)


print(
    "\n95% Confidence Interval:"
)


print(

    f"{lower_limit:.4f} "
    f"to "
    f"{upper_limit:.4f}"

)


# ============================================================
# 10. HYPOTHESIS TESTING
# ============================================================

print("\n" + "=" * 70)
print("HYPOTHESIS TESTING")
print("=" * 70)


# H0:
# Mean number of parking decks = 2
#
# H1:
# Mean number of parking decks != 2


hypothesized_mean = 2


t_statistic, p_value = stats.ttest_1samp(

    decks,

    hypothesized_mean

)


print(
    "\nH0: Mean parking decks = 2"
)


print(
    "H1: Mean parking decks != 2"
)


print(
    "\nt-statistic =",
    round(
        t_statistic,
        4
    )
)


print(
    "p-value =",
    round(
        p_value,
        6
    )
)


if p_value < 0.05:

    print(
        "\nDecision: Reject H0"
    )

else:

    print(
        "\nDecision: Fail to reject H0"
    )


# ============================================================
# 11. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION")
print("=" * 70)


# Target:
# gantry_height
#
# The target itself is NOT used as an input feature.


regression_features = [

    "x_coord",

    "y_coord",

    "car_park_decks",

    "night_parking_binary",

    "basement_binary",

    "free_parking_binary",

    "multi_storey_binary"

]


regression_data = df[

    regression_features +

    ["gantry_height"]

].dropna()


X_reg = regression_data[
    regression_features
]


y_reg = regression_data[
    "gantry_height"
]


X_train_reg, X_test_reg, y_train_reg, y_test_reg = (

    train_test_split(

        X_reg,

        y_reg,

        test_size=0.20,

        random_state=42

    )

)


linear_model = LinearRegression()


linear_model.fit(

    X_train_reg,

    y_train_reg

)


linear_prediction = linear_model.predict(

    X_test_reg

)


linear_mae = mean_absolute_error(

    y_test_reg,

    linear_prediction

)


linear_rmse = np.sqrt(

    mean_squared_error(

        y_test_reg,

        linear_prediction

    )

)


linear_r2 = r2_score(

    y_test_reg,

    linear_prediction

)


print(
    "\nLinear Regression Results:"
)


print(
    "MAE =",
    round(
        linear_mae,
        4
    )
)


print(
    "RMSE =",
    round(
        linear_rmse,
        4
    )
)


print(
    "R² =",
    round(
        linear_r2,
        4
    )
)


# Regression graph

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_test_reg,

    linear_prediction,

    alpha=0.6

)


plt.xlabel(
    "Actual Gantry Height"
)


plt.ylabel(
    "Predicted Gantry Height"
)


plt.title(
    "Linear Regression: Actual vs Predicted"
)


plt.grid()


plt.tight_layout()


plt.show()


# ============================================================
# 12. CLASSIFICATION DATA
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION")
print("=" * 70)


print(
    "\nParking System Categories:"
)


system_counts = (

    df[
        "type_of_parking_system"
    ]

    .value_counts()

)


print(
    system_counts
)


# ------------------------------------------------------------
# Select the two most common categories
# ------------------------------------------------------------

top_classes = (

    system_counts

    .head(2)

    .index

    .tolist()

)


if len(top_classes) < 2:

    raise ValueError(

        "At least two parking-system categories "
        "are required for classification."

    )


print(
    "\nSelected classes:"
)


print(
    "Class 0 =",
    top_classes[0]
)


print(
    "Class 1 =",
    top_classes[1]
)


classification_data = df[

    df[
        "type_of_parking_system"
    ].isin(
        top_classes
    )

].copy()


# ------------------------------------------------------------
# Create target
# ------------------------------------------------------------

classification_data[
    "parking_system_target"
] = (

    classification_data[
        "type_of_parking_system"
    ]

    ==

    top_classes[0]

).astype(int)


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

classification_features = [

    "x_coord",

    "y_coord",

    "car_park_decks",

    "gantry_height",

    "night_parking_binary",

    "basement_binary",

    "free_parking_binary",

    "multi_storey_binary"

]


classification_data = (

    classification_data[

        classification_features +

        ["parking_system_target"]

    ]

    .dropna()

)


X_cls = classification_data[
    classification_features
]


y_cls = classification_data[
    "parking_system_target"
]


print(
    "\nClassification dataset size:"
)


print(
    X_cls.shape
)


print(
    "\nClass distribution:"
)


print(
    y_cls.value_counts()
)


# ============================================================
# 13. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (

    train_test_split(

        X_cls,

        y_cls,

        test_size=0.20,

        random_state=42,

        stratify=y_cls

    )

)


# ============================================================
# 14. STANDARDIZATION
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(

    X_train

)


X_test_scaled = scaler.transform(

    X_test

)


# ============================================================
# 15. EVALUATION FUNCTION
# ============================================================

def evaluate_classification_model(

    model_name,

    model,

    X_train_data,

    X_test_data,

    y_train_data,

    y_test_data

):

    model.fit(

        X_train_data,

        y_train_data

    )


    predictions = model.predict(

        X_test_data

    )


    accuracy = accuracy_score(

        y_test_data,

        predictions

    )


    precision = precision_score(

        y_test_data,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        y_test_data,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        y_test_data,

        predictions,

        zero_division=0

    )


    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1

    }, predictions


# ============================================================
# 16. TEN MODEL ITERATIONS
# ============================================================

print("\n" + "=" * 70)
print("10 MODEL ITERATIONS")
print("=" * 70)


results = []


# ------------------------------------------------------------
# ITERATION 1
# Logistic Regression
# ------------------------------------------------------------

result, pred1 = evaluate_classification_model(

    "Iteration 1 - Logistic Regression",

    LogisticRegression(

        max_iter=1000,

        C=1.0,

        random_state=42

    ),

    X_train_scaled,

    X_test_scaled,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 2
# Logistic Regression with stronger regularization
# ------------------------------------------------------------

result, pred2 = evaluate_classification_model(

    "Iteration 2 - Logistic Regression C=0.1",

    LogisticRegression(

        max_iter=1000,

        C=0.1,

        random_state=42

    ),

    X_train_scaled,

    X_test_scaled,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 3
# Logistic Regression with weaker regularization
# ------------------------------------------------------------

result, pred3 = evaluate_classification_model(

    "Iteration 3 - Logistic Regression C=10",

    LogisticRegression(

        max_iter=1000,

        C=10,

        random_state=42

    ),

    X_train_scaled,

    X_test_scaled,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 4
# kNN K=3
# ------------------------------------------------------------

result, pred4 = evaluate_classification_model(

    "Iteration 4 - kNN K=3",

    KNeighborsClassifier(

        n_neighbors=3

    ),

    X_train_scaled,

    X_test_scaled,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 5
# kNN K=5
# ------------------------------------------------------------

result, pred5 = evaluate_classification_model(

    "Iteration 5 - kNN K=5",

    KNeighborsClassifier(

        n_neighbors=5

    ),

    X_train_scaled,

    X_test_scaled,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 6
# kNN K=7
# ------------------------------------------------------------

result, pred6 = evaluate_classification_model(

    "Iteration 6 - kNN K=7",

    KNeighborsClassifier(

        n_neighbors=7

    ),

    X_train_scaled,

    X_test_scaled,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 7
# Decision Tree Depth 3
# ------------------------------------------------------------

result, pred7 = evaluate_classification_model(

    "Iteration 7 - Decision Tree Depth=3",

    DecisionTreeClassifier(

        criterion="gini",

        max_depth=3,

        random_state=42

    ),

    X_train,

    X_test,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 8
# Decision Tree Depth 5
# ------------------------------------------------------------

result, pred8 = evaluate_classification_model(

    "Iteration 8 - Decision Tree Depth=5",

    DecisionTreeClassifier(

        criterion="gini",

        max_depth=5,

        random_state=42

    ),

    X_train,

    X_test,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 9
# Decision Tree Depth 8
# ------------------------------------------------------------

result, pred9 = evaluate_classification_model(

    "Iteration 9 - Decision Tree Depth=8",

    DecisionTreeClassifier(

        criterion="gini",

        max_depth=8,

        random_state=42

    ),

    X_train,

    X_test,

    y_train,

    y_test

)


results.append(result)


# ------------------------------------------------------------
# ITERATION 10
# Decision Tree Entropy
# ------------------------------------------------------------

result, pred10 = evaluate_classification_model(

    "Iteration 10 - Decision Tree Entropy",

    DecisionTreeClassifier(

        criterion="entropy",

        max_depth=6,

        random_state=42

    ),

    X_train,

    X_test,

    y_train,

    y_test

)


results.append(result)


# ============================================================
# 17. DISPLAY 10 ITERATION RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n10 Iteration Results:"
)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. BEST ITERATION
# ============================================================

best_index = (

    results_df[
        "F1 Score"
    ]

    .idxmax()

)


best_model_name = (

    results_df.loc[

        best_index,

        "Model"

    ]

)


best_accuracy = (

    results_df.loc[

        best_index,

        "Accuracy"

    ]

)


best_f1 = (

    results_df.loc[

        best_index,

        "F1 Score"

    ]

)


print(
    "\n" + "=" * 70
)


print(
    "BEST ITERATION"
)


print(
    "=" * 70
)


print(
    "Best Model:",
    best_model_name
)


print(
    "Accuracy:",
    round(
        best_accuracy,
        4
    )
)


print(
    "F1 Score:",
    round(
        best_f1,
        4
    )
)


# ============================================================
# 19. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(14, 7)
)


plt.bar(

    results_df["Model"],

    results_df["F1 Score"]

)


plt.title(
    "F1 Score Comparison of 10 Model Iterations"
)


plt.xlabel(
    "Model Iteration"
)


plt.ylabel(
    "F1 Score"
)


plt.xticks(

    rotation=60,

    ha="right"

)


plt.ylim(
    0,
    1.05
)


plt.tight_layout()


plt.show()


# ============================================================
# 20. ACCURACY COMPARISON
# ============================================================

plt.figure(
    figsize=(14, 7)
)


plt.bar(

    results_df["Model"],

    results_df["Accuracy"]

)


plt.title(
    "Accuracy Comparison of 10 Model Iterations"
)


plt.xlabel(
    "Model Iteration"
)


plt.ylabel(
    "Accuracy"
)


plt.xticks(

    rotation=60,

    ha="right"

)


plt.ylim(
    0,
    1.05
)


plt.tight_layout()


plt.show()


# ============================================================
# 21. FINAL DECISION TREE
# ============================================================

print("\n" + "=" * 70)
print("FINAL DECISION TREE ANALYSIS")
print("=" * 70)


final_tree = DecisionTreeClassifier(

    criterion="gini",

    max_depth=6,

    min_samples_split=10,

    random_state=42

)


final_tree.fit(

    X_train,

    y_train

)


final_tree_prediction = final_tree.predict(

    X_test

)


final_tree_accuracy = accuracy_score(

    y_test,

    final_tree_prediction

)


final_tree_precision = precision_score(

    y_test,

    final_tree_prediction,

    zero_division=0

)


final_tree_recall = recall_score(

    y_test,

    final_tree_prediction,

    zero_division=0

)


final_tree_f1 = f1_score(

    y_test,

    final_tree_prediction,

    zero_division=0

)


print(
    "\nFinal Decision Tree:"
)


print(
    "Accuracy:",
    round(
        final_tree_accuracy,
        4
    )
)


print(
    "Precision:",
    round(
        final_tree_precision,
        4
    )
)


print(
    "Recall:",
    round(
        final_tree_recall,
        4
    )
)


print(
    "F1 Score:",
    round(
        final_tree_f1,
        4
    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    final_tree_prediction

)


print(
    "\nConfusion Matrix:"
)


print(
    cm
)


plt.figure(
    figsize=(6, 5)
)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)


plt.title(
    "Decision Tree Confusion Matrix"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.tight_layout()


plt.show()


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.Series(

    final_tree.feature_importances_,

    index=classification_features

).sort_values(

    ascending=False

)


print(
    "\nFeature Importance:"
)


print(
    feature_importance
)


plt.figure(
    figsize=(10, 6)
)


feature_importance.plot(
    kind="bar"
)


plt.title(
    "Decision Tree Feature Importance"
)


plt.xlabel(
    "Feature"
)


plt.ylabel(
    "Importance"
)


plt.xticks(

    rotation=45,

    ha="right"

)


plt.tight_layout()


plt.show()


# ============================================================
# 24. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)


cluster_features = [

    "x_coord",

    "y_coord",

    "car_park_decks",

    "gantry_height",

    "night_parking_binary",

    "basement_binary",

    "free_parking_binary",

    "multi_storey_binary"

]


cluster_data = df[

    cluster_features

].dropna().copy()


# ------------------------------------------------------------
# Standardize clustering data
# ------------------------------------------------------------

cluster_scaler = StandardScaler()


X_cluster = cluster_scaler.fit_transform(

    cluster_data

)


# ------------------------------------------------------------
# Test K values
# ------------------------------------------------------------

silhouette_results = {}


for k in range(

    2,

    7

):

    kmeans_test = KMeans(

        n_clusters=k,

        max_iter=10,

        n_init=10,

        random_state=42

    )


    cluster_labels_test = (

        kmeans_test.fit_predict(

            X_cluster

        )

    )


    score = silhouette_score(

        X_cluster,

        cluster_labels_test

    )


    silhouette_results[k] = score


print(
    "\nSilhouette Scores:"
)


for k, score in silhouette_results.items():

    print(

        f"K = {k} : "
        f"{score:.4f}"

    )


# ------------------------------------------------------------
# Best K
# ------------------------------------------------------------

best_k = max(

    silhouette_results,

    key=silhouette_results.get

)


print(
    "\nBest K =",
    best_k
)


# ------------------------------------------------------------
# Final K-Means
# ------------------------------------------------------------

kmeans = KMeans(

    n_clusters=best_k,

    max_iter=10,

    n_init=10,

    random_state=42

)


cluster_labels = kmeans.fit_predict(

    X_cluster

)


cluster_data[
    "cluster"
] = cluster_labels


# ------------------------------------------------------------
# Cluster summary
# ------------------------------------------------------------

print(
    "\nCluster Summary:"
)


print(

    cluster_data

    .groupby(
        "cluster"
    )

    .mean()

)


# ============================================================
# 25. SILHOUETTE GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    list(silhouette_results.keys()),

    list(silhouette_results.values()),

    marker="o"

)


plt.title(
    "Silhouette Score for K-Means"
)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Silhouette Score"
)


plt.xticks(
    range(2, 7)
)


plt.grid()


plt.tight_layout()


plt.show()


# ============================================================
# 26. K-MEANS VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)


sns.scatterplot(

    data=cluster_data,

    x="x_coord",

    y="y_coord",

    hue="cluster",

    palette="Set1",

    s=50

)


plt.title(
    "K-Means Clustering of HDB Car Parks"
)


plt.xlabel(
    "X Coordinate"
)


plt.ylabel(
    "Y Coordinate"
)


plt.tight_layout()


plt.show()


# ============================================================
# 27. FINAL MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)


print(
    "\nDataset records:",
    len(df)
)


print(
    "Mean parking decks:",
    round(
        sample_mean,
        4
    )
)


print(
    "95% CI:",
    round(
        lower_limit,
        4
    ),
    "to",
    round(
        upper_limit,
        4
    )
)


print(
    "Hypothesis test p-value:",
    round(
        p_value,
        6
    )
)


print(
    "\nLinear Regression:"
)


print(
    "MAE =",
    round(
        linear_mae,
        4
    )
)


print(
    "RMSE =",
    round(
        linear_rmse,
        4
    )
)


print(
    "R² =",
    round(
        linear_r2,
        4
    )
)


print(
    "\nBest Classification Iteration:"
)


print(
    best_model_name
)


print(
    "Best Accuracy =",
    round(
        best_accuracy,
        4
    )
)


print(
    "Best F1 Score =",
    round(
        best_f1,
        4
    )
)


print(
    "\nK-Means Best K =",
    best_k
)


print(
    "K-Means max_iter = 10"
)


# ============================================================
# 28. SAVE RESULTS
# ============================================================

df.to_csv(

    "processed_hdb_parking.csv",

    index=False

)


results_df.to_csv(

    "ten_model_iterations.csv",

    index=False

)


cluster_data.to_csv(

    "parking_clusters.csv",

    index=False

)


# Save statistical results

statistical_results = pd.DataFrame({

    "Measure": [

        "Mean Parking Decks",

        "95% CI Lower",

        "95% CI Upper",

        "t-statistic",

        "p-value"

    ],

    "Value": [

        sample_mean,

        lower_limit,

        upper_limit,

        t_statistic,

        p_value

    ]

})


statistical_results.to_csv(

    "statistical_results.csv",

    index=False

)


# Save regression results

regression_results = pd.DataFrame({

    "Metric": [

        "MAE",

        "RMSE",

        "R2"

    ],

    "Value": [

        linear_mae,

        linear_rmse,

        linear_r2

    ]

})


regression_results.to_csv(

    "linear_regression_results.csv",

    index=False

)


# ============================================================
# 29. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "PROJECT COMPLETED SUCCESSFULLY"
)

print("=" * 70)


print(
    "\nGenerated files:"
)


print(
    "1. processed_hdb_parking.csv"
)


print(
    "2. ten_model_iterations.csv"
)


print(
    "3. parking_clusters.csv"
)


print(
    "4. statistical_results.csv"
)


print(
    "5. linear_regression_results.csv"
)


print(
    "\nAll 10 model iterations were completed."
)


print(
    "K-Means was also configured with max_iter=10."
)
