import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Read CSV
data = pd.read_csv("customers.csv")

# Select numerical data
X = data.select_dtypes(include="number")

# Handle missing values
X = X.fillna(X.mean())

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

k_labels = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(
    n_clusters=3
)

h_labels = hierarchical.fit_predict(X_scaled)

# Function for evaluation
def evaluate(name, labels):
    silhouette = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    print("\n", name)
    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Index:", db)
    print("Calinski-Harabasz Score:", ch)

    return silhouette, db, ch

k_scores = evaluate("K-Means", k_labels)
h_scores = evaluate("Hierarchical Clustering", h_labels)

# Comparison
metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]

k_values = list(k_scores)
h_values = list(h_scores)

plt.figure(figsize=(10, 5))

x = range(len(metrics))

plt.bar(
    [i - 0.2 for i in x],
    k_values,
    width=0.4,
    label="K-Means"
)

plt.bar(
    [i + 0.2 for i in x],
    h_values,
    width=0.4,
    label="Hierarchical"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Clustering Performance Comparison")
plt.legend()
plt.show()