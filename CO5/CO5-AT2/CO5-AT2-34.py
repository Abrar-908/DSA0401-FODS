import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Read CSV
data = pd.read_csv("reviews.csv")

# Separate features and sentiment
X = data.drop(columns=["Sentiment"])
y = data["Sentiment"]

# Handle missing values
X = X.fillna(X.mean())

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try different perplexity values
for p in [5, 10, 30, 50]:
    tsne = TSNE(
        n_components=2,
        perplexity=p,
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_scaled)

    plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=pd.factorize(y)[0]
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Customer Reviews - t-SNE, Perplexity = " + str(p))
    plt.show()

# Final t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=pd.factorize(y)[0]
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Customer Review Visualization Using t-SNE")
plt.show()

# PCA comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pd.factorize(y)[0]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Reviews - PCA")
plt.show()