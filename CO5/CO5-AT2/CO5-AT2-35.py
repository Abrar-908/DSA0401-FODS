import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Read CSV
data = pd.read_csv("patients.csv")

# Optional disease/group column
if "Disease" in data.columns:
    y = data["Disease"]
    X = data.drop(columns=["Disease"])
else:
    y = None
    X = data

# Select numerical columns
X = X.select_dtypes(include="number")

# Handle missing values
X = X.fillna(X.mean())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# UMAP
reducer = umap.UMAP(
    n_components=2,
    random_state=42
)

X_umap = reducer.fit_transform(X_scaled)

# Plot UMAP
plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=pd.factorize(y)[0] if y is not None else None
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("Medical Patient Visualization Using UMAP")
plt.show()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pd.factorize(y)[0] if y is not None else None
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Medical Data - PCA")
plt.show()

# t-SNE
tsne = TSNE(
    n_components=2,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=pd.factorize(y)[0] if y is not None else None
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Medical Data - t-SNE")
plt.show()