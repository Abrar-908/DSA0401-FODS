import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read CSV file
data = pd.read_csv("financial_data.csv")

# Select numerical features
features = [
    "Open",
    "Close",
    "Volume",
    "MarketCap",
    "EarningsRatio",
    "DividendYield",
    "Volatility"
]

X = data[features]

# Handle missing values
X = X.fillna(X.mean())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio:")
for i, value in enumerate(pca.explained_variance_ratio_):
    print("PC", i + 1, ":", value)

# Cumulative variance
cumulative = pca.explained_variance_ratio_.cumsum()

print("\nCumulative Variance:")
for i, value in enumerate(cumulative):
    print("PC", i + 1, ":", value)

# Number of components for 95% variance
n_components = next(
    i + 1 for i, value in enumerate(cumulative) if value >= 0.95
)

print("\nComponents required for 95% variance:", n_components)

# Reduced dataset
pca_reduced = PCA(n_components=n_components)
X_reduced = pca_reduced.fit_transform(X_scaled)

print("\nReduced Data:")
print(X_reduced)

# Plot first two principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Financial Data - PCA")
plt.show()

# Find influential variables
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(features))],
    index=features
)

print("\nPCA Loadings:")
print(loadings)

print("\nMost influential variables:")
for pc in ["PC1", "PC2"]:
    print(pc, ":", loadings[pc].abs().sort_values(ascending=False).head(3))
