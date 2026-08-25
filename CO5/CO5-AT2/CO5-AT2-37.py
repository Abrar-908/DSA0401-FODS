import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Read CSV
data = pd.read_csv("insurance_claims.csv")

# Select numerical features
features = [
    "ClaimAmount",
    "ClaimFrequency",
    "CustomerAge",
    "PolicyDuration",
    "AccidentHistory",
    "SettlementAmount"
]

X = data[features]

# Handle missing values
X = X.fillna(X.median())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# Predictions
data["AnomalyLabel"] = model.predict(X_scaled)

# Convert labels
data["FraudStatus"] = data["AnomalyLabel"].map({
    1: "Normal",
    -1: "Suspicious"
})

# Anomaly score
data["AnomalyScore"] = model.decision_function(X_scaled)

# Display suspicious claims
suspicious = data[data["FraudStatus"] == "Suspicious"]

print("Suspicious Claims:")
print(suspicious)

print("\nNumber of suspicious claims:",
      len(suspicious))

# Visualization
plt.scatter(
    data["ClaimAmount"],
    data["SettlementAmount"],
    c=data["AnomalyLabel"]
)

plt.xlabel("Claim Amount")
plt.ylabel("Settlement Amount")
plt.title("Insurance Claim Anomaly Detection")
plt.show()

# Save results
data.to_csv(
    "insurance_fraud_results.csv",
    index=False
)