import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Read sensor data
data = pd.read_csv("factory_sensors.csv")

features = [
    "Temperature",
    "Pressure",
    "Vibration",
    "PowerConsumption",
    "OperatingHours",
    "ProductionOutput"
]

X = data[features]

# Missing values
X = X.fillna(X.mean())

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X_scaled)

# Detect anomalies
data["Anomaly"] = model.predict(X_scaled)

data["Status"] = data["Anomaly"].map({
    1: "Normal",
    -1: "Abnormal"
})

# Anomaly score
data["AnomalyScore"] = model.decision_function(X_scaled)

# Display abnormal machines
abnormal = data[data["Status"] == "Abnormal"]

print("Abnormal operating conditions:")
print(abnormal)

# Plot temperature and vibration
plt.scatter(
    data["Temperature"],
    data["Vibration"],
    c=data["Anomaly"]
)

plt.xlabel("Temperature")
plt.ylabel("Vibration")
plt.title("Factory Equipment Anomaly Detection")
plt.show()

# Another visualization
plt.scatter(
    data["PowerConsumption"],
    data["ProductionOutput"],
    c=data["Anomaly"]
)

plt.xlabel("Power Consumption")
plt.ylabel("Production Output")
plt.title("Power Consumption vs Production Output")
plt.show()

# Save monitoring report
data.to_csv(
    "factory_anomaly_results.csv",
    index=False
)
