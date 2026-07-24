import matplotlib.pyplot as plt

# Monthly Data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

temperature = [22, 24, 28, 32, 35, 37, 36, 35, 33, 30, 26, 23]
rainfall = [15, 20, 30, 45, 80, 120, 180, 160, 110, 70, 35, 20]

# ---------------------------
# 1. Line Plot for Temperature
# ---------------------------
plt.figure(figsize=(7,4))
plt.plot(months, temperature, marker='o', linewidth=2)
plt.title("Monthly Temperature")
plt.xlabel("Months")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()

# ---------------------------
# 2. Scatter Plot for Rainfall
# ---------------------------
plt.figure(figsize=(7,4))
plt.scatter(months, rainfall, s=100)
plt.title("Monthly Rainfall")
plt.xlabel("Months")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.show()