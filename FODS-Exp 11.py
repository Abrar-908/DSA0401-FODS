import matplotlib.pyplot as plt

# Monthly Sales Data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [120, 150, 180, 170, 210, 250]

# ---------------------------
# 1. Line Plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.plot(months, sales, marker='o', linewidth=2)
plt.title("Monthly Sales Prediction - Line Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ---------------------------
# 2. Scatter Plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.scatter(months, sales, s=100)
plt.title("Monthly Sales Prediction - Scatter Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ---------------------------
# 3. Bar Plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.bar(months, sales)
plt.title("Monthly Sales Prediction - Bar Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.grid(axis='y')
plt.show()