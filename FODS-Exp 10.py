# ============================================
# Line Plot and Bar Plot using Matplotlib
# ============================================

import matplotlib.pyplot as plt

# Monthly Sales Data
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [25000, 30000, 28000, 35000, 40000, 38000]

# --------------------------------------------
# 1. Line Plot
# --------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(months, sales, marker='o', linestyle='-', color='blue')
plt.title("Monthly Sales - Line Plot")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# --------------------------------------------
# 2. Bar Plot
# --------------------------------------------
plt.figure(figsize=(8,5))
plt.bar(months, sales, color='green')
plt.title("Monthly Sales - Bar Plot")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()