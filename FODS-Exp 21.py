import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

age = [23, 23, 27, 27, 39, 41, 47, 49, 50,
       52, 54, 54, 56, 57, 58, 58, 60, 61]

fat = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2,
       34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]

df = pd.DataFrame({
    "Age": age,
    "%Fat": fat
})

print("Mean")
print(df.mean())

print("\nMedian")
print(df.median())

print("\nStandard Deviation")
print(df.std())

plt.figure(figsize=(8,4))
df.boxplot(column=["Age", "%Fat"])
plt.title("Boxplot of Age and %Fat")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Age"], df["%Fat"])
plt.xlabel("Age")
plt.ylabel("%Fat")
plt.title("Scatter Plot")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
stats.probplot(df["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")
plt.show()

plt.figure(figsize=(6,4))
stats.probplot(df["%Fat"], dist="norm", plot=plt)
plt.title("Q-Q Plot of %Fat")
plt.show()