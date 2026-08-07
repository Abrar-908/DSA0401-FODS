# ==================================
# Simple EDA - Hospital Patient Data
# ==================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Patient_Dataset.csv")

# Dataset Overview
print(df.head())
print(df.info())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Disease Distribution
print("\nDisease Count")
print(df["Disease_Status"].value_counts())

# Average Values by Disease Status
print("\nAverage Values")
print(df.groupby("Disease_Status")[["Age","Sugar_Level","BP","BMI"]].mean())

# Correlation
df["Disease"] = df["Disease_Status"].map({"No":0, "Yes":1})

print("\nCorrelation Matrix")
print(df[["Age","Sugar_Level","BP","BMI","Disease"]].corr())

# ---------------- Visualizations ----------------

# Histograms
df[["Age","Sugar_Level","BP","BMI"]].hist(figsize=(8,6))
plt.show()

# Disease Count
sns.countplot(x="Disease_Status", data=df)
plt.title("Disease Status")
plt.show()

# Boxplot
plt.figure(figsize=(8,4))
for i, col in enumerate(["Age","Sugar_Level","BP","BMI"]):
    plt.subplot(1,4,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Scatter Plot
sns.scatterplot(data=df,
                x="Sugar_Level",
                y="BMI",
                hue="Disease_Status")
plt.title("Sugar Level vs BMI")
plt.show()

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(df[["Age","Sugar_Level","BP","BMI","Disease"]].corr(),
            annot=True,
            cmap="coolwarm")
plt.show()

# Top Patients with Highest Sugar Level
print("\nTop 5 Patients")
print(df.nlargest(5, "Sugar_Level")[["Patient_ID",
                                     "Age",
                                     "Sugar_Level",
                                     "BP",
                                     "BMI"]])

print("\nEDA Completed Successfully.")