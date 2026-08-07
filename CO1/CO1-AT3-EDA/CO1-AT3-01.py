# ==========================================
# Simple Exploratory Data Analysis (EDA)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Student_Performance_Dataset.csv")

# Display Dataset
print(df.head())
print(df.info())
print("\nShape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Result Distribution
print("\nResult Count")
print(df["Result"].value_counts())

# Average Values by Result
print("\nAverage by Result")
print(df.groupby("Result")[["Attendance",
                            "Study_Hours",
                            "Internal_Marks"]].mean())

# Correlation
df["Final_Result"] = df["Result"].map({"Fail":0,"Pass":1})

print("\nCorrelation Matrix")
print(df[["Attendance",
          "Study_Hours",
          "Internal_Marks",
          "Final_Result"]].corr())

# -------------------- Visualization --------------------

# Histogram
df[["Attendance","Study_Hours","Internal_Marks"]].hist(figsize=(8,5))
plt.show()

# Scatter Plot
sns.scatterplot(data=df,
                x="Attendance",
                y="Internal_Marks",
                hue="Result")
plt.show()

# Box Plot
plt.figure(figsize=(8,4))
for i, col in enumerate(["Attendance",
                         "Study_Hours",
                         "Internal_Marks"]):
    plt.subplot(1,3,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(df[["Attendance",
                "Study_Hours",
                "Internal_Marks",
                "Final_Result"]].corr(),
            annot=True,
            cmap="coolwarm")
plt.show()

# Top Students
print("\nTop 5 Students")
print(df.nlargest(5, "Internal_Marks")[["Student_ID",
                                        "Attendance",
                                        "Study_Hours",
                                        "Internal_Marks"]])

print("\nEDA Completed Successfully.")