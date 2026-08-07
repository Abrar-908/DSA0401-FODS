import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Student-AT2.csv")

print(df)
print("\nFirst 5 Records")
print(df.head())

print("\nLast 5 Records")
print(df.tail())

print("\nMissing Values")
print(df.isnull().sum())

df = df.drop_duplicates()

cols = ["Attendance", "Assignment_Marks", "Internal_Marks", "Marks"]
df[cols] = df[cols].fillna(df[cols].mean())

df["Result"] = np.where(df["Marks"] >= 40, "Pass", "Fail")

print("\nStatistics")
print("Mean :", df["Marks"].mean())
print("Median :", df["Marks"].median())
print("Mode :", df["Marks"].mode()[0])
print("Std Dev :", df["Marks"].std())
print("Variance :", df["Marks"].var())
print("Maximum :", df["Marks"].max())
print("Minimum :", df["Marks"].min())

print("\nDepartment Wise Average")
print(df.groupby("Department")["Marks"].mean())

print("\nSemester Wise Average")
print(df.groupby("Semester")["Marks"].mean())

print("\nPass / Fail")
print(df["Result"].value_counts())

print("\nPass Percentage : {:.2f}%".format((df["Result"]=="Pass").mean()*100))
print("Fail Percentage : {:.2f}%".format((df["Result"]=="Fail").mean()*100))

print("\nTop Performer")
print(df.loc[df["Marks"].idxmax()])

print("\nAttendance Below 75")
print(df[df["Attendance"]<75])

sns.set_style("whitegrid")

plt.figure(figsize=(10,5))
plt.bar(df["Name"], df["Marks"])
plt.title("Student Marks")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df["Name"], df["Attendance"], marker="o")
plt.title("Attendance")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(5,5))
df["Result"].value_counts().plot.pie(autopct="%1.1f%%")
plt.ylabel("")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(df["Marks"], bins=10)
plt.title("Marks Distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Attendance"], df["Marks"])
plt.title("Attendance vs Marks")
plt.show()

sns.boxplot(y=df["Marks"])
plt.show()

sns.countplot(x="Department", data=df)
plt.show()

sns.countplot(x="Result", data=df)
plt.show()

sns.barplot(x="Department", y="Marks", data=df)
plt.show()

sns.boxplot(x="Department", y="Marks", data=df)
plt.show()

sns.scatterplot(x="Attendance", y="Marks", hue="Department", data=df)
plt.show()

sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm")
plt.show()

print("\nAnalysis Completed Successfully.")
