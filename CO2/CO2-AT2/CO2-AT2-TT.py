import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_data_with_fail.csv")

print("\nSTUDENT PERFORMANCE DATA\n")
print(df)

print("\nFIRST 5 RECORDS\n")
print(df.head())

print("\nLAST 5 RECORDS\n")
print(df.tail())

print("\nDATA INFORMATION\n")
print(df.info())

print("\nMISSING VALUES\n")
print(df.isnull().sum())

df = df.drop_duplicates()

df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Assignment_Marks"] = df["Assignment_Marks"].fillna(df["Assignment_Marks"].mean())
df["Internal_Marks"] = df["Internal_Marks"].fillna(df["Internal_Marks"].mean())
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())

df["Result"] = np.where(df["Marks"] >= 40, "Pass", "Fail")

marks = df["Marks"].to_numpy()
attendance = df["Attendance"].to_numpy()

print("\nSTATISTICAL ANALYSIS\n")

print("Mean Marks :", np.mean(marks))
print("Median Marks :", np.median(marks))
print("Standard Deviation :", np.std(marks))
print("Variance :", np.var(marks))
print("Maximum Marks :", np.max(marks))
print("Minimum Marks :", np.min(marks))

print("\nAttendance Statistics\n")

print("Mean Attendance :", np.mean(attendance))
print("Maximum Attendance :", np.max(attendance))
print("Minimum Attendance :", np.min(attendance))

print("\nMode of Marks")
print(df["Marks"].mode())

print("\nDepartment Wise Average Marks")
print(df.groupby("Department")["Marks"].mean())

print("\nSemester Wise Average Marks")
print(df.groupby("Semester")["Marks"].mean())

total_students = len(df)
passed_students = len(df[df["Result"] == "Pass"])
failed_students = len(df[df["Result"] == "Fail"])

pass_percentage = (passed_students / total_students) * 100
fail_percentage = (failed_students / total_students) * 100

print("\nTotal Students :", total_students)
print("Passed Students :", passed_students)
print("Failed Students :", failed_students)
print("Pass Percentage : {:.2f}%".format(pass_percentage))
print("Fail Percentage : {:.2f}%".format(fail_percentage))

print("\nTop Performer")
print(df.loc[df["Marks"].idxmax()])

print("\nStudents with Attendance Below 75%")
print(df[df["Attendance"] < 75])

sns.set_style("whitegrid")

plt.figure(figsize=(12,6))
plt.bar(df["Name"], df["Marks"])
plt.title("Student Marks")
plt.xlabel("Student Name")
plt.ylabel("Marks")
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df["Name"], df["Attendance"], marker='o')
plt.title("Attendance of Students")
plt.xlabel("Student Name")
plt.ylabel("Attendance")
plt.xticks(rotation=90)
plt.show()

result = df["Result"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(result.values, labels=result.index, autopct="%1.1f%%", startangle=90)
plt.title("Pass and Fail Percentage")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Marks"], bins=10)
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["Attendance"], df["Marks"])
plt.title("Attendance vs Marks")
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.show()

plt.figure(figsize=(6,5))
sns.boxplot(y=df["Marks"])
plt.title("Box Plot of Marks")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x="Department", data=df)
plt.title("Department Wise Student Count")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x="Result", data=df)
plt.title("Pass and Fail Count")
plt.show()

plt.figure(figsize=(6,5))
sns.barplot(x="Department", y="Marks", data=df)
plt.title("Average Marks by Department")
plt.show()

plt.figure(figsize=(6,5))
sns.boxplot(x="Department", y="Marks", data=df)
plt.title("Department Wise Marks Distribution")
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(x="Attendance", y="Marks", hue="Department", data=df)
plt.title("Attendance vs Marks by Department")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df[["Attendance", "Assignment_Marks", "Internal_Marks", "Marks"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nAnalysis Completed Successfully.")