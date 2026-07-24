import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\lenovo\Downloads\student_scores.csv")


print("Student Data")
print(data)


correlation = data["Study_Time"].corr(data["Exam_Score"])

print("\nCorrelation between Study Time and Exam Score:", round(correlation, 2))

# -------------------------
# Line Plot
# -------------------------
plt.figure(figsize=(6,4))
plt.plot(data["Study_Time"], data["Exam_Score"], marker='o', linewidth=2)
plt.title("Study Time vs Exam Score (Line Plot)")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# -------------------------
# Scatter Plot
# -------------------------
plt.figure(figsize=(6,4))
plt.scatter(data["Study_Time"], data["Exam_Score"], s=100)
plt.title("Study Time vs Exam Score (Scatter Plot)")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Interpretation
if correlation > 0.7:
    print("\nInsight: Strong Positive Correlation")
elif correlation > 0.3:
    print("\nInsight: Moderate Positive Correlation")
elif correlation < -0.3:
    print("\nInsight: Negative Correlation")
else:
    print("\nInsight: Weak or No Correlation")