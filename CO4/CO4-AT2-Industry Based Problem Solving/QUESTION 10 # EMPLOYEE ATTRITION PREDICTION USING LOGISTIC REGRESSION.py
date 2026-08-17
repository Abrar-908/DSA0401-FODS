import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

data = {
    "Employee": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
    "Experience": [2, 3, 1, 5, 6, 8, 4, 7],
    "Satisfaction_Score": [8, 7, 4, 6, 3, 2, 7, 4],
    "Overtime_Hours": [2, 4, 10, 6, 12, 14, 3, 11],
    "Salary_Increment": [12, 10, 5, 8, 4, 3, 11, 5],
    "Leave": [0, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print(df)

X = df[["Experience", "Satisfaction_Score", "Overtime_Hours", "Salary_Increment"]]
y = df["Leave"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nActual Leave:")
print(y_test.values)

print("\nPredicted Leave:")
print(y_pred)

print("\nLeave Probability:")
print(np.round(y_prob, 2))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nLogistic Regression Coefficients:")
for feature, coefficient in zip(X.columns, model.coef_[0]):
    print(feature, ":", round(coefficient, 4))

print("\nIntercept:", round(model.intercept_[0], 4))

new_employee = pd.DataFrame({
    "Experience": [5],
    "Satisfaction_Score": [4],
    "Overtime_Hours": [9],
    "Salary_Increment": [6]
})

new_prediction = model.predict(new_employee)
new_probability = model.predict_proba(new_employee)[0][1]

print("\nNew Employee:")
print(new_employee)

print("\nLeave Probability:", round(new_probability * 100, 2), "%")

if new_prediction[0] == 1:
    print("Predicted Status: Employee May Leave")
else:
    print("Predicted Status: Employee May Stay")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stay", "Leave"]
)

disp.plot(values_format="d")
plt.title("Confusion Matrix - Employee Attrition")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(X.columns, model.coef_[0])
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Logistic Regression Coefficients")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(df["Satisfaction_Score"], df["Leave"], s=80)
plt.xlabel("Satisfaction Score")
plt.ylabel("Leave")
plt.title("Satisfaction Score vs Employee Attrition")
plt.yticks([0, 1], ["Stay", "Leave"])
plt.grid(True)
plt.show()

