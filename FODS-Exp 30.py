import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Read CSV file
data = pd.read_csv("patient_symptoms.csv")

# Features
X = data[[
    "Fever",
    "Cough",
    "Fatigue",
    "BodyPain"
]]

# Target
y = data["Condition"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Input K
k = int(input("Enter value of K: "))

# Create KNN model
model = KNeighborsClassifier(n_neighbors=k)

# Train model
model.fit(X_train, y_train)

# Input new patient
fever = int(input("Fever (0/1): "))
cough = int(input("Cough (0/1): "))
fatigue = int(input("Fatigue (0/1): "))
body_pain = int(input("Body Pain (0/1): "))

new_patient = [[
    fever,
    cough,
    fatigue,
    body_pain
]]

# Scale new patient
new_patient = scaler.transform(new_patient)

# Predict
prediction = model.predict(new_patient)

print("\nPrediction:")

if prediction[0] == 1:
    print("Patient has the medical condition.")
else:
    print("Patient does not have the medical condition.")