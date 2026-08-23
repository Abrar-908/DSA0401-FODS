from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset from scikit-learn
iris = load_iris()

X = iris.data
y = iris.target

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Get input from user
sepal_length = float(input("Enter Sepal Length: "))
sepal_width = float(input("Enter Sepal Width: "))
petal_length = float(input("Enter Petal Length: "))
petal_width = float(input("Enter Petal Width: "))

# Create new flower
new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Predict
prediction = model.predict(new_flower)

# Display result
print("\nPredicted Species:",
      iris.target_names[prediction[0]])