import pandas as pd

data = {
    "Customer": ["A", "B", "C", "D", "E", "F"],
    "Age": [22, 25, 22, 30, 25, 22]
}

df = pd.DataFrame(data)

frequency = df["Age"].value_counts()

print("Frequency Distribution of Customer Ages:\n")
print(frequency)