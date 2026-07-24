import pandas as pd

# Read the dataset
data = pd.read_csv(r"C:\Users\lenovo\Downloads\temperature_data.csv")

# Mean temperature for each city
mean_temperature = data.groupby("City")["Temperature"].mean()

# Standard deviation for each city
std_temperature = data.groupby("City")["Temperature"].std()

# Temperature range for each city
temperature_range = data.groupby("City")["Temperature"].agg(lambda x: x.max() - x.min())

# Display results
print("Mean Temperature for Each City")
print(mean_temperature)

print("\nStandard Deviation for Each City")
print(std_temperature)

print("\nTemperature Range for Each City")
print(temperature_range)

# City with highest temperature range
highest_range_city = temperature_range.idxmax()

# City with most consistent temperature
most_consistent_city = std_temperature.idxmin()

print("\nCity with Highest Temperature Range :", highest_range_city)
print("Temperature Range :", temperature_range.max())

print("\nCity with Most Consistent Temperature :", most_consistent_city)
print("Standard Deviation :", round(std_temperature.min(), 2))