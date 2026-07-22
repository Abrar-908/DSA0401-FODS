import numpy as np

# Fuel efficiency (miles per gallon) of different car models
fuel_efficiency = np.array([25, 30, 28, 35, 40])

# Calculate average fuel efficiency
average_efficiency = np.mean(fuel_efficiency)

# Compare Model 1 and Model 5
old_model = fuel_efficiency[0]
new_model = fuel_efficiency[4]

# Calculate percentage improvement
percentage_improvement = ((new_model - old_model) / old_model) * 100

print("Fuel Efficiency (MPG):", fuel_efficiency)
print("Average Fuel Efficiency:", average_efficiency)
print("Percentage Improvement from Model 1 to Model 5: {:.2f}%".format(percentage_improvement))