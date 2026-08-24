occupancy = [80, 85, 90, 88, 92, 95, 90]

average = sum(occupancy) / len(occupancy)

print("Average Bed Occupancy:", average)
print("Predicted Occupancy for Next Week:", round(average))