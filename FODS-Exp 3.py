import numpy as np

# House data
# Columns: Bedrooms, Square Footage, Sale Price
house_data = np.array([
    [3, 1500, 250000],
    [5, 2200, 450000],
    [4, 1800, 320000],
    [6, 2800, 600000],
    [5, 2400, 500000]
])

# Select houses with more than 4 bedrooms
houses = house_data[house_data[:, 0] > 4]

# Calculate average sale price
average_price = np.mean(houses[:, 2])

print("Average Sale Price:", average_price)