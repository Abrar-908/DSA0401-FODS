import numpy as np

sales_data = np.array([
    [100, 120, 110],
    [150, 140, 160],
    [200, 190, 210]
])

# Calculate the average price of all products sold
average_price = np.mean(sales_data)

print("Sales Data:")
print(sales_data)

print("\nAverage Price of All Products Sold:", average_price)