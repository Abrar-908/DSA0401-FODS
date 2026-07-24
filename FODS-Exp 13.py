import pandas as pd
import numpy as np

# Read stock data from CSV file
data = pd.read_csv(r"C:\Users\lenovo\Downloads\stock_data.csv")

# Display the dataset
print("Stock Data")
print(data)

# Extract Closing Prices
closing_prices = data["Close"]

# Calculate statistics
mean_price = np.mean(closing_prices)
variance = np.var(closing_prices)
std_deviation = np.std(closing_prices)

# Display results
print("\nStock Price Analysis")
print("Average Closing Price :", mean_price)
print("Variance :", variance)
print("Standard Deviation :", std_deviation)

# Interpretation
if std_deviation > 5:
    print("\nInsight: The stock prices show HIGH variability.")
else:
    print("\nInsight: The stock prices show LOW variability.")